"""Search for reaction paths through a microkinetic network.

A path is a set of elementary reactions (with a direction) whose linear
combination produces a requested overall reaction.  The search walks forward
from the initial gas/site state and records paths that reach a state without
surface species and whose stoichiometry reproduces one of the total reactions.

The original script needed the user to supply *auto-catalytic species* (species
that appear both as reactants and products).  They are now detected
automatically; supplying ``auto_catalytic_species`` still overrides the
automatic detection, and an explicit empty list disables it.

Two search modes are supported:

* one search with one ``auto_catalytic_species`` list (the classic mode), and
* a sharded search that runs one search per seed group and merges the
  results (:func:`find_reaction_paths_sharded`).  With one species per group
  this is the old "one job per intermediate" strategy, and sharding is both a
  strong pruning strategy and a natural way to parallelise the search.

Some pathways require more than one species to be present initially, and a
gas product in the initial state changes which reactions can be used later in
the walk.  The sharded API therefore also accepts explicit multi-species seed
groups, e.g. ``shards=[["CH3#", "CO#", "H#"]]``, and gas species such as
``"CH3OH"`` or ``"CH2CO"``.  ``shards="all"`` enumerates surface intermediates
and gas species.
"""

from __future__ import annotations

import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from random import Random, shuffle
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .model import MKM

# A coefficient is treated as zero when its absolute value is below this
# threshold.  The same threshold is used by the old scripts and by the path
# de-duplication below.
ZERO_THRESHOLD = 0.01


@dataclass
class ReactionPathResult:
    """Result of :func:`find_reaction_paths`.

    Attributes
    ----------
    paths:
        ``(npaths, nreactions)`` array.  ``paths[p, r]`` is the signed
        stoichiometric coefficient of elementary reaction ``r`` in path ``p``.
    index:
        For each total reaction, the slice of rows in :attr:`paths` that
        belongs to it.
    total_reactions:
        String representation of the requested overall reactions.
    auto_catalytic_species:
        Species used to seed the search.
    stats:
        Search statistics.  For a sharded search ``stats["shards"]`` contains
        the statistics of the individual shards.
    """

    paths: np.ndarray
    index: List[np.ndarray]
    total_reactions: List[str] = field(default_factory=list)
    auto_catalytic_species: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    # -- tuple-like compatibility with the old scripts --------------------
    def __iter__(self):
        yield self.paths
        yield self.index

    def __getitem__(self, item):
        return (self.paths, self.index)[item]

    def __len__(self):
        return 2

    @classmethod
    def from_legacy(cls, data, total_reactions=None,
                    auto_catalytic_species=None) -> "ReactionPathResult":
        """Build a result from an old ``[paths, index]`` pickle."""
        paths, index = data
        arr = np.asarray(paths, dtype=float)
        return cls(
            paths=arr,
            index=[np.asarray(i, dtype=int) for i in index],
            total_reactions=list(total_reactions or []),
            auto_catalytic_species=list(auto_catalytic_species or []),
            stats={},
        )


def detect_autocatalytic_species(mkm: MKM) -> List[str]:
    """Return surface species that appear both as reactant and as product.

    Bare sites (``#``, ``#1`` ...) are excluded; they are always present in
    the initial state anyway.
    """
    consumed, produced = set(), set()
    for reaction in mkm.reaction_list:
        consumed.update(a.name for a in reaction.reactants if a.nsites != 0)
        produced.update(a.name for a in reaction.products if a.nsites != 0)
    species = consumed & produced
    return sorted(s for s in species if not re.fullmatch(r'#\d*', s))


def list_shard_species(mkm: MKM, include_gas: bool = False) -> List[str]:
    """Return candidate initial species, in network declaration order.

    By default this is the set of non-bare surface "intermediates" used by the
    old per-species workflow.  With ``include_gas=True`` the gas species are
    appended as well; the legacy jobs enumerated every species in the network,
    and a gas species can seed a pathway through its adsorption steps.

    Every returned name can be supplied as the initial auto-catalytic species
    of a separate shard search.
    """
    species = [s for s in mkm.surface_species if not re.fullmatch(r'#\d*', s)]
    if include_gas:
        species.extend(s for s in mkm.gas_species if s not in species)
    return species


# ---------------------------------------------------------------------------
# Merging shard results
# ---------------------------------------------------------------------------


def _path_signature(row: np.ndarray):
    """Canonical (support, signs, nnz) signature of a path row.

    The support is the set of reactions with a non-zero coefficient.  The sign
    mask distinguishes paths that use the same reactions in opposite net
    directions.  Magnitudes are intentionally ignored: for a fixed overall
    reaction the magnitudes are determined by the linear system, and
    different shards may solve it with tiny numerical differences.
    """
    support = 0
    signs = 0
    nnz = 0
    for j in np.where(np.abs(row) > ZERO_THRESHOLD)[0]:
        j = int(j)
        support |= 1 << j
        nnz += 1
        if row[j] < 0:
            signs |= 1 << j
    return support, signs, nnz


def _coerce_result(item) -> ReactionPathResult:
    if isinstance(item, ReactionPathResult):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        # Old pickles store ``[paths, index]``.
        return ReactionPathResult.from_legacy(item)
    raise TypeError(
        "Expected a ReactionPathResult or an old [paths, index] pair, "
        f"got {type(item).__name__}.")


def merge_reaction_paths(results: Sequence[ReactionPathResult],
                         deduplicate: bool = True) -> ReactionPathResult:
    """Merge several :class:`ReactionPathResult` objects into one.

    Parameters
    ----------
    results:
        Results from independent shard searches.  They must describe the same
        elementary network and the same overall reactions.
    deduplicate:
        Drop paths with the same total reaction, reaction support and sign
        pattern.  This is what makes the union of per-species shard searches
        usable.
    """
    results = [_coerce_result(r) for r in results]
    if not results:
        raise ValueError("merge_reaction_paths needs at least one result.")

    first = results[0]
    n_reactions = first.paths.shape[1] if first.paths.ndim == 2 else 0
    n_groups = max(
        [len(first.total_reactions), len(first.index)]
        + [len(r.total_reactions) for r in results]
        + [len(r.index) for r in results]
    )

    kept: List[np.ndarray] = []
    kept_group: List[int] = []
    kept_nnz: List[int] = []
    kept_support: List[int] = []
    kept_signs: List[int] = []
    seen = set()

    for result in results:
        if result.paths.ndim != 2 or result.paths.shape[1] != n_reactions:
            raise ValueError("Cannot merge results with different reaction "
                             "networks.")
        row_group = np.full(len(result.paths), -1, dtype=int)
        for gi, ind in enumerate(result.index):
            ind = np.asarray(ind, dtype=int)
            if ind.size:
                row_group[ind] = gi
        if len(result.index) == 0 and len(result.paths):
            row_group[:] = 0
        for row, gi in zip(result.paths, row_group):
            if gi < 0:
                # A result produced before the index was padded for empty
                # groups: treat it as belonging to the first total reaction.
                gi = 0
            support, signs, nnz = _path_signature(row)
            key = (int(gi), support, signs)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            kept.append(np.asarray(row, dtype=float).copy())
            kept_group.append(int(gi))
            kept_nnz.append(nnz)
            kept_support.append(support)
            kept_signs.append(signs)

    if kept:
        order = sorted(
            range(len(kept)),
            key=lambda i: (kept_group[i], kept_nnz[i],
                           kept_support[i], kept_signs[i]),
        )
        paths = np.asarray([kept[i] for i in order], dtype=float)
        groups = np.asarray([kept_group[i] for i in order], dtype=int)
    else:
        paths = np.empty((0, n_reactions), dtype=float)
        groups = np.empty((0,), dtype=int)

    index: List[np.ndarray] = []
    for gi in range(n_groups):
        index.append(np.where(groups == gi)[0])

    auto_species: List[str] = []
    for result in results:
        for species in result.auto_catalytic_species:
            if species not in auto_species:
                auto_species.append(species)

    total_reactions = [str(r) for r in first.total_reactions]
    if not total_reactions:
        total_reactions = [str(r) for r in results[-1].total_reactions]

    sum_keys = (
        "n_walk_node", "n_check_rank", "n_meet_end", "n_noraml",
        "n_dead_end", "n_path_unchanged", "n_solved_path",
    )
    stats: Dict[str, object] = {
        key: sum(int(r.stats.get(key, 0)) for r in results)
        for key in sum_keys
    }
    stats["n_shards"] = len(results)
    stats["n_unique_paths"] = len(paths)
    stats["shards"] = [dict(r.stats) for r in results]

    return ReactionPathResult(
        paths=paths,
        index=index,
        total_reactions=total_reactions,
        auto_catalytic_species=auto_species,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def find_reaction_paths(
    mkm: MKM,
    mkm_gas: Optional[MKM] = None,
    find_reverse: bool = False,
    auto_catalytic_species: Union[None, str, Sequence[str]] = None,
    center_species: Optional[Sequence[str]] = None,
    check_rank: bool = False,
    max_walk_node: int = -1,
    print_path: bool = False,
    save_depth: int = 2,
    nstat: int = -1,
    nwait: int = -1,
    verbose: bool = True,
    on_path: Optional[Callable[[np.ndarray, int], None]] = None,
    store_paths: bool = True,
    random_seed: Optional[int] = None,
) -> ReactionPathResult:
    """Exhaustively (or randomly, if ``max_walk_node > 0``) search paths.

    Parameters
    ----------
    mkm:
        Model whose ``reaction_list`` describes the elementary steps.
    mkm_gas:
        Model whose ``reaction_list`` describes the requested overall
        reactions.  If omitted, the same reactions as *mkm* are used (useful
        when only one overall reaction is wanted).
    auto_catalytic_species:
        Species that are allowed to be present initially.  Both surface
        species and gas species are valid (a gas product seeds the reverse of
        the elementary steps that form it).  ``None`` (default) auto-detects
        species that are both consumed and produced.  Pass ``[]`` to disable.
    center_species:
        Species that identify a different branch; a reaction that produces
        an already-present centre species is not added to the current path.
    on_path:
        Optional callback ``on_path(coefficients, total_reaction_index)`` that
        is called for every newly found path.  Useful for streaming very large
        searches to disk.
    store_paths:
        If False, paths are not accumulated in memory; use :arg:`on_path` to
        consume them.  The returned ``paths`` array is then empty while the
        statistics still report the number of paths found.
    """
    if mkm_gas is None:
        mkm_gas = mkm
    if auto_catalytic_species is None:
        auto_catalytic_species = detect_autocatalytic_species(mkm)
    elif isinstance(auto_catalytic_species, str):
        auto_catalytic_species = [auto_catalytic_species]
    else:
        auto_catalytic_species = list(auto_catalytic_species)
    if isinstance(center_species, str):
        center_species = [center_species]
    center_species = list(center_species or [])
    rng = Random(random_seed) if random_seed is not None else None

    stat_dict = {
        "n_path_unchanged": 0,
        "n_solved_path": 0,
        "n_walk_node": 0,
        "n_check_rank": 0,
        "i_print_walk": 0,
        "n_meet_end": 0,
        "n_noraml": 0,
        "n_dead_end": 0,
        "stack_depth": 0,
        "target_depth": -1,
        "stack_depth_dict": {},
    }

    good_path_set = set()
    good_path_index = []
    solved_path = []
    path_state_stack = deque()
    nbrach_stack = deque()

    def print_stat_info():
        nsolved_path = len(solved_path) if store_paths else \
            stat_dict["n_solved_path"]
        if nsolved_path == stat_dict["n_solved_path"]:
            stat_dict["n_path_unchanged"] += 1
        else:
            stat_dict["n_path_unchanged"] = 0
            stat_dict["n_solved_path"] = nsolved_path

        if nwait > 0 and stat_dict["n_path_unchanged"] >= nwait:
            stat_dict["target_depth"] = min(stat_dict["stack_depth_dict"].keys()) - 1
        else:
            stat_dict["target_depth"] = -1

        print("Total step:", stat_dict["n_walk_node"])
        print("Total check rank:", stat_dict["n_check_rank"])
        print("Found path:", nsolved_path)
        print("Normal pass:", stat_dict["n_noraml"],
              "Dead end:", stat_dict["n_dead_end"],
              "Meet end:", stat_dict["n_meet_end"])
        depth_count = list(stat_dict["stack_depth_dict"].items())
        print("Depth count:", ", ".join(f"{a}:{b}" for a, b in sorted(depth_count)))
        brach_count = [(i, nbrach_stack[i]) for i in range(len(nbrach_stack))]
        print("Search brach:", ", ".join(f"{a}:{b}" for a, b in brach_count))
        print("\n", flush=True)

    def walk_node(path_state: list, path: np.ndarray, current_state: int,
                  previous_state: int):
        path_state_stack.append(set())
        nbrach_stack.append(0)

        stat_dict["stack_depth"] += 1
        stat_dict["i_print_walk"] += 1
        stat_dict["n_walk_node"] += 1
        if nstat > 0 and stat_dict["i_print_walk"] >= nstat:
            print_stat_info()
            stat_dict["i_print_walk"] = 0
            stat_dict["stack_depth_dict"].clear()
            stat_dict["n_dead_end"] = 0
            stat_dict["n_meet_end"] = 0
            stat_dict["n_noraml"] = 0
        stat_dict["stack_depth_dict"][stat_dict["stack_depth"]] = \
            stat_dict["stack_depth_dict"].get(stat_dict["stack_depth"], 0) + 1

        # No surface species left in the current state: try to close the path.
        if np.any(path) and not (current_state & surface_mask):
            stat_dict["n_meet_end"] += 1
            total_state = current_state | previous_state
            total_reac_index = [
                i for i, mask in enumerate(gas_reactant_masks)
                if (mask & total_state) == mask
            ]
            coeff = tot_species_coeff[path, :]
            for i_tot_reac in total_reac_index:
                coeff_1 = np.vstack((coeff, gas_species_coeff[i_tot_reac, :]))
                rank_1 = np.linalg.matrix_rank(coeff_1)
                stat_dict["n_check_rank"] += 1
                if rank_1 == coeff.shape[0]:
                    Q, R = np.linalg.qr(coeff.T)
                    c = np.linalg.solve(
                        R, np.matmul(gas_species_coeff[i_tot_reac, :], Q))
                    reac_c = np.zeros((n_reactions,))
                    reac_c[path] = c
                    support, signs, nnz = _path_signature(reac_c)
                    item = (int(i_tot_reac), support, signs)
                    if item in good_path_set:
                        continue
                    good_path_set.add(item)
                    stat_dict["n_solved_path"] += 1
                    if on_path is not None:
                        on_path(reac_c, int(i_tot_reac))
                    if store_paths:
                        good_path_index.append([i_tot_reac, nnz])
                        solved_path.append(reac_c)
                    path_state_stack.pop()
                    nbrach_stack.pop()
                    stat_dict["stack_depth"] -= 1
                    return
            path_state_stack.pop()
            nbrach_stack.pop()
            stat_dict["stack_depth"] -= 1
            return

        if max_walk_node > 0 and stat_dict["n_walk_node"] > max_walk_node:
            path_state_stack.pop()
            nbrach_stack.pop()
            stat_dict["stack_depth"] -= 1
            return

        if stat_dict["target_depth"] > 0 and stat_dict["stack_depth"] > stat_dict["target_depth"]:
            path_state_stack.pop()
            nbrach_stack.pop()
            stat_dict["stack_depth"] -= 1
            return
        else:
            stat_dict["target_depth"] = -1

        # A reaction is a candidate when one of its consumed species is
        # present.  Product gas species are ignored, exactly as in the old
        # implementation.
        mask_current_state = current_state & ~exclude_gas_mask
        candidate_index = [
            (i, True) for i in range(n_reactions)
            if not path[i] and (reactant_int[i] & mask_current_state)
        ]
        if find_reverse:
            candidate_index.extend(
                (i, False) for i in range(n_reactions)
                if (not path[i] and reaction_reversible[i]
                    and (product_int[i] & mask_current_state))
            )

        # A path never contains the same reaction twice, so the set of
        # ``(reaction, direction)`` pairs in a suffix is enough to identify
        # it.  Pack that set into one integer (two bits per reaction, even
        # bit = forward, odd bit = reverse) instead of a frozenset of tuples.
        # This is what keeps the per-ancestor visited sets small on large,
        # deep searches; a frozenset per suffix can use an order of magnitude
        # more memory and was the reason the uncapped ex1 run hit the OOM
        # killer in 7 GB.
        suffix_keys: List[int] = []
        suffix_key = 0
        for i_pss in range(1, min(len(path_state), save_depth) + 1):
            reaction_i, is_forward_i = path_state[-i_pss]
            suffix_key |= 1 << (2 * reaction_i + (1 if is_forward_i else 0))
            suffix_keys.append(suffix_key)

        total_state = previous_state | current_state
        select_result = []
        for ci, is_forward in candidate_index:
            path_has_been_visited = False
            candidate_bit = 1 << (2 * ci + (1 if is_forward else 0))
            for i_pss, suffix in enumerate(suffix_keys, start=1):
                saved_path_set = path_state_stack[-(1 + i_pss)]
                ps_0 = suffix | candidate_bit
                if ps_0 in saved_path_set:
                    path_has_been_visited = True
                    break
                saved_path_set.add(ps_0)
            if path_has_been_visited:
                continue

            if is_forward:
                if (reactant_int[ci] & total_state) == reactant_int[ci]:
                    all_prod_included = (
                        (product_int[ci] & total_state) == product_int[ci])
                    if not all_prod_included or not (product_int[ci] & center_mask):
                        select_result.append((ci, True))
            else:
                if (product_int[ci] & total_state) == product_int[ci]:
                    all_reac_included = (
                        (reactant_int[ci] & total_state) == reactant_int[ci])
                    if not all_reac_included or not (reactant_int[ci] & center_mask):
                        select_result.append((ci, False))

        nbrach_stack[-1] = len(select_result)
        if len(select_result) == 0:
            stat_dict["n_dead_end"] += 1
            stat_dict["stack_depth"] -= 1
            path_state_stack.pop()
            nbrach_stack.pop()
            return

        stat_dict["n_noraml"] += 1
        if max_walk_node > 0:
            if rng is None:
                shuffle(select_result)
            else:
                rng.shuffle(select_result)

        for index, is_forward in select_result:
            path_state_new = path_state.copy()
            path_state_new.append((index, is_forward))
            if is_forward:
                consumed = reactant_int[index]
                remain = product_int[index]
            else:
                consumed = product_int[index]
                remain = reactant_int[index]
            path_new = path.copy()
            path_new[index] = True
            if check_rank:
                stat_dict["n_check_rank"] += 1
                rank = np.linalg.matrix_rank(tot_species_coeff[path_new, :])
                if rank != np.sum(path_new):
                    continue

            previous_state_new = previous_state | consumed
            current_state_new = (current_state | remain) & ~previous_state_new
            walk_node(path_state_new, path_new, current_state_new,
                      previous_state_new)

        stat_dict["stack_depth"] -= 1
        path_state_stack.pop()
        nbrach_stack.pop()

    # ------------------------------------------------------------------
    # Set up index arrays
    # ------------------------------------------------------------------
    n_reactions = len(mkm.reaction_list)
    surface_species = list(mkm.surface_species)
    gas_species = list(dict.fromkeys(
        list(getattr(mkm, "gas_species", []) or [])
        + list(getattr(mkm_gas, "gas_species", []) or [])))
    tot_species_list = np.array(surface_species + gas_species, dtype=str)
    n_tot_species = len(tot_species_list)
    species_to_index = {tot_species_list[i]: i for i in range(n_tot_species)}

    center_species_flag = np.zeros((n_tot_species,), dtype=bool)
    center_species_set = set()
    for cs in center_species:
        if cs in tot_species_list:
            center_species_set.add(cs)
        elif len(cs) == 2 and cs[1] == '*':
            center_species_set.update(t for t in tot_species_list if cs[0] in t)
    center_species_flag[[species_to_index[name] for name in center_species_set]] = True
    center_mask = 0
    for i in np.where(center_species_flag)[0]:
        center_mask |= 1 << int(i)

    sites_name = []
    surface_species_name = []
    gas_species_name = []
    for s in tot_species_list:
        if s[0] == '#':
            sites_name.append(s)
        elif '#' in s:
            surface_species_name.append(s)
        else:
            gas_species_name.append(s)
    n_sites = len(sites_name)
    n_surface_species = len(surface_species_name)

    surface_mask = 0
    for name in surface_species_name:
        surface_mask |= 1 << species_to_index[name]

    # Overall reactions
    n_gas_reactions = len(mkm_gas.reaction_list)
    gas_species_coeff = np.zeros((n_gas_reactions, n_tot_species))
    product_gas_names = set()
    reactant_gas_names = set()
    for ir, reaction in enumerate(mkm_gas.reaction_list):
        for r in reaction.reactants:
            reactant_gas_names.add(r.name)
            gas_species_coeff[ir, species_to_index[r.name]] = -r.num
        for p in reaction.products:
            product_gas_names.add(p.name)
            gas_species_coeff[ir, species_to_index[p.name]] = p.num
    gas_species_flag = gas_species_coeff != 0
    exclude_gas_index = [species_to_index[name] for name in product_gas_names]
    exclude_gas_mask = 0
    for i in exclude_gas_index:
        exclude_gas_mask |= 1 << int(i)

    gas_reactant_masks = []
    for ir in range(n_gas_reactions):
        mask = 0
        for i in np.where(gas_species_flag[ir])[0]:
            mask |= 1 << int(i)
        gas_reactant_masks.append(mask)

    reactant_coeff = np.zeros((n_reactions, n_tot_species), dtype=bool)
    product_coeff = np.zeros((n_reactions, n_tot_species), dtype=bool)
    reaction_reversible = []

    tot_species_coeff = np.zeros((n_reactions, n_tot_species))
    for ir, reaction in enumerate(mkm.reaction_list):
        for r in reaction.reactants:
            tot_species_coeff[ir, species_to_index[r.name]] = -r.num
            reactant_coeff[ir, species_to_index[r.name]] = True
        for p in reaction.products:
            tot_species_coeff[ir, species_to_index[p.name]] = p.num
            product_coeff[ir, species_to_index[p.name]] = True
        reaction_reversible.append(bool(reaction.reversible))

    reactant_int = []
    product_int = []
    for ir in range(n_reactions):
        rm = 0
        pm = 0
        for i in np.where(reactant_coeff[ir])[0]:
            rm |= 1 << int(i)
        for i in np.where(product_coeff[ir])[0]:
            pm |= 1 << int(i)
        reactant_int.append(rm)
        product_int.append(pm)

    current_state = 0
    previous_state = 0
    path = np.zeros((n_reactions,), dtype=bool)
    current_state_name = sites_name + list(auto_catalytic_species) + list(reactant_gas_names)
    for name in current_state_name:
        current_state |= 1 << species_to_index[name]
    walk_node([], path, current_state, previous_state)

    npath = len(solved_path) if store_paths else 0
    index_list: List[np.ndarray] = [
        np.empty(0, dtype=int) for _ in range(n_gas_reactions)
    ]
    if npath > 0:
        solved_path_arr = np.array(solved_path)
        good_path_index_arr = np.array(good_path_index)
        ind = np.lexsort((good_path_index_arr[:, 1], good_path_index_arr[:, 0]))
        solved_path_arr = solved_path_arr[ind, :]
        good_path_index_arr = good_path_index_arr[ind, :]

        for i in range(n_gas_reactions):
            i_index = np.where(good_path_index_arr[:, 0] == i)[0]
            index_list[i] = i_index
            if verbose and len(i_index) == 0:
                continue
            if verbose:
                print(mkm_gas.reaction_list[i], "from", i_index[0],
                      "to", i_index[-1])
    else:
        solved_path_arr = np.empty((0, n_reactions))

    if verbose:
        print("+++++++++ DONE +++++++++")
        print("n_walk_node = ", stat_dict["n_walk_node"])
        print("n_check_rank = ", stat_dict["n_check_rank"])
        print("size of stack = ", len(path_state_stack))
        print("stack depth = ", stat_dict["stack_depth"])
        print("The number of path: ",
              npath if store_paths else stat_dict["n_solved_path"])
        if print_path and npath > 0:
            for ip in range(min(npath, 51)):
                c = solved_path_arr[ip, :]
                flag = np.abs(c) > 0.1
                print(np.where(flag)[0])
                for ir in range(n_reactions):
                    if flag[ir]:
                        print(f"{str(mkm.reaction_list[ir]):<35s}x {round(c[ir])}")
                print("===================")

    return ReactionPathResult(
        paths=solved_path_arr,
        index=index_list,
        total_reactions=[str(r) for r in mkm_gas.reaction_list],
        auto_catalytic_species=list(auto_catalytic_species),
        stats=dict(stat_dict),
    )


# ---------------------------------------------------------------------------
# Sharded search (one search per intermediate)
# ---------------------------------------------------------------------------


def _coerce_shard_groups(mkm: MKM, shards) -> List[List[str]]:
    """Normalise *shards* to a list of initial-species groups.

    A plain sequence of names gives one group per name (the classic
    one-intermediate-per-job workflow).  A sequence of sequences gives
    explicit multi-species seed groups, which is needed for pathways that
    require more than one species to be present initially.  Gas species are
    accepted as seeds too.

    ``"each"`` gives one group per non-bare surface species; ``"all"`` also
    includes the gas species; ``"auto"`` uses the detected auto-catalytic
    species.
    """
    if shards is None or (isinstance(shards, str) and shards == "each"):
        groups = [[s] for s in list_shard_species(mkm)]
    elif isinstance(shards, str) and shards == "all":
        groups = [[s] for s in list_shard_species(mkm, include_gas=True)]
    elif isinstance(shards, str) and shards == "auto":
        groups = [[s] for s in detect_autocatalytic_species(mkm)]
    elif isinstance(shards, str):
        groups = [[s.strip()] for s in shards.split(",") if s.strip()]
    else:
        groups = []
        for item in shards:
            if isinstance(item, str):
                groups.append([item])
            else:
                groups.append([str(s) for s in item])

    # Keep order, drop duplicates inside and between groups.
    known = set(mkm.surface_species) | set(getattr(mkm, "gas_species", []) or [])
    seen = set()
    result: List[List[str]] = []
    for group in groups:
        group = list(dict.fromkeys(group))
        unknown = [s for s in group if s not in known]
        if unknown:
            raise ValueError(
                "Unknown shard species: " + ", ".join(unknown)
                + ". Available species: " + ", ".join(sorted(known)))
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return result


def _run_shard(payload):
    mkm, mkm_gas, group, kwargs = payload
    options = dict(kwargs)
    options["auto_catalytic_species"] = list(group)
    options["verbose"] = False
    return find_reaction_paths(mkm, mkm_gas, **options)


def find_reaction_paths_sharded(
    mkm: MKM,
    mkm_gas: Optional[MKM] = None,
    shards: Union[None, str, Sequence[str],
                  Sequence[Sequence[str]]] = None,
    merge: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
    on_shard: Optional[Callable[[str, ReactionPathResult], None]] = None,
    **kwargs,
) -> Union[ReactionPathResult, List[ReactionPathResult]]:
    """Run one path search per seed group and merge the results.

    Parameters
    ----------
    shards:
        ``None`` or ``"each"``: one group per non-bare surface species (the
        classic per-intermediate workflow).  ``"all"``: the same, plus one
        group per gas species.  ``"auto"``: one group per detected
        auto-catalytic species.  A single species, a comma-separated string,
        or a sequence of names gives explicit one-species groups.  A sequence
        of sequences gives explicit multi-species seed groups, e.g.
        ``[["CH3#", "CO#", "H#"]]``.  Multi-species groups are needed for
        pathways that require more than one species to be present initially;
        a one-species shard cannot find those.  Gas species are accepted as
        seeds, both alone and in groups.
    merge:
        If True (default), return one merged result with duplicate paths
        removed.  If False, return the list of per-shard results.
    n_jobs:
        Number of worker processes.  ``1`` runs serially.  This is useful on
        a multi-core machine; for Slurm, run one shard per job instead (use
        ``shards="..."`` with one species, or the CLI ``--shard`` option).
    on_shard:
        Optional callback ``on_shard(label, result)`` invoked after each
        shard finishes; *label* is the group joined with ``+``.
    kwargs:
        Forwarded to :func:`find_reaction_paths` (``find_reverse``,
        ``check_rank``, ``max_walk_node``, ``save_depth``, ...).  The
        ``auto_catalytic_species`` option is ignored here; it is replaced by
        the shard seed group.
    """
    if mkm_gas is None:
        mkm_gas = mkm
    shard_groups = _coerce_shard_groups(mkm, shards)
    options = dict(kwargs)
    options.pop("auto_catalytic_species", None)
    options.pop("verbose", None)

    if not shard_groups:
        # Nothing to shard: fall back to a single unrestricted search.
        options["auto_catalytic_species"] = []
        result = find_reaction_paths(mkm, mkm_gas, verbose=verbose, **options)
        return result if merge else [result]

    if n_jobs > 1 and options.get("on_path") is not None:
        raise ValueError("on_path callbacks cannot be used with n_jobs > 1.")
    if merge and options.get("store_paths") is False:
        raise ValueError("merge=True requires store_paths=True; use "
                         "merge=False when streaming paths with on_path.")

    labels = ["+".join(group) for group in shard_groups]
    results: List[Optional[ReactionPathResult]] = [None] * len(shard_groups)
    if n_jobs <= 1:
        for i, (label, group) in enumerate(zip(labels, shard_groups)):
            if verbose:
                print(f"[{i + 1}/{len(shard_groups)}] shard {label} ... ",
                      end="", flush=True)
            result = _run_shard((mkm, mkm_gas, group, options))
            results[i] = result
            if verbose:
                print(f"{len(result.paths)} paths, "
                      f"{result.stats.get('n_walk_node', 0)} nodes")
            if on_shard is not None:
                on_shard(label, result)
    else:
        payloads = [(mkm, mkm_gas, group, options)
                    for group in shard_groups]
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for i, (label, result) in enumerate(
                    zip(labels, pool.map(_run_shard, payloads))):
                results[i] = result
                if verbose:
                    print(f"shard {label}: {len(result.paths)} paths, "
                          f"{result.stats.get('n_walk_node', 0)} nodes")
                if on_shard is not None:
                    on_shard(label, result)

    finished = [r for r in results if r is not None]
    if merge:
        return merge_reaction_paths(finished)
    return finished


# Backwards-compatible alias.
find_reaction_path_dynamic = find_reaction_paths
