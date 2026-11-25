"""Core reaction-network model and code generation of the ODE/Jacobian.

The central object is :class:`MKM`.  It stores the reaction network, the gas
species, the current temperature and the rate constants.  The residues of the
surface-coverages ODE (``F``) and its Jacobian (``J``) are built from the
network with :func:`write_F_and_J_function` (to a file) or
:func:`build_rate_functions` (in memory).
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence

import mpmath as mp
import numpy as np

from .constants import EV_TO_J_MOL, KBH, R_GAS

_MP = mp.mp

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class GasSpecies:
    """A gas-phase species.

    Parameters
    ----------
    name:
        Species name as it appears in the reaction network.
    pressure:
        List/array of partial pressures used in the calculation.  The values
        are interpreted in the same units as the rate constants.
    """

    def __init__(self, name: str = "", pressure=None, charge: int = 0):
        self.name = name
        self.pressure = pressure if pressure is not None else [1.0]
        self.charge = charge
        self.diffusion_constant = None
        self.effective_cell_size = 0.0
        self.sticking_coeff = None
        self.mole_mass = None
        self.redundant_species = False
        # Free-form per-species thermo data (wavenums, shomate_params, ...)
        self.thermo_params: Dict = {}

    def __repr__(self) -> str:
        return f"GasSpecies(name={self.name!r}, pressure={list(self.pressure)!r})"


class Species:
    """A species taking part in a reaction.

    ``nsites`` is the number of surface sites occupied (0 for gas species).
    """

    __slots__ = ("name", "nsites", "num", "site")

    def __init__(self, name: str, nsites: int, num: float, site: Optional[str] = None):
        self.name = name
        self.nsites = nsites
        self.num = num
        self.site = site

    def __str__(self) -> str:
        if self.site:
            return (f"{self.name}, nsites:{self.nsites}, num:{self.num}, "
                    f"site:{self.site}")
        return f"{self.name}, nsites:{self.nsites}, num:{self.num}"


class Reaction:
    """A reaction ``reactants <-> products`` (optionally irreversible)."""

    def __init__(self, reactants, transition_state, products, reversible=True):
        self.reactants: List[Species] = reactants
        self.transition_state: Optional[Species] = transition_state
        self.products: List[Species] = products
        self.reversible = reversible

    def __str__(self) -> str:
        def fmt(species):
            return " + ".join(
                r.name if abs(r.num - 1.0) < 0.001 else f"{r.num}*{r.name}"
                for r in species
            )
        s = fmt(self.reactants)
        s += " <-> " if self.reversible else " -> "
        s += fmt(self.products)
        return s

    def __repr__(self) -> str:
        return f"Reaction({str(self)!r})"


class MKM:
    """Microkinetic model container.

    Attributes are intentionally close to the old catrap scripts so that the
    refactoring stays familiar.
    """

    def __init__(self):
        self.T = None                 # list of temperatures (calculation loop)
        self.temperature = None
        self.gas_list: List[GasSpecies] = []
        self.gas = None               # numpy array of current partial pressures

        self.reaction_list: List[Reaction] = []
        self.surface_species: List[str] = []
        self.gas_species: List[str] = []
        self.redundant_gas_names: List[str] = []

        self.nreactions = 0
        self.nspecies = 0

        self.rate_constants = None      # numpy (nreactions, 2)
        self.rate_constants_mpf = None  # mpmath matrix (nreactions, 2)
        self.change_gas = False         # gas partial pressures as ODE variables
        self.sites_total_coverage = None
        self.sites_number_density = None
        self.normalize_cvr = None
        self.constrain_list = None
        self.one_type_site_and_single_occu = True
        self.calc_F = None
        self.calc_J = None
        self.data_ndigit = 15

        # bookkeeping populated by parse_network
        self.sites_name: List[str] = []
        self.reaction_strings: List[str] = []

    # -- ODE callables ----------------------------------------------------
    @property
    def nvar(self) -> int:
        """Number of ODE state variables (surface, plus gas if ``change_gas``)."""
        return self.nspecies + (len(self.gas_list) if self.change_gas else 0)

    def F(self, y):
        R = _MP.matrix(self.nreactions, 1)
        D = _MP.matrix(self.nvar, 1)
        return self.calc_F(y, self.rate_constants_mpf, R, D, self.gas)

    def J(self, y):
        R = _MP.matrix(self.nreactions, self.nvar)
        D = _MP.matrix(self.nvar, self.nvar)
        return self.calc_J(y, self.rate_constants_mpf, R, D, self.gas)

    def Ft(self, t, y):
        R = np.zeros((self.nreactions,))
        D = np.zeros((self.nvar,))
        return self.calc_F(y, self.rate_constants, R, D, self.gas)

    def Ft_vec(self, t, y):
        R = np.zeros((self.nreactions, y.shape[1]))
        D = np.zeros((y.shape[0], y.shape[1]))
        return self.calc_F(y, self.rate_constants, R, D, self.gas)

    def Jt(self, t, y):
        R = np.zeros((self.nreactions, self.nvar))
        D = np.zeros((self.nvar, self.nvar))
        return self.calc_J(y, self.rate_constants, R, D, self.gas)


# Backwards-compatible names used by the old scripts / examples.
MKM_model = MKM
Gas_species = GasSpecies


# ---------------------------------------------------------------------------
# Network parsing
# ---------------------------------------------------------------------------


def _split_items(text: str, reaction: str, redundant_species: Iterable[str]):
    """Parse one side of a reaction, e.g. ``2*H# + CO``."""
    count_item: Dict[str, int] = {}
    for item in text.split('+'):
        if '*' in item:
            if item.count('*') != 1:
                raise ValueError(
                    f"{reaction}\nERROR: the asterisk means multiply; it can "
                    "only appear once in a species.")
            num_str, name = item.split('*')
            if name in redundant_species:
                continue
            try:
                num = int(num_str)
            except ValueError:
                raise ValueError(
                    f"{reaction}\nERROR: expected an integer on the left of '*'.")
            count_item[name] = count_item.get(name, 0) + num
        else:
            if item in redundant_species:
                continue
            count_item[item] = count_item.get(item, 0) + 1

    result: List[Species] = []
    sites_count: Dict[str, int] = {}
    for name, num in count_item.items():
        nsites = name.count('#')
        if nsites == 0:
            result.append(Species(name, nsites, num))
            continue
        sites = np.array(re.findall(r'#\d*', name), dtype=str)
        if not np.all(sites == sites[0]):
            raise ValueError(
                f"{reaction}\nERROR: a species can only occupy one kind of site.")
        site = sites[0]
        sites_count[site] = sites_count.get(site, 0) + nsites * num
        result.append(Species(name, nsites, num, site))
    return result, sites_count


def parse_network(
    mkm: MKM,
    reactions: Sequence[str],
    check: bool = True,
    redundant_species: Optional[Sequence[str]] = None,
    check_gas: bool = True,
) -> MKM:
    """Parse reaction strings and fill *mkm*.

    Parameters
    ----------
    check:
        Verify that the number of surface sites is balanced for each reaction.
    redundant_species:
        Species that should be ignored while parsing.
    check_gas:
        If True, every gas species in the network must be present in
        ``mkm.gas_list``.  Set to False for reaction-path searches, which do
        not require thermodynamic data.
    """
    redundant_species = list(redundant_species or [])
    mkm.nreactions = len(reactions)
    mkm.reaction_strings = list(reactions)
    reaction_list: List[Reaction] = []
    species_set = set()
    gas_set = set()
    sites_name = set()

    for original in reactions:
        reac = original.replace(' ', '').replace('\t', '')
        reversible = True
        if '<->' in reac:
            parts = reac.split('<->')
            if any(('->' in b or '<-' in b) for b in parts):
                raise ValueError(
                    f"{reac}\nERROR: '<->' cannot be combined with '->' or '<-'.")
        elif '<-' in reac:
            raise ValueError(
                f"{reac}\nERROR: only '->' is allowed for irreversible reactions.")
        elif '->' in reac:
            parts = reac.split('->')
            reversible = False
        else:
            raise ValueError(f"{reac}\nERROR: '<->' or '->' must appear in a reaction.")

        if len(parts) == 2:
            r_txt, ts_txt, p_txt = parts[0], None, parts[1]
        elif len(parts) == 3:
            r_txt, ts_txt, p_txt = parts[0], parts[1], parts[2]
        else:
            raise ValueError(
                f"{reac}\nERROR: a reaction must contain two or three items.")

        reactants, sites_r = _split_items(r_txt, reac, redundant_species)
        products, sites_p = _split_items(p_txt, reac, redundant_species)
        transition_state = None
        if ts_txt is not None:
            transition_state = Species(
                ts_txt, 1, 1,
                ts_txt[ts_txt.index('#'):] if '#' in ts_txt else None)

        species_set.update(a.name for a in reactants if a.nsites != 0)
        species_set.update(a.name for a in products if a.nsites != 0)
        gas_set.update(a.name for a in reactants if a.nsites == 0)
        gas_set.update(a.name for a in products if a.nsites == 0)
        sites_name.update(a.site for a in reactants if a.nsites != 0)
        sites_name.update(a.site for a in products if a.nsites != 0)

        if check:
            if set(sites_p) != set(sites_r) or any(
                    sites_r[k] != sites_p.get(k) for k in sites_r):
                raise ValueError(
                    f"{reac}\nERROR: the number of sites must be balanced.")

        reaction_list.append(Reaction(reactants, transition_state, products,
                                      reversible))

    species_list = sorted(species_set)
    sites_name = sorted(sites_name)

    if (all(s.count('#') == 1 for s in species_list) and len(sites_name) == 1):
        mkm.one_type_site_and_single_occu = True
    else:
        mkm.one_type_site_and_single_occu = False
        if mkm.sites_total_coverage is None:
            mkm.sites_total_coverage = {'#': 1.0}

    if len(sites_name) == 1 and sites_name[0] != '#':
        raise ValueError("ERROR: if there is only one kind of site, use '#'.")
    if len(sites_name) > 1 and any(re.search(r'#\d+', s) is None
                                   for s in sites_name):
        raise ValueError(
            "ERROR: if there are multiple sites, use #1, #2, ... to represent them.")

    if '#' in species_list and species_list.index('#') != 0:
        species_list[species_list.index('#')] = species_list[0]
        species_list[0] = '#'

    mkm.reaction_list = reaction_list
    mkm.surface_species = species_list
    mkm.sites_name = sites_name
    mkm.nspecies = len(species_list)

    gas_name_set = {gas.name for gas in mkm.gas_list}
    gas_set.difference_update(mkm.redundant_gas_names)
    mkm.gas_species = sorted(gas_set)
    if check_gas:
        missing = gas_set.difference(gas_name_set)
        if missing:
            raise ValueError(f"ERROR: the gas pressure of {sorted(missing)} is not set.")
    return mkm


# ---------------------------------------------------------------------------
# Random initial guesses
# ---------------------------------------------------------------------------


def random_disturb(y, normalize_cvr):
    """Randomly perturb a coverage vector while preserving normalisation."""
    y = np.asarray(y, dtype=float).copy()
    for i in range(len(y)):
        y[i] = y[i] + abs(y[i]) * np.random.random() * 0.2
    if isinstance(normalize_cvr, (int, np.integer)):
        return y / np.sum(y)
    for nc in normalize_cvr:
        factor = nc[1] / np.sum(y[nc[2]] * nc[3])
        y[nc[2]] *= factor
    return y


def random_guess(normalize_cvr, constrain_list):
    """Generate a random, physically admissible initial coverage vector."""
    if isinstance(normalize_cvr, (int, np.integer)):
        nspecies = normalize_cvr
        min_cvr = 0.3 / nspecies
    else:
        nspecies = 0
        min_cvr = 1.0
        for nc in normalize_cvr:
            nspecies += len(nc[2])
            min_cvr = min(min_cvr, 0.3 * nc[1] / len(nc[2]))

    if constrain_list:
        ntry = 0
        while ntry < 100:
            y2 = np.random.random((nspecies,))
            for c_list in constrain_list:
                y2[c_list[0][0]] = 0.0
                for i in range(1, len(c_list)):
                    y2[c_list[0][0]] -= c_list[i][1] * y2[c_list[i][0]]
                y2[c_list[0][0]] /= c_list[0][1]
            if np.all(y2 > 0):
                break
            ntry += 1
        else:
            y2 = np.random.random((nspecies,))
    else:
        y2 = np.random.random((nspecies,))

    y2 *= min_cvr
    if isinstance(normalize_cvr, (int, np.integer)):
        y2[0] = 1.0 - np.sum(y2[1:nspecies])
    else:
        for nc in normalize_cvr:
            a = np.sum([y2[nc[2][j]] * nc[3][j]
                        for j in range(len(nc[2])) if nc[2][j] != nc[0]])
            y2[nc[0]] = nc[1] - a
    return y2


# ---------------------------------------------------------------------------
# Rate constants
# ---------------------------------------------------------------------------


def collision_theory_gas_adsorption(mkm: MKM, rate_constants, T):
    """Replace adsorption rate constants by collision-theory values.

    Only gas species that provide ``mole_mass`` and ``sticking_coeff`` are
    treated.  ``mkm.sites_number_density`` maps a site name to a site density.
    """
    if mkm.sites_number_density is None:
        return

    def adsorption_reaction(name_list):
        if len(name_list) != 2:
            return None, None
        if '#' not in name_list[0] and name_list[1].startswith('#'):
            return name_list[0], name_list[1]
        if '#' not in name_list[1] and name_list[0].startswith('#'):
            return name_list[1], name_list[0]
        return None, None

    C0 = 101325.0 / R_GAS / T
    RT = R_GAS * T
    collision_info = [
        (gas.name, gas.mole_mass, gas.sticking_coeff)
        for gas in mkm.gas_list
        if gas.mole_mass is not None and gas.sticking_coeff is not None
    ]
    if not collision_info:
        return

    def overwrite(ir, gas_name, site_name, forward: bool):
        for name, mass, s0 in collision_info:
            if name != gas_name:
                continue
            L_tot = mkm.sites_number_density.get(site_name)
            if L_tot is None:
                return False
            K = rate_constants[ir, 0] / rate_constants[ir, 1]
            k_ads = C0 * s0 * (RT / (2 * np.pi * mass)) ** 0.5 / L_tot
            if forward:
                rate_constants[ir, 0] = k_ads
                rate_constants[ir, 1] = k_ads / K
            else:
                rate_constants[ir, 1] = k_ads
                rate_constants[ir, 0] = k_ads * K
            return True
        return False

    for ir, reaction in enumerate(mkm.reaction_list):
        gas_name, site_name = adsorption_reaction(
            [r.name for r in reaction.reactants])
        if gas_name is not None and overwrite(ir, gas_name, site_name, True):
            continue
        gas_name, site_name = adsorption_reaction(
            [p.name for p in reaction.products])
        if gas_name is not None:
            overwrite(ir, gas_name, site_name, False)


def get_rate_constants_EaH(barrier_data, T, ndigit: int):
    """Arrhenius rate constants from ``[Ea, reaction_energy]`` rows."""
    barrier_data = np.asarray(barrier_data, dtype=float)
    n = barrier_data.shape[0]
    if ndigit <= 0:
        rate_constants = np.zeros((n, 2))
        C = EV_TO_J_MOL
        AT = KBH * T
        RT = R_GAS * T
        for i in range(n):
            Ea = barrier_data[i, 0]
            H0 = barrier_data[i, 1]
            if H0 > Ea:
                Ea = H0
            rate_constants[i, 0] = AT * np.exp(-C * Ea / RT)
            K = np.exp(-C * H0 / RT)
            rate_constants[i, 1] = rate_constants[i, 0] / K
        return rate_constants

    rate_constants_mpf = _MP.matrix(n, 2)
    fs = "{:." + str(ndigit) + "f}"
    C = _MP.mpf(str(EV_TO_J_MOL))
    A = _MP.mpf(str(KBH))
    R = _MP.mpf(str(R_GAS))
    AT = A * _MP.mpf(fs.format(T))
    RT = R * _MP.mpf(fs.format(T))
    for i in range(n):
        Ea = barrier_data[i, 0]
        H0 = barrier_data[i, 1]
        if H0 > Ea:
            Ea = H0
        rate_constants_mpf[i, 0] = AT * _MP.exp(-C * _MP.mpf(fs.format(Ea)) / RT)
        K = _MP.exp(-C * _MP.mpf(fs.format(H0)) / RT)
        rate_constants_mpf[i, 1] = rate_constants_mpf[i, 0] / K
    return rate_constants_mpf


def get_rate_constants_ads_data(mkm: MKM, ads_dict: dict, T, ndigit: int):
    """Arrhenius rate constants from per-species energy dictionaries."""
    if ndigit <= 0:
        rate_constants = np.zeros((mkm.nreactions, 2))
        C = EV_TO_J_MOL
        AT = KBH * T
        RT = R_GAS * T
    else:
        rate_constants = _MP.matrix(mkm.nreactions, 2)
        fs = "{:." + str(ndigit) + "f}"
        C = _MP.mpf(str(EV_TO_J_MOL))
        A = _MP.mpf(str(KBH))
        R = _MP.mpf(str(R_GAS))
        AT = A * _MP.mpf(fs.format(T))
        RT = R * _MP.mpf(fs.format(T))

    for ir, r in enumerate(mkm.reaction_list):
        E_INI = sum(ads_dict[a.name] * a.num for a in r.reactants)
        E_FIN = sum(ads_dict[a.name] * a.num for a in r.products)
        H = E_FIN - E_INI
        if r.transition_state is None:
            if H > 0:
                Ea_fwd, Ea_bwd = H, 0.0
            else:
                Ea_fwd, Ea_bwd = 0.0, -H
        else:
            E_TS = max(E_INI, ads_dict[r.transition_state.name], E_FIN)
            Ea_fwd = max(E_TS - E_INI, 0.0)
            Ea_bwd = Ea_fwd - H

        if ndigit <= 0:
            rate_constants[ir, 0] = AT * np.exp(-C * Ea_fwd / RT)
            rate_constants[ir, 1] = AT * np.exp(-C * Ea_bwd / RT)
        else:
            rate_constants[ir, 0] = AT * _MP.exp(-C * _MP.mpf(fs.format(Ea_fwd)) / RT)
            rate_constants[ir, 1] = AT * _MP.exp(-C * _MP.mpf(fs.format(Ea_bwd)) / RT)
    return rate_constants


def get_rate_constants(mkm: MKM, barrier_data, T, ndigit: int):
    """Dispatch to the appropriate rate-constant builder."""
    if isinstance(barrier_data, (list, tuple)):
        barrier_data = np.asarray(barrier_data, dtype=float)
    if isinstance(barrier_data, np.ndarray):
        rate_constants = get_rate_constants_EaH(barrier_data, T, ndigit)
    else:
        rate_constants = get_rate_constants_ads_data(mkm, barrier_data, T, ndigit)
    collision_theory_gas_adsorption(mkm, rate_constants, T)
    return rate_constants


# ---------------------------------------------------------------------------
# ODE / Jacobian code generation
# ---------------------------------------------------------------------------


def gauss_elimination(species_coeff):
    """Gaussian elimination used to detect rank-deficiency constraints."""
    A = species_coeff.copy()
    nrow, ncol = A.shape
    pivots = list(range(nrow))
    B = np.eye(nrow)

    for i in range(nrow):
        sum_abs_row = np.sum(np.abs(A[i:, ]), axis=0)
        start_j = 0
        while start_j < ncol and sum_abs_row[start_j] < 1.0e-15:
            start_j += 1
        if start_j == ncol:
            return pivots, A, B, i
        max_row_index = int(np.argmax(np.abs(A[i:, start_j]))) + i
        if max_row_index != i:
            A[[max_row_index, i]] = A[[i, max_row_index]]
            B[[max_row_index, i]] = B[[i, max_row_index]]
            pivots[max_row_index], pivots[i] = pivots[i], pivots[max_row_index]
        for j in range(i + 1, nrow):
            if A[j, start_j] != 0:
                factor = float(A[j, start_j]) / A[i, start_j]
                A[j, :] -= factor * A[i, :]
                B[j, :] -= factor * B[i, :]
    return pivots, A, B, nrow


def _prepare_generation(mkm: MKM):
    """Validate the model and compute normalisation/rank constraints."""
    surface_species = mkm.surface_species
    ns = len(surface_species)
    ng = len(mkm.gas_list)
    nr = len(mkm.reaction_list)
    gas_str = [gas.name for gas in mkm.gas_list]

    if mkm.sites_total_coverage is not None:
        if (len(mkm.sites_total_coverage) == 1
                and abs(mkm.sites_total_coverage['#'] - 1.0) > 0.001):
            raise ValueError(
                "ERROR: there is only one kind of site; the total coverage "
                "must be 1.0.")

    return surface_species, gas_str, ns, ng, nr


def _generate_rate_source(mkm: MKM, fname_header: bool = True) -> str:
    """Return Python source code defining ``calc_F`` and ``calc_J``."""
    surface_species, gas_str, ns, ng, nr = _prepare_generation(mkm)
    reaction_list = mkm.reaction_list

    species_to_index = {surface_species[i]: i for i in range(ns)}
    species_to_str = {surface_species[i]: f'y[{i}]' for i in range(ns)}
    if mkm.change_gas:
        species_to_index.update({gas_str[i]: ns + i for i in range(ng)})
        species_to_str.update({gas_str[i]: f'y[{ns + i}]' for i in range(ng)})
        species_coeff = np.zeros((ns + ng, nr))
    else:
        species_to_index.update({gas_str[i]: i for i in range(ng)})
        species_to_str.update({gas_str[i]: f'gas[{i}]' for i in range(ng)})
        species_coeff = np.zeros((ns, nr))

    def pro_str(name, n, c):
        result = '*'.join([name] * n)
        if abs(c - 1.0) < 0.001:
            return result
        return f'{c}*{result}'

    lines: List[str] = []
    if fname_header:
        for i in range(ns):
            lines.append(f"#{surface_species[i]:<15s}{i:<6d}")
        if mkm.change_gas:
            for i in range(ng):
                lines.append(f"#{gas_str[i]:<15s}{ns + i:<6d}")
        else:
            for i in range(ng):
                lines.append(f"#{gas_str[i]:<15s} gas[{i}]")
    lines += [
        "import numpy as np",
        "import mpmath as mp",
        "mp = mp.mp",
        "mp.dps = 500",
        "def calc_F(y, K, R, D, gas=None):",
    ]

    for i, r in enumerate(reaction_list):
        fwd_parts = []
        for a in r.reactants:
            fwd_parts.append('*'.join([species_to_str[a.name]] * a.num))
            site_factor = int(a.nsites ** a.num)
            if site_factor not in (0, 1):
                fwd_parts.append(str(site_factor))
            if a.name not in gas_str or mkm.change_gas:
                species_coeff[species_to_index[a.name], i] -= a.num

        bwd_parts = []
        for a in r.products:
            bwd_parts.append('*'.join([species_to_str[a.name]] * a.num))
            site_factor = int(a.nsites ** a.num)
            if site_factor not in (0, 1):
                bwd_parts.append(str(site_factor))
            if a.name not in gas_str or mkm.change_gas:
                species_coeff[species_to_index[a.name], i] += a.num

        fwd_str = '*'.join(fwd_parts) if fwd_parts else "1"
        bwd_str = '*'.join(bwd_parts) if bwd_parts else "1"
        if not r.reversible:
            bwd_str = "0"
        lines.append(
            f"   R[{i}] = K[{i}, 0]*{fwd_str} - K[{i}, 1]*{bwd_str}    # {r}")

    # Rank-deficiency constraints are meaningful for surface species only.
    # When the gas pressures are state variables, their stoichiometric rows
    # are usually linearly dependent (element conservation) but we must keep
    # their ODEs untouched so that the CSTR feed/outflow terms remain valid.
    coeff_for_rank = species_coeff[:ns, :] if mkm.change_gas else species_coeff
    nrow = coeff_for_rank.shape[0]
    index_ok = ([0] if mkm.one_type_site_and_single_occu
                else [species_to_index[key] for key in mkm.sites_total_coverage])
    flag = np.ones((nrow,), dtype=bool)
    flag[index_ok] = False
    index_ori = np.where(flag)[0]
    nrow_reduce = len(index_ori)

    pivots, A, B, rank = gauss_elimination(coeff_for_rank[flag, :])
    index_ok += index_ori[pivots[:rank]].tolist()
    index_ok = set(index_ok)
    remain_index = set(range(nrow)).difference(index_ok)

    constrain_list = []
    if remain_index:
        print("WARNING: this system is 'rank deficient'.")
        for i in range(rank, nrow_reduce):
            c_list = []
            lcm = 1
            for j in range(nrow_reduce):
                if abs(B[i, j]) > 1.0e-15:
                    frac = Fraction(B[i, j]).limit_denominator(max_denominator=100)
                    if i != j:
                        c_list.append([index_ori[j], frac])
                    else:
                        c_list.insert(0, [index_ori[j], frac])
                    lcm = lcm * c_list[-1][1].denominator // np.gcd(
                        lcm, c_list[-1][1].denominator)
            for item in c_list:
                item[1] *= lcm
                item[1] = item[1].numerator
            constrain_list.append(c_list)

    mkm.constrain_list = constrain_list

    for i in range(species_coeff.shape[0]):
        index = np.where(species_coeff[i, :] != 0)[0]
        if len(index):
            r_terms = [f'R[{j}]' for j in index]
            c_terms = [f'+{species_coeff[i, j]}' if species_coeff[i, j] > 0
                       else f'{species_coeff[i, j]}' for j in index]
            rc = ''.join(
                o[0][0] + o[1] if abs(int(float(o[0]))) == 1 else o[0] + "*" + o[1]
                for o in zip(c_terms, r_terms))
            if rc.startswith('+'):
                rc = rc[1:]
        else:
            rc = "0"
        lines.append(f"   D[{i}] = {rc}")

    lines.append("   if(type(R) is mp.matrix):")
    if mkm.one_type_site_and_single_occu:
        lines.append(f"      D[0] = mp.fsum(y[0:{ns}]) - 1")
        mkm.normalize_cvr = ns
    else:
        mkm.normalize_cvr = []
        for key in mkm.sites_total_coverage:
            sum_index, nums = [], []
            sum_str = ""
            for i in range(ns):
                a = surface_species[i].count(key)
                if a != 0:
                    sum_index.append(i)
                    nums.append(a)
                    sum_str += f"+y[{i}]" + (f"*{a}" if a > 1 else "")
            lines.append(
                f"      D[{species_to_index[key]}] = "
                f"mp.mpf('-{mkm.sites_total_coverage[key]}'){sum_str}")
            mkm.normalize_cvr.append(
                (species_to_index[key], mkm.sites_total_coverage[key],
                 sum_index, np.array(nums, dtype=float)))

    if constrain_list:
        for c_list in constrain_list:
            sum_str = ""
            for idx, a in c_list:
                if abs(a) == 1:
                    sum_str += ("+" if a > 0 else "-") + f"y[{idx}]"
                else:
                    sum_str += ("+" + str(a) if a > 0 else str(a)) + f"*y[{idx}]"
            lines.append(f"      D[{c_list[0][0]}] = {sum_str}")

    lines += ["   return D", "", "", "def calc_J(y, K, R, D, gas=None):"]

    for i, r in enumerate(reaction_list):
        for j, reactant in enumerate(r.reactants):
            if reactant.name in gas_str and not mkm.change_gas:
                continue
            fwd_parts = []
            for k, other in enumerate(r.reactants):
                a = int(other.nsites ** other.num)
                if a == 0:
                    a = 1
                if k != j:
                    fwd_parts.append(pro_str(species_to_str[other.name],
                                             other.num, 1.0 * a))
                elif other.num != 1:
                    fwd_parts.append(pro_str(species_to_str[other.name],
                                             other.num - 1, other.num * a))
                elif a != 1:
                    fwd_parts.append(str(a))
            fwd_parts = [s for s in fwd_parts if s]
            fwd_str = '*'.join(fwd_parts) if fwd_parts else "1"
            lines.append(
                f"   R[{i}, {species_to_index[reactant.name]}] = "
                f"K[{i}, 0]*{fwd_str}")

        if not r.reversible:
            continue

        for j, product in enumerate(r.products):
            if product.name in gas_str and not mkm.change_gas:
                continue
            bwd_parts = []
            for k, other in enumerate(r.products):
                a = int(other.nsites ** other.num)
                if a == 0:
                    a = 1
                if k != j:
                    bwd_parts.append(pro_str(species_to_str[other.name],
                                             other.num, 1.0 * a))
                elif other.num != 1:
                    bwd_parts.append(pro_str(species_to_str[other.name],
                                             other.num - 1, other.num * a))
                elif a != 1:
                    bwd_parts.append(str(a))
            bwd_parts = [s for s in bwd_parts if s]
            bwd_str = '*'.join(bwd_parts) if bwd_parts else "1"
            lines.append(
                f"   R[{i}, {species_to_index[product.name]}] = "
                f"-K[{i}, 1]*{bwd_str}")
        lines.append("")

    for i in range(species_coeff.shape[0]):
        index = np.where(species_coeff[i, :] != 0)[0]
        if len(index):
            r_terms = [f'R[{j}, :]' for j in index]
            c_terms = [f'+{species_coeff[i, j]}' if species_coeff[i, j] > 0
                       else f'{species_coeff[i, j]}' for j in index]
            rc = ''.join(
                o[0][0] + o[1] if abs(int(float(o[0]))) == 1 else o[0] + "*" + o[1]
                for o in zip(c_terms, r_terms))
            if rc.startswith('+'):
                rc = rc[1:]
        else:
            rc = "0"
        lines.append(f"   D[{i}, :] = {rc}")

    lines.append("   if(type(R) is mp.matrix):")
    if mkm.one_type_site_and_single_occu:
        lines.append(f"      D[0, 0:{ns}] = 1.0")
        lines.append(f"      D[0, {ns}:] = 0.0")
    else:
        for entry in mkm.normalize_cvr:
            lines.append(f"      D[{entry[0]}, :] = 0.0")
            for j, idx in enumerate(entry[2]):
                lines.append(f"      D[{entry[0]}, {idx}] = {entry[3][j]}")

    if constrain_list:
        for c_list in constrain_list:
            lines.append(f"      D[{c_list[0][0]}, :] = 0.0")
            for idx, a in c_list:
                lines.append(
                    f"      D[{c_list[0][0]}, {idx}] = mp.mpf('{a}')")
    lines += ["   return D", "", ""]
    return "\n".join(lines)


def write_F_and_J_function(mkm: MKM, fname: str) -> None:
    """Write the ODE/Jacobian source file for *mkm*.

    The generated module defines ``calc_F(y, K, R, D, gas=None)`` and
    ``calc_J(y, K, R, D, gas=None)``.  As a side effect, ``mkm.normalize_cvr``
    and ``mkm.constrain_list`` are populated.
    """
    source = _generate_rate_source(mkm, fname_header=True)
    with open(fname, "w") as fh:
        fh.write(source)


def build_rate_functions(mkm: MKM) -> MKM:
    """Compile the network ODE/Jacobian in memory and attach them to *mkm*.

    This is the code-callable alternative to :func:`write_F_and_J_function`.
    It does not create any temporary files.
    """
    source = _generate_rate_source(mkm, fname_header=False)
    namespace: Dict = {}
    exec(compile(source, "<catmkm-generated>", "exec"), namespace)
    mkm.calc_F = namespace["calc_F"]
    mkm.calc_J = namespace["calc_J"]
    return mkm
