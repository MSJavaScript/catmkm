import numpy as np

from catmkm import (
    build_model,
    find_reaction_paths,
    find_reaction_paths_sharded,
    list_shard_species,
    load_json,
    merge_reaction_paths,
    run_input,
    run_reaction_path,
)


CLEAN = {
    "reactions": [
        "O2 + # <-> O2#",
        "O2# + # <-> 2*O#",
        "OOH# + # <-> OH# + O#",
        "H2O + O# + # <-> 2*OH#",
        "O# + H <-> OH#",
        "OH# + H <-> H2O + #",
        "O2# + H <-> OOH#",
        "OOH# + H <-> H2O + O#",
    ],
    "total_reactions": ["O2 + 4*H <-> 2*H2O"],
}


def _models():
    mkm = build_model(CLEAN, for_path=True)
    mkm_gas = build_model(
        {"reactions": CLEAN["total_reactions"]}, for_path=True)
    return mkm, mkm_gas


def test_list_shard_species_contains_all_intermediates():
    mkm, _ = _models()
    shards = list_shard_species(mkm)
    assert shards == ["O#", "O2#", "OH#", "OOH#"]
    assert "#" not in shards


def test_list_shard_species_can_include_gas_species():
    mkm, _ = _models()
    with_gas = list_shard_species(mkm, include_gas=True)
    assert with_gas[:4] == ["O#", "O2#", "OH#", "OOH#"]
    assert "O2" in with_gas and "H2O" in with_gas and "H" in with_gas
    assert len(with_gas) == len(set(with_gas))


def test_gas_species_is_accepted_as_a_shard_seed():
    mkm, mkm_gas = _models()
    direct = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["H2O"],
        find_reverse=True, check_rank=True, verbose=False,
    )
    sharded = find_reaction_paths_sharded(
        mkm, mkm_gas, shards=["H2O"],
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert _path_keys(sharded) == _path_keys(direct)
    assert sharded.auto_catalytic_species == ["H2O"]


def test_shards_all_equals_each_plus_gas_shards():
    mkm, mkm_gas = _models()
    all_shards = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="all", merge=False,
        find_reverse=True, check_rank=True, verbose=False,
    )
    surface = list_shard_species(mkm)
    gas = [s for s in list_shard_species(mkm, include_gas=True)
           if s not in surface]
    assert len(all_shards) == len(surface) + len(gas)
    assert [r.auto_catalytic_species[0] for r in all_shards] == \
        surface + gas


def test_sharded_search_merges_and_matches_single_search():
    mkm, mkm_gas = _models()
    single = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["OOH#"],
        find_reverse=True, check_rank=True, verbose=False,
    )
    sharded = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="each",
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert len(sharded.paths) == len(single.paths) == 16
    assert len(sharded.index) == 1
    assert len(sharded.index[0]) == 16
    assert sharded.stats["n_shards"] == 4
    # Every sharded path is unique (the merge de-duplicates).
    keys = {tuple(np.where(np.abs(row) > 0.01)[0])
            for row in sharded.paths}
    assert len(keys) == len(sharded.paths)


def test_parallel_shards_give_the_same_result():
    mkm, mkm_gas = _models()
    serial = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="each", n_jobs=1,
        find_reverse=True, check_rank=True, verbose=False,
    )
    parallel = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="each", n_jobs=2,
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert len(parallel.paths) == len(serial.paths)
    assert np.allclose(parallel.paths, serial.paths)
    assert all(np.array_equal(a, b)
               for a, b in zip(parallel.index, serial.index))


def test_merge_deduplicates_repeated_results():
    result = run_reaction_path(load_json("examples3/example_OER_path.json"))
    merged = merge_reaction_paths([result, result])
    assert len(merged.paths) == len(result.paths)
    assert len(merged.index[0]) == len(result.paths)
    assert merged.stats["n_shards"] == 2


def test_merge_accepts_old_paths_index_pickles():
    result = run_reaction_path(load_json("examples3/example_OER_path.json"))
    legacy = [result.paths.tolist(), [i.copy() for i in result.index]]
    merged = merge_reaction_paths([legacy])
    assert len(merged.paths) == len(result.paths)
    assert len(merged.index[0]) == len(result.paths)


def _path_keys(result):
    return {tuple(np.where(np.abs(row) > 0.01)[0]) for row in result.paths}


def test_multi_species_seed_groups_are_supported():
    mkm, mkm_gas = _models()
    direct = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["O#", "OH#"],
        find_reverse=True, check_rank=True, verbose=False,
    )
    grouped = find_reaction_paths_sharded(
        mkm, mkm_gas, shards=[["O#", "OH#"]],
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert _path_keys(grouped) == _path_keys(direct)
    assert len(grouped.paths) == len(direct.paths)
    assert grouped.auto_catalytic_species == ["O#", "OH#"]


def test_single_and_group_shards_can_be_mixed():
    mkm, mkm_gas = _models()
    result = find_reaction_paths_sharded(
        mkm, mkm_gas, shards=["O#", ["OH#", "OOH#"]],
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert result.stats["n_shards"] == 2
    assert _path_keys(result) == _path_keys(find_reaction_paths_sharded(
        mkm, mkm_gas, shards="each",
        find_reverse=True, check_rank=True, verbose=False))


def test_unknown_species_in_seed_group_is_rejected():
    mkm, mkm_gas = _models()
    try:
        find_reaction_paths_sharded(
            mkm, mkm_gas, shards=[["O#", "NOPE#"]], verbose=False)
    except ValueError as exc:
        assert "NOPE#" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown shard should raise")


def test_shard_species_can_be_configured_in_json():
    obj = dict(CLEAN)
    obj["find_reaction_path"] = {
        "shard_species": "each",
        "merge": True,
        "n_jobs": 1,
    }
    result = run_input(obj)
    assert len(result.paths) == 16
    assert result.stats["n_shards"] == 4


def test_streaming_callback_does_not_store_paths():
    mkm, mkm_gas = _models()
    collected = []
    result = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["OOH#"],
        find_reverse=True, check_rank=True, verbose=False,
        store_paths=False,
        on_path=lambda coeff, group: collected.append((coeff.copy(), group)),
    )
    assert len(collected) == 16
    assert result.paths.shape[0] == 0
    assert result.stats["n_solved_path"] == 16


def test_zero_paths_still_pads_the_index_per_total_reaction():
    mkm = build_model(
        {"reactions": ["O2 + # <-> O2#", "O2# + # <-> 2*O#"]}, for_path=True)
    mkm_gas = build_model(
        {"reactions": ["N2 + 3*H <-> 2*NH3", "O2 + 4*H <-> 2*H2O"]},
        for_path=True,
    )
    single = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["O#"],
        find_reverse=True, check_rank=True, verbose=False,
    )
    sharded = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="each",
        find_reverse=True, check_rank=True, verbose=False,
    )
    for result in (single, sharded):
        assert result.paths.shape == (0, 2)
        assert len(result.index) == 2
        assert all(len(ind) == 0 for ind in result.index)
        assert len(result.total_reactions) == 2


def test_unknown_shard_species_is_rejected():
    mkm, mkm_gas = _models()
    try:
        find_reaction_paths_sharded(
            mkm, mkm_gas, shards=["NOPE#"], verbose=False)
    except ValueError as exc:
        assert "NOPE#" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown shard should raise")


def test_merge_rejects_mismatched_networks():
    mkm, mkm_gas = _models()
    small = find_reaction_paths(
        build_model({"reactions": CLEAN["reactions"][:3]}, for_path=True),
        mkm_gas, auto_catalytic_species=[], verbose=False,
    )
    large = find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=["OOH#"], verbose=False)
    try:
        merge_reaction_paths([small, large])
    except ValueError as exc:
        assert "different reaction" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched networks should raise")


def test_explicit_single_shard():
    mkm, mkm_gas = _models()
    result = find_reaction_paths_sharded(
        mkm, mkm_gas, shards="OOH#",
        find_reverse=True, check_rank=True, verbose=False,
    )
    assert len(result.paths) == 16
    assert result.auto_catalytic_species == ["OOH#"]
