import numpy as np

from catmkm import load_json, run_cstr


def test_cstr_reaches_steady_state():
    # Use a feed with significant conversions so that the gas composition
    # visibly evolves away from the feed.
    np.random.seed(0)
    obj = load_json("examples3/example_basic.json")
    obj["cstr"] = {
        "feed": {"CO": 1.0, "O2": 0.5, "CO2": 0.0},
        "residence_time": 1.0,
        "gas_factor": 1.0e-3,
        "max_time": 100.0,
    }
    result = run_cstr(obj)
    assert result["success"]

    surface = result["surface_species"]
    coverage = result["steady_coverage"]
    site_weighted = sum(c * name.count("#") for c, name in zip(coverage, surface))
    assert abs(site_weighted - 1.0) < 1e-7

    # The feed is CO + 0.5 O2 -> CO2, so the gas pressures must move.
    assert not np.allclose(result["steady_pressure"], [1.0, 0.5, 0.0])
    assert result["steady_pressure"][2] > 0.0

    # Stoichiometric consistency of the elementary rates at steady state.
    rate_co, rate_o2, rate_co2 = result["reaction_rates"]
    assert abs(rate_co - rate_co2) / abs(rate_co) < 1e-4
    assert abs(rate_o2 - 0.5 * rate_co) / abs(rate_co) < 1e-4

    assert len(result["time"]) == len(result["coverage"][0])


def test_cstr_requires_change_gas():
    from catmkm.model import MKM, GasSpecies, parse_network
    from catmkm.reactor import simulate_cstr

    mkm = MKM()
    mkm.gas_list = [GasSpecies("A", [1.0])]
    parse_network(mkm, ["A + # <-> A#"])
    try:
        simulate_cstr(mkm, feed={"A": 1.0})
    except ValueError as exc:
        assert "change_gas" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
