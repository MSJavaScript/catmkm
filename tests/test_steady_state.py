import numpy as np

from catmkm import load_json, run_steady_state
from catmkm.io import build_model
from catmkm.model import build_rate_functions, get_rate_constants, random_guess
from catmkm.steady_state import single_point_calculation


EXAMPLE = "examples3/example_basic.json"


def test_run_steady_state_basic():
    result = run_steady_state(load_json(EXAMPLE))
    point = result["results"][0]
    assert abs(sum(point["coverage"]) - 1.0) < 1e-8
    assert len(point["rates"]) == 3
    # The three elementary rates are consistent with the single path.
    assert abs(point["rates"][0] - point["rates"][2]) < 1e-18


def test_low_level_single_point():
    obj = load_json(EXAMPLE)
    mkm = build_model(obj)
    build_rate_functions(mkm)
    T = 500
    mkm.gas = np.array([g.pressure[0] for g in mkm.gas_list])
    barriers = np.array(obj["barriers"], dtype=float)
    mkm.rate_constants = get_rate_constants(mkm, barriers, T, 0)
    mkm.rate_constants_mpf = get_rate_constants(mkm, barriers, T, mkm.data_ndigit)

    y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
    coverage, rates = single_point_calculation(
        {"ode": {"ode_eps": 1e-6, "ode_nmax": 30},
         "newton": {"newton_eps": "1.0e-100", "newton_nmax": 30}},
        y0, mkm)

    assert abs(sum(float(c) for c in coverage) - 1.0) < 1e-8
    assert abs(float(rates[0]) - float(rates[2])) < 1e-15


def test_multisite_example():
    obj = load_json("examples3/difficult_CO0.json")
    result = run_steady_state(obj)
    point = result["results"][0]
    coverage = dict(zip(result["surface_species"], point["coverage"]))
    total_1 = sum(v for k, v in coverage.items() if "#1" in k)
    total_2 = sum(v for k, v in coverage.items() if "#2" in k)
    assert abs(total_1 - 0.95) < 1e-8
    assert abs(total_2 - 0.05) < 1e-8
