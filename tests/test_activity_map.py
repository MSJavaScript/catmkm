import numpy as np

from catmkm import load_json, run_activity_map


def test_activity_map_matches_known_point():
    obj = load_json("examples3/example_map_CO_oxi.json")
    # Keep the smoke test small; the production grids contain 30x30 points.
    obj["activity_map"]["map_grid"] = ["-0.5,0.5,2", "-1,0,2"]
    result = run_activity_map(obj)

    assert len(result["results"]) == 4
    assert result["linear_relationship"]["type"] == "ads_energy"

    point = result["results"][0]
    assert point["descriptor"] == [-0.5, -1.0]
    # Reference values produced by the legacy mkm_map.py driver.
    assert abs(point["coverage"][0] - 1.406349195595342e-26) < 1e-30
    assert abs(point["rates"][0] - 1.0232852994681855e-18) < 1e-23


def test_activity_map_requires_descriptors():
    from catmkm import InputError, run_input

    obj = {
        "temperature": [500],
        "reactions": ["2*# + CO <-> CO##"],
        "barriers": [[0, -0.2]],
        "gas_info": [{"name": "CO", "pressure": [1.0]}],
        "activity_map": {},
    }
    try:
        run_input(obj)
    except InputError as exc:
        assert "activity_map" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected InputError")
