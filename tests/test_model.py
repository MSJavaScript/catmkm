import numpy as np
import pytest

from catmkm.model import (
    GasSpecies,
    MKM,
    build_rate_functions,
    get_rate_constants,
    parse_network,
    random_guess,
    write_F_and_J_function,
)


def basic_network():
    mkm = MKM()
    mkm.gas_list = [
        GasSpecies("CO", [1.0e-10]),
        GasSpecies("O2", [1.0e-10]),
        GasSpecies("CO2", [1.0e10]),
    ]
    parse_network(mkm, [
        "2*# + CO <-> CO##",
        "2*# + O2 <-> 2*O#",
        "CO## + O# <-> CO2 + 3*#",
    ])
    return mkm


def test_parse_network_and_sites():
    mkm = basic_network()
    assert mkm.nreactions == 3
    assert mkm.surface_species == ["#", "CO##", "O#"]
    assert mkm.gas_species == ["CO", "CO2", "O2"]
    assert mkm.one_type_site_and_single_occu is False


def test_build_rate_functions_dimensions():
    mkm = basic_network()
    build_rate_functions(mkm)
    barriers = np.array([[0, -0.22312], [0.82186, -1.28814], [0.72, -1.575498]])
    mkm.rate_constants = get_rate_constants(mkm, barriers, 500, 0)
    mkm.rate_constants_mpf = get_rate_constants(mkm, barriers, 500, 15)
    mkm.gas = np.array([1e-10, 1e-10, 1e10])
    y = random_guess(mkm.normalize_cvr, mkm.constrain_list)
    F = mkm.Ft(0.0, y)
    assert F.shape == (mkm.nspecies,)
    J = mkm.Jt(0.0, y)
    assert J.shape == (mkm.nspecies, mkm.nspecies)


def test_write_F_and_J_function(tmp_path):
    mkm = basic_network()
    path = tmp_path / "F_and_J.py"
    write_F_and_J_function(mkm, str(path))
    text = path.read_text()
    assert "def calc_F" in text
    assert "def calc_J" in text


def test_rank_deficient_constraints_are_detected():
    from catmkm.io import build_model, load_json

    mkm = build_model(load_json("examples3/difficult_C2H6O2_O2.json"))
    build_rate_functions(mkm)
    assert len(mkm.constrain_list) > 0
