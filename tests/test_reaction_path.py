from catmkm import load_json, run_input, run_reaction_path
from catmkm.reaction_path import detect_autocatalytic_species
from catmkm.io import build_model


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


def test_detect_autocatalytic_species():
    mkm = build_model(CLEAN, for_path=True)
    species = detect_autocatalytic_species(mkm)
    assert "OOH#" in species
    assert "#" not in species


def test_clean_path_input_needs_no_thermo_data():
    result = run_input(CLEAN)
    assert len(result.paths) == 16
    assert len(result.index) == 1
    assert len(result.index[0]) == 16


def test_explicit_auto_list_matches_detection():
    obj = load_json("examples3/example_OER_path.json")
    with_explicit = run_reaction_path(obj)
    del obj["find_reaction_path"]["auto_catalytic_species"]
    with_detection = run_reaction_path(obj)
    assert len(with_explicit.paths) == len(with_detection.paths)


def test_disable_autocatalytic_gives_fewer_paths():
    obj = load_json("examples3/example_OER_path.json")
    obj["find_reaction_path"]["auto_catalytic_species"] = []
    result = run_reaction_path(obj)
    assert len(result.paths) < 16
