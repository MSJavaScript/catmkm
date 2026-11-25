"""catmkm - a microkinetic modelling toolkit.

The package is organised around a small number of building blocks:

``catmkm.model``
    Reaction-network parsing and the :class:`~catmkm.model.MKM` container.
    ``write_F_and_J_function`` generates the ODE residual ``F`` and its
    Jacobian ``J`` for a network.

``catmkm.steady_state``
    ``single_point_calculation`` combines a robust ODE integration with an
    arbitrary-precision Newton iteration.

``catmkm.drc``
    Degree-of-rate-control analysis.

``catmkm.thermo``
    Thermodynamic corrections for gas-phase species (Shomate/ideal gas).

``catmkm.reaction_path``
    Exhaustive (and randomised) reaction-path search.

``catmkm.plotting``
    Scaling-relation and activity/mechanism-map drawing helpers.

``catmkm.reactor``
    Continuous stirred-tank reactor (CSTR) simulation in which gas partial
    pressures are allowed to change with time.
"""

from .model import (
    GasSpecies,
    Gas_species,
    MKM,
    MKM_model,
    Reaction,
    Species,
    build_rate_functions,
    get_rate_constants,
    parse_network,
    random_guess,
    write_F_and_J_function,
)
from .steady_state import (
    SteadyStateSettings,
    newton_iteration,
    ode_iteration,
    single_point_calculation,
)
from .drc import DRC
from .thermo import correct_barriers_with_gas, thermo_correction_of_gas
from .reaction_path import (
    ReactionPathResult,
    detect_autocatalytic_species,
    find_reaction_path_dynamic,
    find_reaction_paths,
    find_reaction_paths_sharded,
    list_shard_species,
    merge_reaction_paths,
)
from .reactor import CSTRResult, simulate_cstr
from .map import fit_linear_relationship, get_data_from_linear_relationship
from .io import (
    InputError,
    build_model,
    detect_mode,
    load_json,
    run_activity_map,
    run_cstr,
    run_file,
    run_input,
    run_reaction_path,
    run_steady_state,
    save_result,
)

__all__ = [
    "GasSpecies",
    "Gas_species",
    "MKM",
    "MKM_model",
    "Reaction",
    "Species",
    "build_rate_functions",
    "get_rate_constants",
    "parse_network",
    "random_guess",
    "write_F_and_J_function",
    "SteadyStateSettings",
    "newton_iteration",
    "ode_iteration",
    "single_point_calculation",
    "DRC",
    "correct_barriers_with_gas",
    "thermo_correction_of_gas",
    "ReactionPathResult",
    "detect_autocatalytic_species",
    "find_reaction_path_dynamic",
    "find_reaction_paths",
    "find_reaction_paths_sharded",
    "list_shard_species",
    "merge_reaction_paths",
    "CSTRResult",
    "simulate_cstr",
    "fit_linear_relationship",
    "get_data_from_linear_relationship",
    "InputError",
    "build_model",
    "detect_mode",
    "load_json",
    "run_activity_map",
    "run_cstr",
    "run_file",
    "run_input",
    "run_reaction_path",
    "run_steady_state",
    "save_result",
]

__version__ = "0.1.0"
