"""Input parsing and high-level drivers for catmkm.

Calculations can be driven either from Python::

    from catmkm import run_input
    result = run_input({"reactions": [...], ...})

or from the command line::

    catmkm input.json --output result.json

The reaction-path mode does not require temperatures or barriers:

.. code-block:: json

    {
      "reactions": ["O2 + # <-> O2#", "..."],
      "total_reactions": ["O2 + 4*H <-> 2*H2O"],
      "auto_catalytic_species": ["OOH#"]
    }
"""

from __future__ import annotations

import json
import pickle
from typing import List, Optional, Union

import numpy as np

from .drc import DRC
from .map import fit_linear_relationship, get_data_from_linear_relationship
from .model import (
    GasSpecies,
    MKM,
    build_rate_functions,
    get_rate_constants,
    parse_network,
    random_guess,
)
from .reaction_path import (
    ReactionPathResult,
    find_reaction_paths,
    find_reaction_paths_sharded,
)
from .reactor import simulate_cstr
from .steady_state import single_point_calculation
from .thermo import correct_barriers_with_gas


class InputError(ValueError):
    """Raised for invalid or missing input data."""


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(source):
    """Load a JSON file (path/file object) and return a dictionary."""
    try:
        if hasattr(source, "read"):
            return json.load(source)
        with open(source) as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise InputError(f"The file {source!r} was not found.") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {source!r}: {exc}") from exc


def _temperatures(obj) -> list:
    value = obj.get("temperature", obj.get("T"))
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(value)]


def _index_grid(counts):
    """Cartesian product of indices (same ordering as the old helper)."""
    n = len(counts)
    m = int(np.prod(counts)) if n else 1
    counter = np.zeros((n,), dtype=int)
    index = np.zeros((m, n), dtype=int)
    for i in range(1, m):
        for j in range(n):
            if counter[j] < counts[j] - 1:
                counter[j] += 1
                break
            elif j < n - 1:
                counter[j] = 0
        index[i, :] = counter
    return index


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _gas_list_from_info(gas_info):
    gas_list, redundant = [], []
    for gas in gas_info or []:
        if gas.get("redundant_species", False):
            redundant.append(gas["name"])
            continue
        name = gas.get("name")
        if name is None:
            raise InputError("Every entry of 'gas_info' needs a 'name'.")
        pressure = gas.get("pressure", [1.0])
        if isinstance(pressure, (int, float)):
            pressure = [float(pressure)]
        obj = GasSpecies(name, [float(p) for p in pressure], gas.get("charge", 0))
        obj.diffusion_constant = gas.get("diffusion_constant")
        obj.effective_cell_size = gas.get("effective_cell_size", 0.0)
        obj.sticking_coeff = gas.get("sticking_coeff")
        obj.mole_mass = gas.get("mole_mass")
        obj.thermo_params = {k: v for k, v in gas.items()
                             if k not in ("name", "pressure", "charge")}
        obj.thermo_params.update(gas.get("thermo_params", {}) or {})
        gas_list.append(obj)
    return gas_list, redundant


def build_model(obj: dict, *, change_gas: bool = False,
                for_path: bool = False, extra_reactions=None) -> MKM:
    """Construct an :class:`MKM` from an input dictionary.

    The input may contain only ``reactions`` for reaction-path searches; all
    thermodynamic fields are optional in that case.
    """
    reactions = obj.get("reactions")
    if extra_reactions is not None:
        reactions = extra_reactions
    if not reactions:
        raise InputError("'reactions' is missing or empty in the input.")

    mkm = MKM()
    temps = _temperatures(obj)
    mkm.T = temps if temps else None
    mkm.temperature = temps[0] if temps else None

    gas_list, redundant = _gas_list_from_info(obj.get("gas_info", []))
    mkm.gas_list = gas_list
    mkm.redundant_gas_names = redundant

    mkm.sites_total_coverage = obj.get("set_multi_site")
    mkm.sites_number_density = obj.get("sites_number_density")
    mkm.data_ndigit = obj.get("data_ndigit", 15)
    mkm.change_gas = bool(change_gas)

    # Optional collision-theory settings for adsorption.
    for entry in obj.get("collision_gas", []) or []:
        for gas in mkm.gas_list:
            if gas.name == entry.get("name"):
                gas.mole_mass = entry.get("M", gas.mole_mass)
                gas.sticking_coeff = entry.get("S0", gas.sticking_coeff)
        if mkm.sites_number_density is None and "L_tot" in entry:
            mkm.sites_number_density = {}
    if obj.get("sites_number_density") is None and obj.get("collision_gas"):
        # Use the first site's density for the simplified collision input.
        L_tot = obj["collision_gas"][0].get("L_tot")
        if L_tot is not None:
            mkm.sites_number_density = {"#": L_tot, "#1": L_tot, "#2": L_tot}

    check_gas = not for_path
    parse_network(mkm, reactions, True, redundant, check_gas=check_gas)
    return mkm


def _parse_barriers(obj: dict, mkm: MKM):
    """Return ``(barrier_data, ads_dict, ndata)``."""
    barrier_ori = obj.get("barriers")
    if barrier_ori is None:
        return None, None, 0
    if not isinstance(barrier_ori, list) or len(barrier_ori) == 0:
        raise InputError("'barriers' must be a non-empty list.")

    if isinstance(barrier_ori[0], dict):
        ndata = len(barrier_ori[0].get("data", []))
        ads_dict = {}
        for b in barrier_ori:
            try:
                data = np.array(b["data"], dtype=float)
            except KeyError as exc:
                raise InputError(
                    "Each barrier entry must be {\"name\": ..., \"data\": ...}.") from exc
            except ValueError as exc:
                raise InputError(f"Cannot convert barrier data for {b.get('name')!r}.") from exc
            if len(data) != ndata:
                raise InputError("All barrier data arrays must have the same length.")
            ads_dict[b["name"]] = data
        return None, ads_dict, ndata

    if isinstance(barrier_ori[0], list):
        try:
            barrier_data = np.array(barrier_ori, dtype=float)
        except ValueError as exc:
            raise InputError("All barrier rows must have the same length.") from exc
        if barrier_data.shape[0] != mkm.nreactions:
            raise InputError(
                "The number of barriers is not equal to the number of reactions.")
        ndata = int(barrier_data.shape[1] * 0.5)
        return barrier_data, None, ndata

    raise InputError("Wrong type in 'barriers'.")


def _single_barrier_data(barrier_data, ads_dict):
    if barrier_data is not None:
        return barrier_data.copy()
    if ads_dict is not None:
        return {key: value[0] for key, value in ads_dict.items()}
    return None


def _settings(obj):
    ode = dict(obj.get("ode_setting") or {})
    newton = dict(obj.get("newton_setting") or {})
    return {"ode": ode, "newton": newton,
            "nmax_try": obj.get("nmax_try", 20)}


def _correct_type(obj):
    if obj.get("thermo_correction", True) is False:
        return None
    return obj.get("gas_correct_type")


def _output_header(mkm: MKM, gas_list=None):
    return {
        "reactions": [str(r) for r in mkm.reaction_list],
        "surface_species": list(mkm.surface_species),
        "gas_species": [g.name for g in (gas_list or mkm.gas_list)],
        "results": [],
    }


# ---------------------------------------------------------------------------
# Steady-state driver
# ---------------------------------------------------------------------------

def run_steady_state(obj: dict) -> dict:
    """Run a (multi-)temperature steady-state calculation, DRC included."""
    mkm = build_model(obj)
    settings = _settings(obj)
    temps = mkm.T
    if not temps:
        raise InputError("'temperature' is required for a steady-state run.")
    if not mkm.gas_list:
        raise InputError("'gas_info' is required for a steady-state run.")

    barrier_data, ads_dict, ndata = _parse_barriers(obj, mkm)
    if ads_dict is not None and ndata > 1:
        raise InputError(
            "Multiple values per species are only supported by activity maps.")
    if barrier_data is not None and ndata > 1 and "activity_map" not in obj:
        raise InputError(
            "Multiple barrier values require an 'activity_map' section.")

    correct_type = _correct_type(obj)
    drc_delta_e = obj.get("drc_delta_e", -1.0)

    build_rate_functions(mkm)
    output = _output_header(mkm)

    y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
    gas_counts = [max(1, len(g.pressure)) for g in mkm.gas_list]
    gas_n_index = _index_grid(gas_counts)

    for T in temps:
        for ig in range(gas_n_index.shape[0]):
            mkm.gas = np.array([
                mkm.gas_list[i].pressure[gas_n_index[ig][i]]
                for i in range(len(mkm.gas_list))])
            print(f"Temperature: {T:<15.3f}")
            print("Gas pressure: ")
            for i, gas in enumerate(mkm.gas_list):
                print(f"Name:{gas.name:<15s} pressure:{mkm.gas[i]:<15.6e}")

            single_data = _single_barrier_data(barrier_data, ads_dict)
            if single_data is not None:
                single_data = correct_barriers_with_gas(
                    mkm, single_data, T, correct_type, obj.get("gas_info", []))
            mkm.rate_constants = get_rate_constants(mkm, single_data, T, 0)
            mkm.rate_constants_mpf = get_rate_constants(
                mkm, single_data, T, mkm.data_ndigit)

            coverage, reaction_rates = single_point_calculation(
                settings, y0, mkm)
            collect_header, collect_drc = DRC(
                coverage0=coverage, rate0=reaction_rates, delta_e=drc_delta_e,
                T=T, params=settings, mkm=mkm, single_barrier_data=single_data)

            result = {
                "T": T,
                "pressure": np.asarray(mkm.gas).tolist(),
                "coverage": [float(coverage[ns]) for ns in range(mkm.nspecies)],
                "rates": [float(reaction_rates[nr]) for nr in range(mkm.nreactions)],
            }
            if collect_header is not None:
                output.setdefault("collect_header", collect_header)
                result["DRC"] = [[float(v) for v in drc] for drc in collect_drc]
            output["results"].append(result)

            for ns in range(mkm.nspecies):
                print(f"{mkm.surface_species[ns]:<15s}{float(coverage[ns]):<15.6e}")
            for nr in range(mkm.nreactions):
                print(f"{str(mkm.reaction_list[nr]):<50s}"
                      f"{float(reaction_rates[nr]):<15.6e}")
            print("*******************")
    return output


# ---------------------------------------------------------------------------
# Activity-map driver
# ---------------------------------------------------------------------------

def _parse_map(obj: dict, mkm: MKM):
    a = obj.get("activity_map")
    if not a:
        return None
    if "descriptor" not in a or "map_grid" not in a:
        raise InputError("'activity_map' needs 'descriptor' and 'map_grid'.")
    ndes = len(a["descriptor"])
    if ndes not in (1, 2):
        raise InputError("The number of descriptors must be 1 or 2.")

    des_names, des_datas, des_grids = [], [], []
    discrete_grid = False
    for i in range(ndes):
        des_names.append(a["descriptor"][i]["name"])
        des_datas.append(np.array(a["descriptor"][i]["data"], dtype=float))
        grid = a["map_grid"][i]
        if isinstance(grid, str):
            ra, rb, N = grid.split(",")
            des_grids.append(np.linspace(float(ra), float(rb), int(N)))
        elif isinstance(grid, list):
            discrete_grid = True
            des_grids.append(np.array(grid, dtype=float))
        else:
            raise InputError("Wrong type of 'map_grid'.")
    if ndes == 2 and len(des_datas[0]) != len(des_datas[1]):
        raise InputError("Descriptor data arrays must have the same length.")
    if discrete_grid and ndes == 2 and len(des_grids[0]) != len(des_grids[1]):
        raise InputError("Discrete 2D grids must have matching lengths.")
    return {
        "ndes": ndes,
        "ndes_data": len(des_datas[0]),
        "des_names": des_names,
        "des_datas": des_datas,
        "discrete_grid": discrete_grid,
        "des_grids": des_grids,
    }


def run_activity_map(obj: dict) -> dict:
    """Run an activity-map calculation over one or two descriptors."""
    mkm = build_model(obj)
    settings = _settings(obj)
    temps = mkm.T
    if not temps:
        raise InputError("'temperature' is required for an activity map.")
    if not mkm.gas_list:
        raise InputError("'gas_info' is required for an activity map.")

    map_params = _parse_map(obj, mkm)
    if map_params is None:
        raise InputError("'activity_map' is missing.")
    barrier_data, ads_dict, ndata = _parse_barriers(obj, mkm)
    if ndata != map_params["ndes_data"]:
        raise InputError(
            "The barrier data length is inconsistent with the descriptor data.")
    if ndata < 2:
        raise InputError("An activity map needs at least two data points.")

    correct_type = _correct_type(obj)
    drc_delta_e = obj.get("drc_delta_e", -1.0)
    build_rate_functions(mkm)

    map_coeffs, map_residuals, map_r2 = fit_linear_relationship(
        mkm, barrier_data, ads_dict, map_params)

    output = _output_header(mkm)
    map_fitting_name, map_fitting_coeffs = [], []
    map_fitting_r2, map_fitting_residuals = [], []
    if isinstance(map_coeffs, np.ndarray):
        map_fitting_type = "EaH"
        for ir in range(mkm.nreactions):
            map_fitting_name.append(str(mkm.reaction_list[ir]))
            map_fitting_coeffs.append(
                [float(v) for v in map_coeffs[ir * 2, :]]
                + [float(v) for v in map_coeffs[ir * 2 + 1, :]])
            map_fitting_r2.append([float(map_r2[ir * 2]),
                                   float(map_r2[ir * 2 + 1])])
            map_fitting_residuals.append(
                [float(v) for v in map_residuals[ir * 2, :]]
                + [float(v) for v in map_residuals[ir * 2 + 1, :]])
    else:
        map_fitting_type = "ads_energy"
        for key in map_coeffs:
            map_fitting_name.append(key)
            map_fitting_coeffs.append([float(v) for v in map_coeffs[key]])
            map_fitting_r2.append(float(map_r2[key]))
            map_fitting_residuals.append(
                [float(v) for v in map_residuals[key]])
    output["linear_relationship"] = {
        "type": map_fitting_type,
        "name": map_fitting_name,
        "coeffs": map_fitting_coeffs,
        "R2": map_fitting_r2,
        "residuals": map_fitting_residuals,
        "des_names": map_params["des_names"],
    }

    gas_counts = [max(1, len(g.pressure)) for g in mkm.gas_list]
    gas_n_index = _index_grid(gas_counts)
    for T in temps:
        for ig in range(gas_n_index.shape[0]):
            mkm.gas = np.array([
                mkm.gas_list[i].pressure[gas_n_index[ig][i]]
                for i in range(len(mkm.gas_list))])
            print(f"Temperature: {T:<15.3f}")

            if map_params["ndes"] == 1:
                points = [(des1, None) for des1 in map_params["des_grids"][0]]
            else:
                points = [(des1, des2)
                          for des1 in map_params["des_grids"][0]
                          for des2 in map_params["des_grids"][1]]

            for des1, des2 in points:
                label = f"{des1:.3f}" if des2 is None else f"({des1:.3f}, {des2:.3f})"
                print("Point", label)
                fdata = get_data_from_linear_relationship(
                    des1, des2, mkm, map_coeffs)
                correct_barriers_with_gas(
                    mkm, fdata, T, correct_type, obj.get("gas_info", []))
                mkm.rate_constants = get_rate_constants(mkm, fdata, T, 0)
                mkm.rate_constants_mpf = get_rate_constants(
                    mkm, fdata, T, mkm.data_ndigit)

                y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
                coverage, reaction_rates = single_point_calculation(
                    settings, y0, mkm)
                collect_header, collect_drc = DRC(
                    coverage0=coverage, rate0=reaction_rates,
                    delta_e=drc_delta_e, T=T, params=settings, mkm=mkm,
                    single_barrier_data=fdata)

                descriptor = ([float(des1)] if des2 is None
                              else [float(des1), float(des2)])
                result = {
                    "T": T,
                    "pressure": np.asarray(mkm.gas).tolist(),
                    "descriptor": descriptor,
                    "coverage": [float(coverage[ns]) for ns in range(mkm.nspecies)],
                    "rates": [float(reaction_rates[nr])
                              for nr in range(mkm.nreactions)],
                }
                if collect_header is not None:
                    output.setdefault("collect_header", collect_header)
                    result["DRC"] = [[float(v) for v in drc]
                                     for drc in collect_drc]
                output["results"].append(result)
    return output


# ---------------------------------------------------------------------------
# Reaction-path driver
# ---------------------------------------------------------------------------

def run_reaction_path(obj: dict) -> Union[ReactionPathResult,
                                          List[ReactionPathResult]]:
    """Search all reaction paths.

    Only ``reactions`` and the overall ``total_reactions`` are required.
    ``temperature``, ``barriers`` and ``gas_info`` are optional and ignored.

    Two kinds of input are accepted:

    * the classic single search with ``auto_catalytic_species`` set to a list,
      and
    * a sharded search.  Set ``shard_species`` to ``"each"`` (every surface
      intermediate), ``"all"`` (surface intermediates plus gas species),
      ``"auto"`` (detected auto-catalytic species), or an explicit list.  A
      list of lists gives explicit multi-species seed groups, one search per
      group; those are needed for pathways that require several species to be
      present initially.  ``merge`` (default True) combines the shard results
      and removes duplicate paths; ``n_jobs`` runs shards in parallel.
    """
    mkm = build_model(obj, for_path=True)
    opts = dict(obj.get("find_reaction_path") or {})
    total_reactions = opts.get("total_reactions", obj.get("total_reactions"))
    if not total_reactions:
        raise InputError(
            "Reaction-path search needs 'total_reactions' (overall reactions).")

    mkm_gas = build_model(obj, for_path=True, extra_reactions=total_reactions)

    auto = opts.get("auto_catalytic_species",
                    obj.get("auto_catalytic_species", None))
    center = opts.get("center_species", obj.get("center_species", []))
    shards = opts.get("shard_species", obj.get("shard_species"))
    if auto in ("each", "all", "auto"):
        shards = auto
        auto = None

    common = dict(
        find_reverse=opts.get("find_reverse", True),
        center_species=center,
        check_rank=opts.get("check_rank", True),
        max_walk_node=opts.get("max_walk_node", -1),
        print_path=opts.get("print_path", False),
        save_depth=opts.get("save_depth", 2),
        nstat=opts.get("nstat", -1),
        nwait=opts.get("nwait", -1),
        verbose=opts.get("verbose", True),
        random_seed=opts.get("random_seed"),
    )

    if shards is not None:
        return find_reaction_paths_sharded(
            mkm,
            mkm_gas,
            shards=shards,
            merge=opts.get("merge", True),
            n_jobs=opts.get("n_jobs", 1),
            verbose=common.pop("verbose"),
            **common,
        )

    return find_reaction_paths(
        mkm, mkm_gas, auto_catalytic_species=auto, **common)


# ---------------------------------------------------------------------------
# CSTR driver
# ---------------------------------------------------------------------------

def run_cstr(obj: dict) -> dict:
    """Run a CSTR simulation with evolving gas partial pressures."""
    cstr = obj.get("cstr") or {}
    mkm = build_model(obj, change_gas=True)
    temps = mkm.T or [mkm.temperature]
    if temps[0] is None:
        raise InputError("'temperature' is required for a CSTR run.")
    T = temps[0]
    mkm.T = [T]

    barrier_data, ads_dict, ndata = _parse_barriers(obj, mkm)
    single_data = _single_barrier_data(barrier_data, ads_dict)
    single_data = correct_barriers_with_gas(
        mkm, single_data, T, _correct_type(obj), obj.get("gas_info", []))
    mkm.rate_constants = get_rate_constants(mkm, single_data, T, 0)
    mkm.rate_constants_mpf = get_rate_constants(
        mkm, single_data, T, mkm.data_ndigit)

    build_rate_functions(mkm)
    feed = cstr.get("feed")
    if isinstance(feed, list):
        feed = {gas.name: feed[i] for i, gas in enumerate(mkm.gas_list)}
    result = simulate_cstr(
        mkm,
        feed=feed,
        residence_time=cstr.get("residence_time", 1.0),
        gas_factor=cstr.get("gas_factor", 1.0),
        p0=cstr.get("initial_pressure"),
        max_time=cstr.get("max_time"),
        surface_tol=cstr.get("surface_tol", 1.0e-10),
        gas_tol=cstr.get("gas_tol", 1.0e-10),
        settings=_settings(obj),
    )
    return {
        "T": T,
        "gas_names": result.gas_names,
        "surface_species": result.surface_species,
        "time": result.time.tolist(),
        "coverage": result.coverage.tolist(),
        "pressure": result.pressure.tolist(),
        "reaction_rates": result.reaction_rates.tolist(),
        "steady_coverage": result.steady_coverage.tolist(),
        "steady_pressure": result.steady_pressure.tolist(),
        "success": bool(result.success),
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def detect_mode(obj: dict) -> str:
    explicit = obj.get("mode") or obj.get("task")
    if explicit:
        return str(explicit).replace("_", "-")
    if "find_reaction_path" in obj or "total_reactions" in obj:
        return "reaction-path"
    if "cstr" in obj:
        return "cstr"
    if "activity_map" in obj:
        return "activity-map"
    return "steady-state"


def run_input(obj: dict, mode: Optional[str] = None):
    """Run the calculation described by *obj* and return the result object."""
    if not isinstance(obj, dict):
        raise InputError("Input must be a dictionary.")
    if "electro_chem_calculation" in obj:
        raise InputError(
            "Electrochemical (Poisson-Nernst-Planck) calculations were removed "
            "from catmkm.")
    if "sol_reacs" in obj or "species_concentration" in obj:
        raise InputError(
            "Pure-phase equilibrium calculations (mkm_sol_equ) were removed "
            "from catmkm.")
    mode = (mode or detect_mode(obj)).replace("_", "-")
    if mode in ("reaction-path", "path", "reaction_path"):
        return run_reaction_path(obj)
    if mode in ("cstr", "reactor"):
        return run_cstr(obj)
    if mode in ("activity-map", "map", "activity_map"):
        return run_activity_map(obj)
    if mode in ("steady-state", "steady", "single", "rate"):
        return run_steady_state(obj)
    raise InputError(f"Unknown mode {mode!r}.")


def run_file(path: str, output: Optional[str] = None,
             mode: Optional[str] = None):
    """Load a JSON input file, run it, and optionally save the result."""
    obj = load_json(path)
    result = run_input(obj, mode=mode)
    if output:
        save_result(result, output)
    return result


def save_result(result, path: str) -> None:
    """Save a result as JSON, or pickle for non-serialisable results."""
    if isinstance(result, ReactionPathResult) or (
            isinstance(result, (list, tuple))
            and all(isinstance(r, ReactionPathResult) for r in result)):
        with open(path, "wb") as fh:
            pickle.dump(result, fh)
        return
    try:
        with open(path, "w") as fh:
            json.dump(result, fh, indent=4)
    except TypeError:
        with open(path, "wb") as fh:
            pickle.dump(result, fh)


# Backwards-compatible helper mimicking the old script: run and write output.
def read_input_json(json_name: str, output: str = "output.json"):
    result = run_file(json_name)
    if isinstance(result, ReactionPathResult) or (
            isinstance(result, list)
            and all(isinstance(r, ReactionPathResult) for r in result)):
        import uuid
        output = uuid.uuid4().hex + ".pkl"
    save_result(result, output)
    return result
