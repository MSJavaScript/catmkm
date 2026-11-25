"""Thermodynamic corrections for gas-phase species.

The corrections are used to convert electronic-structure energies plus
vibrational data into Gibbs free energies at the reaction temperature.  Two
families are supported: NIST Shomate polynomials and an ideal-gas
(translational + rotational + vibrational) partition-function approach.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import constants as _c


def ideal_gas(p, T, m, sigma, Tt, wavenums):
    """Ideal-gas free-energy correction (eV) at pressure *p* (bar)."""
    B = _c.R_GAS * 0.001
    C = _c.EV_TO_J_MOL / 1000.0
    # The following constants are logarithms of the collected constants.
    D = 40.33208057775844
    E = -52.636903808053

    def translation(d, mass, T, p):
        logZ = 0.5 * d * (np.log(mass * T) + D) + np.log(T / p) + E
        return -logZ * B * T / C

    def rotation(T, sigma, Tt):
        if len(Tt) == 1:
            logZ = np.log(T / (sigma * Tt[0]))
        else:
            logZ = np.log(np.sqrt((np.pi * T ** 3) / (Tt[0] * Tt[1] * Tt[2])) / sigma)
        return -logZ * B * T / C

    return translation(3.0, m, T, p) + rotation(T, sigma, Tt) + vibration(T, wavenums)


def vibration(T, wavenums):
    """Vibrational contribution to the free-energy correction (eV)."""
    if len(wavenums) == 0:
        return 0.0
    C = _c.EV_TO_J_MOL / 1000.0
    A = 1.43877735382772
    B = _c.R_GAS * 0.001
    logZ = 0.0
    for w in wavenums:
        if w < 0.01:
            continue
        logZ += (-w * A) / (2 * T) - np.log(1 - np.exp(-w * A / T))
    return -logZ * T * B / C


def shomate_gas(T, shomate_params, frequencies):
    """Gibbs free energy (eV) from Shomate coefficients and ZPE frequencies."""
    _kJmol2eV = 0.01036427
    temperature_ref = 298.15

    def H(T, params):
        A, B, C, D, E, F, G, Hh = params
        t = T / 1000.0
        return A * t + (B / 2.0) * t ** 2 + (C / 3.0) * t ** 3 + \
            (D / 4.0) * t ** 4 - E / t + F - Hh

    def S(T, params):
        A, B, C, D, E, F, G, Hh = params
        t = T / 1000.0
        return A * math.log(t) + B * t + (C / 2.0) * t ** 2 + \
            (D / 3.0) * t ** 3 - E / (2.0 * t ** 2) + G

    def Cp(T, params):
        A, B, C, D, E, F, G, Hh = params
        t = T / 1000.0
        return A + B * t + C * t ** 2 + D * t ** 3 + E / (t ** 2)

    cp_ref = Cp(temperature_ref, shomate_params)
    dH = H(T, shomate_params) - H(temperature_ref, shomate_params)
    dS = S(T, shomate_params)
    dH = (temperature_ref * cp_ref / 1000.0 + dH) * _kJmol2eV
    dS = dS * (_kJmol2eV / 1e3)
    zpe = sum(frequencies) * 5.98132828196056e-3
    # 96.4853240536413 = kJ/mol per eV
    return zpe / (_c.EV_TO_J_MOL / 1000.0) + (dH - T * dS)


def shomate_gas_by_name(T, gas_name, wavenums=None):
    """Look up built-in Shomate parameters by gas name and temperature."""
    if wavenums is None:
        wavenums = _default_wavenumbers(gas_name)
    for key, params in _c.shomate_params.items():
        if key.startswith(gas_name + '_g'):
            _, trange = key.split(':')
            tlow, thigh = (float(v) for v in trange.split('-'))
            if tlow <= T <= thigh:
                return shomate_gas(T, params, wavenums)
    return None


def _default_wavenumbers(gas_name: str) -> List[float]:
    if gas_name + '_g' in _c.experimental_gas_frequencies:
        return list(_c.experimental_gas_frequencies[gas_name + '_g'])
    if gas_name in _c.gas_wave_numbers:
        return list(_c.gas_wave_numbers[gas_name])
    return []


def _as_params_dict(gas) -> Dict:
    if isinstance(gas, dict):
        params = dict(gas)
        # Allow nested custom thermo data.
        for key in ("thermo", "thermo_params"):
            if isinstance(params.get(key), dict):
                params.update(params[key])
        return params
    params = {"name": gas.name}
    if getattr(gas, "pressure", None):
        params["pressure"] = gas.pressure[0]
    params.update(getattr(gas, "thermo_params", {}) or {})
    for key in ("mole_mass", "sticking_coeff", "charge"):
        value = getattr(gas, key, None)
        if value is not None:
            params[key] = value
    return params


def thermo_correction_of_gas(T: float, p: float, gas_name: str,
                             correct_type: Optional[str], params: dict):
    """Free-energy correction (eV) for one gas species.

    ``correct_type`` may be ``"shomate"``, ``"ideal_gas_standard"``,
    ``"ideal_gas_nonstandard"`` or ``None`` (no correction).
    """
    if correct_type is None:
        return 0.0
    params = dict(params or {})
    if "wavenums" in params:
        wavenums = params["wavenums"]
    else:
        wavenums = _default_wavenumbers(gas_name)
        if not wavenums:
            print("Warning: no wavenumbers for", gas_name,
                  "; no thermo correction is done.")
            return 0.0

    if correct_type == "shomate":
        if "shomate_params" in params:
            return shomate_gas(T, params["shomate_params"], wavenums)
        result = shomate_gas_by_name(T, gas_name, wavenums)
        return 0.0 if result is None else result

    if correct_type in ("ideal_gas_standard", "ideal_gas_nonstandard"):
        try:
            mole_mass = params.get("mole_mass", _c.gas_mole_mass[gas_name])
            rotation_T = params.get("rotation_T",
                                    _c.gas_rotation_temperatures[gas_name])
            symm_num = params.get("symm_num",
                                  _c.gas_rotation_symmetry_numbers[gas_name])
        except KeyError:
            print("Warning: cannot find mole_mass, rotation_T or symm_num for",
                  gas_name, "; no thermo correction is done.")
            return 0.0
        if correct_type == "ideal_gas_standard":
            return ideal_gas(101325, T, mole_mass, symm_num, rotation_T, wavenums)
        return ideal_gas(101325 * p, T, mole_mass, symm_num, rotation_T, wavenums)

    print("Warning: unknown thermo correction type", correct_type,
          "; no correction is done.")
    return 0.0


def correct_barriers_with_gas(mkm, single_barrier_data, T: float,
                              correct_type: Optional[str], gas_info):
    """Correct barriers/energies for gas-phase free energies in-place.

    ``single_barrier_data`` is either a ``(nreactions, 2)`` array of
    ``[Ea, reaction_energy]`` or a dictionary mapping species names to
    energies.  ``gas_info`` may contain dictionaries or
    :class:`~catmkm.model.GasSpecies` objects.
    """
    if correct_type is None:
        return single_barrier_data
    if isinstance(single_barrier_data, (list, tuple)):
        single_barrier_data = np.asarray(single_barrier_data, dtype=float)

    gas_correction = {}
    for gas in gas_info:
        params = _as_params_dict(gas)
        name = params["name"]
        pressure = params.get("pressure", [1.0])
        p = pressure[0] if isinstance(pressure, (list, tuple, np.ndarray)) else pressure
        gas_correction[name] = thermo_correction_of_gas(
            T, p, name, correct_type, params)

    if isinstance(single_barrier_data, dict):
        for gas_key, correction in gas_correction.items():
            if gas_key in single_barrier_data:
                single_barrier_data[gas_key] += correction
        return single_barrier_data

    for ir, reaction in enumerate(mkm.reaction_list):
        ini_corr = sum(gas_correction[a.name] * a.num
                       for a in reaction.reactants if a.nsites == 0)
        fin_corr = sum(gas_correction[a.name] * a.num
                       for a in reaction.products if a.nsites == 0)

        single_barrier_data[ir, 1] += fin_corr - ini_corr
        if abs(single_barrier_data[ir, 0]) < 1.0e-10:
            single_barrier_data[ir, 0] = 0
        elif abs(single_barrier_data[ir, 0] - single_barrier_data[ir, 1]) < 1.0e-10:
            single_barrier_data[ir, 0] = single_barrier_data[ir, 1]
        else:
            single_barrier_data[ir, 0] -= ini_corr
    return single_barrier_data
