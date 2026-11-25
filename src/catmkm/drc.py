"""Degree-of-rate-control (DRC) analysis."""

from __future__ import annotations

import re

import mpmath as mp
import numpy as np

from .constants import EV_TO_J_MOL, R_GAS
from .model import MKM, get_rate_constants
from .steady_state import single_point_calculation

_MP = mp.mp


def DRC(coverage0, rate0, delta_e: float, T: float, params, mkm: MKM,
        single_barrier_data):
    """Compute the degree of rate control for every species/transition state.

    Parameters
    ----------
    coverage0, rate0:
        Steady-state coverage and reaction rates (``mpmath.matrix``).
    delta_e:
        Finite perturbation of the energy (eV).  A non-positive value disables
        the analysis.
    params:
        Steady-state settings (or legacy ``PARAMS`` dict).
    single_barrier_data:
        Either a dictionary of species energies or an ``(nreactions, 2)``
        array of ``[Ea, reaction_energy]``.
    """
    if delta_e <= 0:
        return None, None
    if not (isinstance(coverage0, _MP.matrix) and isinstance(rate0, _MP.matrix)):
        raise RuntimeError("coverage0 and rate0 must be mpmath matrices.")

    np_coverage0 = np.array([float(c) for c in coverage0])
    factor = _MP.mpf('1.0') / (
        _MP.exp(-_MP.mpf(str(EV_TO_J_MOL)) * _MP.mpf(delta_e)
                / _MP.mpf(str(R_GAS)) / _MP.mpf(T)) - _MP.mpf('1.0'))

    def control_from_rates(rates):
        drc = _MP.matrix(mkm.nreactions, 1)
        for ir in range(mkm.nreactions):
            drc[ir] = (rates[ir] - rate0[ir]) / rates[ir] * factor
        return drc

    collect_drc = []
    collect_header = []

    if isinstance(single_barrier_data, dict):
        print("The energies of species are given. "
              "Calculate DRC of every intermediate species.")
        for key in single_barrier_data.keys():
            if re.match(r'^[#\d]+$', key) is None or '#' not in key:
                continue
            print("Species:", key)
            collect_header.append(key)

            data = dict(single_barrier_data)
            data[key] += delta_e
            mkm.rate_constants = get_rate_constants(mkm, data, T, 0)
            mkm.rate_constants_mpf = get_rate_constants(
                mkm, data, T, mkm.data_ndigit)
            _, rates = single_point_calculation(params, np_coverage0, mkm)
            collect_drc.append(control_from_rates(rates))
    else:
        print("Only forward barriers and reaction energies are given. "
              "Calculate the DRC of every transition state.")
        for ir in range(mkm.nreactions):
            print("Reaction: ", mkm.reaction_list[ir])
            collect_header.append(str(mkm.reaction_list[ir]))

            data = np.array(single_barrier_data, dtype=float, copy=True)
            data[ir, 0] += delta_e
            mkm.rate_constants = get_rate_constants(mkm, data, T, 0)
            mkm.rate_constants_mpf = get_rate_constants(
                mkm, data, T, mkm.data_ndigit)
            _, rates = single_point_calculation(params, np_coverage0, mkm)
            collect_drc.append(control_from_rates(rates))
    return collect_header, collect_drc
