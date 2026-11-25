"""Continuous stirred-tank reactor (CSTR) simulation.

In the classical steady-state calculation the gas partial pressures are fixed.
A CSTR instead evolves them according to

.. math::

    \\frac{dp_i}{dt} = \\frac{p_i^{feed} - p_i}{\\tau}
                       + s \\, \\sum_j \\nu_{ij} R_j

where ``tau`` is the residence time and ``s`` the ``gas_factor`` that converts
surface reaction rates into partial-pressure rates.  Surface coverages follow
their usual microkinetic ODE.  The network must be built with
``mkm.change_gas = True`` so that the gas pressures are part of the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from .model import MKM, random_guess
from .steady_state import SteadyStateSettings, newton_iteration

_MP = mp.mp


@dataclass
class CSTRResult:
    """Container for a CSTR simulation."""

    time: np.ndarray
    coverage: np.ndarray          # (nsurface, ntimes)
    pressure: np.ndarray          # (ngas, ntimes)
    reaction_rates: np.ndarray    # (nreactions,) at the final state
    steady_coverage: np.ndarray   # (nsurface,)
    steady_pressure: np.ndarray   # (ngas,)
    success: bool
    message: str = ""
    gas_names: list = field(default_factory=list)
    surface_species: list = field(default_factory=list)


def _feed_vector(mkm: MKM, feed: Optional[Dict[str, float]]):
    names = [gas.name for gas in mkm.gas_list]
    if feed is None:
        return np.array([float(g.pressure[0]) for g in mkm.gas_list])
    if not isinstance(feed, dict):
        values = np.asarray(feed, dtype=float).reshape(-1)
        if values.size != len(names):
            raise ValueError("CSTR feed must contain one pressure per gas species.")
        return values
    values = []
    for name in names:
        if name not in feed:
            raise ValueError(f"CSTR feed does not specify a pressure for {name!r}.")
        values.append(float(feed[name]))
    return np.array(values)


def make_cstr_rhs(mkm: MKM, feed, residence_time: float = 1.0,
                  gas_factor: float = 1.0):
    """Return ``(rhs, jac)`` for the coupled surface/gas CSTR ODE.

    This is useful if the caller wants to plug the CSTR into a custom
    integrator.  ``feed`` is either a dict or a sequence of pressures in the
    order of ``mkm.gas_list``.
    """
    if not mkm.change_gas:
        raise ValueError("CSTR simulation requires mkm.change_gas = True and "
                         "build_rate_functions(mkm) to be called afterwards.")
    ns = mkm.nspecies
    ng = len(mkm.gas_list)
    feed_vec = _feed_vector(mkm, feed)
    if len(feed_vec) != ng:
        raise ValueError("feed must contain one pressure per gas species.")

    def rhs(t, y):
        d = np.asarray(mkm.Ft(t, y), dtype=float)
        d[ns:] = gas_factor * d[ns:] + (feed_vec - y[ns:]) / residence_time
        return d

    def jac(t, y):
        J = np.array(mkm.Jt(t, y), dtype=float, copy=True)
        J[ns:, :ns] *= gas_factor
        J[ns:, ns:] = gas_factor * J[ns:, ns:]
        J[ns:, ns:] -= np.eye(ng) / residence_time
        return J

    return rhs, jac


def _mp_residuals(mkm: MKM, feed_vec, residence_time, gas_factor):
    ns = mkm.nspecies
    ng = len(mkm.gas_list)

    def F(y):
        R = _MP.matrix(mkm.nreactions, 1)
        D = _MP.matrix(ns + ng, 1)
        mkm.calc_F(y, mkm.rate_constants_mpf, R, D, None)
        for i in range(ng):
            D[ns + i] = (gas_factor * D[ns + i]
                         + (_MP.mpf(feed_vec[i]) - y[ns + i])
                         / _MP.mpf(residence_time))
        return D

    def J(y):
        R = _MP.matrix(mkm.nreactions, ns + ng)
        D = _MP.matrix(ns + ng, ns + ng)
        mkm.calc_J(y, mkm.rate_constants_mpf, R, D, None)
        for i in range(ng):
            for j in range(ns):
                D[ns + i, j] = gas_factor * D[ns + i, j]
            for j in range(ng):
                D[ns + i, ns + j] = gas_factor * D[ns + i, ns + j]
            D[ns + i, ns + i] = D[ns + i, ns + i] - _MP.mpf('1') / _MP.mpf(residence_time)
        return D

    return F, J


def simulate_cstr(
    mkm: MKM,
    feed: Optional[Dict[str, float]] = None,
    residence_time: float = 1.0,
    gas_factor: float = 1.0,
    y0: Optional[np.ndarray] = None,
    p0: Optional[np.ndarray] = None,
    max_time: Optional[float] = None,
    surface_tol: float = 1.0e-10,
    gas_tol: float = 1.0e-10,
    refine: bool = True,
    settings=None,
) -> CSTRResult:
    """Integrate a CSTR to steady state.

    Parameters
    ----------
    mkm:
        Model with ``change_gas = True`` and generated rate functions.
    feed:
        Feed partial pressures as ``{gas_name: pressure}``.  Defaults to the
        pressures stored in ``mkm.gas_list``.
    residence_time:
        CSTR residence time ``tau``.
    gas_factor:
        Conversion factor between surface reaction rates and pressure rates.
    y0, p0:
        Initial surface coverage / gas pressures.  Random coverage and the
        feed pressure are used by default.
    max_time:
        Maximum integration time (defaults to ``1e4 * residence_time``).
    refine:
        Polish the integrated steady state with Newton iteration.
    """
    if not mkm.change_gas:
        raise ValueError("simulate_cstr requires mkm.change_gas = True and "
                         "build_rate_functions(mkm) to be called afterwards.")
    ns = mkm.nspecies
    ng = len(mkm.gas_list)
    if mkm.rate_constants is None or mkm.rate_constants_mpf is None:
        raise ValueError("rate constants are not set on the model.")
    if y0 is None:
        y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
    y0 = np.asarray(y0, dtype=float).reshape(ns)
    feed_vec = _feed_vector(mkm, feed)
    if p0 is None:
        p0 = feed_vec.copy()
    elif isinstance(p0, dict):
        p0 = _feed_vector(mkm, p0)
    p0 = np.asarray(p0, dtype=float).reshape(ng)
    if max_time is None:
        max_time = 1.0e4 * residence_time

    rhs, jac = make_cstr_rhs(mkm, feed_vec, residence_time, gas_factor)
    y = np.concatenate([y0, p0])

    times = [0.0]
    states = [y.copy()]
    dt = max(min(1.0e-3 * residence_time, 1.0e-3), 1.0e-12)
    t_cur = 0.0
    success = False
    message = "maximum time reached"
    rtol, atol = 1.0e-9, 1.0e-12

    while t_cur < max_time:
        sol = solve_ivp(rhs, (t_cur, t_cur + dt), y, method='BDF',
                        jac=jac, rtol=rtol, atol=atol)
        if not sol.success:
            message = sol.message
            break
        y = sol.y[:, -1]
        t_cur += dt
        times.append(t_cur)
        states.append(y.copy())

        resid = rhs(t_cur, y)
        surface_ok = np.linalg.norm(resid[:ns], ord=np.inf) < surface_tol
        gas_ok = np.linalg.norm(resid[ns:], ord=np.inf) < gas_tol
        if surface_ok and gas_ok:
            success = True
            message = "converged"
            break
        # Exponential time stepping, but do not overshoot max_time.
        dt = min(dt * 2.0, max_time - t_cur)

    y = states[-1].copy()
    # Optional Newton polish.  It also gives a clear success flag.
    if refine:
        try:
            F, J = _mp_residuals(mkm, feed_vec, residence_time, gas_factor)
            cfg = SteadyStateSettings.from_dict(settings)
            ok, y_mp, err, _ = newton_iteration(
                _MP.matrix(y), F, J, _MP.mpf(cfg.newton.newton_eps),
                cfg.newton.newton_nmax)
            if ok:
                y = np.array([float(v) for v in y_mp])
                success = True
                message = "converged (Newton)"
            elif success:
                message = "ODE converged; Newton did not improve the solution"
        except Exception as exc:  # pragma: no cover - defensive
            if success:
                message = f"ODE converged; Newton failed: {exc}"

    states_arr = np.column_stack(states)
    coverage = states_arr[:ns, :]
    pressure = states_arr[ns:, :]

    # Reaction rates at the final state.
    R = np.zeros(mkm.nreactions)
    D = np.zeros(ns + ng)
    mkm.calc_F(y, mkm.rate_constants, R, D, None)

    return CSTRResult(
        time=np.asarray(times),
        coverage=coverage,
        pressure=pressure,
        reaction_rates=R,
        steady_coverage=y[:ns].copy(),
        steady_pressure=y[ns:].copy(),
        success=success,
        message=message,
        gas_names=[g.name for g in mkm.gas_list],
        surface_species=list(mkm.surface_species),
    )
