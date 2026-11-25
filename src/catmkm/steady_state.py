"""Steady-state solution of the microkinetic rate equations.

The strategy is the one used by the original catrap scripts: integrate the
coverage ODE for a sequence of increasing time spans to obtain a robust (but
only moderately accurate) initial guess, then polish it with Newton iteration
at arbitrary precision.  The ODE is stable because it always relaxes to the
same steady state; Newton is fast and accurate but needs a good initial guess.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field
from typing import Optional

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from .model import MKM, random_disturb, random_guess

_MP = mp.mp


@dataclass
class ODESettings:
    ode_eps: float = 1.0e-13
    ode_nmax: int = 50
    ode_timeout: int = 10
    ode_max_perturb: int = 5


@dataclass
class NewtonSettings:
    newton_eps: str = "1.0e-100"
    newton_nmax: int = 500
    use_ode_if_fail: bool = True


@dataclass
class SteadyStateSettings:
    ode: ODESettings = field(default_factory=ODESettings)
    newton: NewtonSettings = field(default_factory=NewtonSettings)
    nmax_try: int = 20

    @classmethod
    def from_dict(cls, params) -> "SteadyStateSettings":
        """Build settings from the legacy ``PARAMS``-style dictionary."""
        if isinstance(params, SteadyStateSettings):
            return params
        if params is None:
            return cls()
        ode_raw = dict(params.get("ode") or {})
        newton_raw = dict(params.get("newton") or {})
        ode = ODESettings(**{k: v for k, v in ode_raw.items()
                             if k in ODESettings.__dataclass_fields__})
        newton = NewtonSettings(**{k: v for k, v in newton_raw.items()
                                   if k in NewtonSettings.__dataclass_fields__})
        return cls(ode=ode, newton=newton,
                   nmax_try=params.get("nmax_try", 20))


# ---------------------------------------------------------------------------
# Timeout helper (POSIX)
# ---------------------------------------------------------------------------

def _set_timeout(func, num):
    """Wrap *func* so that it raises RuntimeError after *num* seconds."""
    if num is None or num <= 0 or threading.current_thread() is not threading.main_thread():
        return func

    def handle(signum, frame):
        raise RuntimeError("ODE integration timed out")

    def to_do(*args, **kwargs):
        old_handler = signal.signal(signal.SIGALRM, handle)
        signal.alarm(int(num))
        try:
            return func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return to_do


# ---------------------------------------------------------------------------
# ODE and Newton iterations
# ---------------------------------------------------------------------------

def ode_iteration(y0: np.ndarray, Ft, Jt, ode_eps, ode_nmax, timeout=5):
    """Integrate the coverage ODE in growing time spans.

    Returns
    -------
    (success, y, error, iterations)
    """
    finite_time_solve = _set_timeout(solve_ivp, timeout) if timeout > 1 else solve_ivp
    eps = ode_eps
    nmax = ode_nmax
    collect_sol = np.zeros((ode_nmax, len(y0)))
    collect_log_err = np.zeros((ode_nmax,))
    collect_i = 0

    atol = eps * 0.1 if eps > 1.0e-14 else 1.0e-14
    rtol = atol * 1000.0
    up_factor = 5.0
    down_factor = 0.1
    t = 0.001
    i = 1
    y = np.asarray(y0, dtype=float).copy()
    check_y = np.all(y < 1)
    y_old = y.copy()

    while i < nmax:
        solve_ivp_error = False
        try:
            sol = finite_time_solve(Ft, (0, t), y, method='BDF',
                                    t_eval=np.array([t]), jac=Jt,
                                    rtol=rtol, atol=atol)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            print(exc)
            sol = None
            solve_ivp_error = True
        if solve_ivp_error or sol is None or sol.status != 0:
            t *= down_factor
            y = y_old
        else:
            y = np.real(sol.y.flatten())
            if check_y and (np.any(y < 0) or np.any(y > 1)):
                t *= down_factor
                y = y_old
            else:
                err = np.linalg.norm(Ft(0, y))
                if err < eps:
                    return True, y, err, i
                collect_sol[collect_i, :] = y
                collect_log_err[collect_i] = np.log10(err)
                collect_i += 1
                t *= up_factor
                y_old = y.copy()
        i += 1

    if collect_i == 0:
        return False, y, np.linalg.norm(Ft(0, y)), i
    min_index = int(np.argmin(collect_log_err[:collect_i]))
    y = collect_sol[min_index, :]
    return False, y, np.linalg.norm(Ft(0, y)), i


def newton_iteration(x, F, J, newton_eps, newton_nmax, normalize_cvr=None):
    """Damped-free Newton iteration on the (rank-reduced) residual.

    Returns ``(success, x, residual_norm, iterations)``.
    """
    eps = newton_eps
    nmax = newton_nmax
    n = 0
    x_old = x.copy()
    x_old_norm = _MP.norm(F(x_old))
    x_norm = x_old_norm

    while n < nmax:
        try:
            dx, _ = _MP.qr_solve(J(x), F(x))
        except (ValueError, ZeroDivisionError):
            return False, x_old, float(x_old_norm), n
        x_old = x.copy()
        x = x - dx
        x_norm = _MP.norm(F(x))
        if x_norm < eps:
            return True, x, float(x_norm), n
        n += 1
    return False, x, float(x_norm), n


def _rates_from_coverage(mkm: MKM, coverage):
    coverage = _MP.matrix(coverage)
    R = _MP.matrix(mkm.nreactions, 1)
    D = _MP.matrix(mkm.nvar, 1)
    mkm.calc_F(coverage, mkm.rate_constants_mpf, R, D, mkm.gas)
    return R


def single_point_calculation(params, y0: np.ndarray, mkm: MKM,
                             nmax_try: Optional[int] = None,
                             nmax_perturb: Optional[int] = None):
    """Solve one steady-state point.

    Parameters
    ----------
    params:
        A :class:`SteadyStateSettings` or the legacy ``PARAMS`` dictionary
        containing ``"ode"`` and ``"newton"`` sub-dictionaries.
    y0:
        Initial surface coverage vector.
    mkm:
        Model with ``calc_F``/``calc_J`` (call :func:`build_rate_functions`
        first) and rate constants set.

    Returns
    -------
    (coverage, reaction_rates)
        ``coverage`` is an ``mpmath.matrix`` (or a NaN array on complete
        failure) and ``reaction_rates`` an ``mpmath.matrix`` (or NaN array).
    """
    settings = SteadyStateSettings.from_dict(params)
    if nmax_try is None:
        nmax_try = settings.nmax_try
    if nmax_perturb is None:
        nmax_perturb = settings.ode.ode_max_perturb

    ode_eps = settings.ode.ode_eps
    ode_nmax = settings.ode.ode_nmax
    ode_timeout = settings.ode.ode_timeout
    newton_eps = _MP.mpf(settings.newton.newton_eps)
    newton_nmax = settings.newton.newton_nmax
    use_ode_if_fail = settings.newton.use_ode_if_fail

    i_try = 0
    nfail = 0
    coverage = y0
    while i_try < nmax_try:
        success, coverage, err, n = ode_iteration(
            y0, mkm.Ft, mkm.Jt, ode_eps, ode_nmax, ode_timeout)
        if not success:
            print(f"Try {i_try}, ODE FAIL, iter= {n}, error= {err:15.6e}")
        else:
            print(f"Try {i_try}, ODE success, iter= {n}, error= {err:15.6e}")

        success, coverage2, err, n = newton_iteration(
            _MP.matrix(coverage), mkm.F, mkm.J, newton_eps, newton_nmax,
            mkm.normalize_cvr)
        if not success:
            print(f"Try {i_try}, NEWTON fail, iter= {n}, error= {err:15.6e}")
            if use_ode_if_fail:
                print("Return ODE results:")
                coverage = _MP.matrix(coverage)
                return coverage, _rates_from_coverage(mkm, coverage)
            coverage = np.array([float(c) for c in coverage2])
            if nfail < nmax_perturb and not np.isnan(coverage[0]):
                y0 = coverage if mkm.constrain_list else random_disturb(
                    coverage, mkm.normalize_cvr)
            else:
                y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
            nfail += 1
        else:
            print(f"Try {i_try}, Newton success, iter= {n}, error= {err:15.6e}")
            return coverage2, _rates_from_coverage(mkm, coverage2)
        i_try += 1

    return (np.zeros((mkm.nspecies,)) * np.nan,
            np.zeros((mkm.nreactions,)) * np.nan)
