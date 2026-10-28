"""DC power flow: single-scenario solves + multi-scenario stats.

Origin: ``DCPowerFlowSimple.m`` (loop workhorse) and ``DCPowerFlowFull.m``
(statistics pass) — algorithm doc 04 §3-§4, both REUSE-WITH-EXTENSION over
gridforge ``scale.power_flow.DCPowerFlow`` (cached splu — MATLAB
refactorized per scenario per pass; the factor-once win is the point).

Extensions per doc 04: EXACT-degree angles (Q9 — gridforge reproduces SGG's
arctan-compressed ``angle(1+jθ)`` variant; GridWeld's DCPowerFlowSimple uses
``rad2deg(θ)``, and the difference matters exactly in the
overloaded-corridor regime), single +1/X flow sign (DV-B — MATLAB's
DCPowerFlowFull flows are negated; fixtures compare |flow|), and the
DCStats table formulas.

Sign convention: ``snet_mw`` is NET INJECTION (gen +, load −) in MW, the
PSERCMain convention — internally /s_base and handed to gridforge, whose
B·θ = P solve is convention-agnostic; flows come back with
DCPowerFlowSimple's sign (PLineSend = s_base·BPrim·Aᵀ·θ).

Failure policy: raise ``SingularNetworkError`` on disconnected systems
(DV-H/DV-I — replaces the condest>1e16/sprank proxies and silent scenario
skips). Slack: internal bus 0, unconditionally (quirk Q-d — the DC slack is
NOT the AC stages' argmax-MaxMW slack; two slacks coexist upstream).

Units: injections MW; flows MW; angles degrees (exact).
Phase: P2b. Validated: iteration-1 MaxDCPF envelope vs s7_maxdcpf_iter01.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gridforge.scale.power_flow import DCPowerFlow

from ..model import PowerFlowResult, SingularNetworkError, WeldModel


def _exact_angles(result) -> tuple[np.ndarray, np.ndarray]:
    """Recover θ from gridforge's arctan-compressed degrees, return exact
    degrees (Q9): θ = tan(radians(compressed)) is lossless for the arctan
    image (−90°, 90°)."""
    theta = np.tan(np.radians(result.angles_deg))
    return theta, np.degrees(theta)


def _solver(model: WeldModel, s_base: float) -> DCPowerFlow:
    """gridforge DCPowerFlow over a WeldModel (duck-typed: incidence(),
    n_buses, branches[from_bus/to_bus/r_pu/x_pu/b_pu]). strict_matlab=True:
    disconnection → bail, exactly DCPowerFlowSimple's screen intent."""
    return DCPowerFlow(model, strict_matlab=True, s_base=s_base)


def _to_result(model: WeldModel, res) -> PowerFlowResult:
    if res.singular or not res.converged:
        raise SingularNetworkError(
            "DC power flow: network is disconnected/singular (DV-I exact "
            "connectivity check; legacy returned empty outputs here)")
    _, exact_deg = _exact_angles(res)
    a = model.incidence()
    return PowerFlowResult(
        converged=True,
        v_mag_pu=np.ones(model.n_buses),
        v_angle_deg=exact_deg,
        flow_mva=res.flows_mva,
        angle_diff_deg=np.abs(a.T @ exact_deg),
    )


def solve_dc(model: WeldModel, snet_mw: np.ndarray, *, s_base: float = 100.0,
             contingency: int | None = None) -> PowerFlowResult:
    """One B-θ solve (DCPowerFlowSimple.m): flows in MW, exact-degree
    angle differences. ``snet_mw`` is one scenario column (n_bus,); the
    real part is used (scenario matrices are complex MW+jMVar)."""
    solver = _solver(model, s_base)
    res = solver.solve(np.real(np.asarray(snet_mw)) / s_base,
                       contingency=contingency)
    return _to_result(model, res)


def solve_dc_scenarios(model: WeldModel, snet_mw: np.ndarray, *,
                       s_base: float = 100.0) -> list[PowerFlowResult]:
    """All scenario columns (n_bus × n_scen) with ONE factorization."""
    solver = _solver(model, s_base)
    out = []
    for i in range(snet_mw.shape[1]):
        res = solver.solve(np.real(snet_mw[:, i]) / s_base)
        out.append(_to_result(model, res))
    return out


def dc_envelope(model: WeldModel, snet_mw: np.ndarray, *,
                s_base: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """(MaxDCPF, MaxAngleDiff): elementwise max of |flow| [MW] and |Δθ|
    [deg] across scenarios — the DC-resize loop envelope (PSERCMain §13)."""
    results = solve_dc_scenarios(model, snet_mw, s_base=s_base)
    flows = np.abs(np.column_stack([r.flow_mva for r in results]))
    diffs = np.column_stack([r.angle_diff_deg for r in results])
    return flows.max(axis=1), diffs.max(axis=1)


def dc_stats(model: WeldModel, results: list[PowerFlowResult],
             snet_mw: np.ndarray, *, s_base: float = 100.0) -> pd.DataFrame:
    """DCPowerFlowFull.m per-scenario statistics (doc 04 §4 step 5).

    Losses = s_base·Σ(flow/s_base)²·R; loading metrics over in-service
    lines; failed scenarios RAISE upstream instead of leaving silent zero
    rows (DV-H).
    """
    r_pu = model.branches["r_pu"].to_numpy(dtype=float)
    cap = model.branches["capacity_mva"].to_numpy(dtype=float)
    rows = []
    for i, res in enumerate(results):
        p = np.real(snet_mw[:, i])
        flow = res.flow_mva
        util = np.abs(flow) / np.where(cap > 0, cap, np.inf)
        loaded = util[util < 1]
        over = util[util > 1]
        load_mw = float(np.abs(p[p < 0]).sum())
        losses = float(s_base * np.sum((flow / s_base) ** 2 * r_pu))
        rows.append({
            "gen_mw": float(p[p > 0].sum()),
            "load_mw": load_mw,
            "max_angle_diff_deg": float(res.angle_diff_deg.max()),
            "mean_angle_diff_deg": float(res.angle_diff_deg.mean()),
            "losses_mw": losses,
            "loss_percent": 100.0 * losses / load_mw if load_mw else np.nan,
            "percent_loaded_mean": float(loaded.mean() * 100) if len(loaded) else np.nan,
            "percent_loaded_std": float(loaded.std() * 100) if len(loaded) else np.nan,
            "lines_overloaded": int((util > 1).sum()),
            "percent_lines_overloaded": float((util > 1).mean() * 100),
            "overload_percent_max": float(over.max() * 100) if len(over) else 0.0,
            "overload_percent_mean": float(over.mean() * 100) if len(over) else 0.0,
        })
    return pd.DataFrame(rows)
