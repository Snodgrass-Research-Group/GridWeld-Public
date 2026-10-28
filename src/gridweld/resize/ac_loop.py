"""AC resize loop (MiddleLoopAC).

Origin: ``MiddleLoopAC.m`` — algorithm doc 04 §10. No MATLAB fixture: the
dump's stage 8b failed upstream (cascade of the stage-8 crash). AC envelope
over scenarios, refine while ELIGIBLE overloads (more than one conductor
step of headroom: ``(MaxCondIndex − CondIndex) > 1``) exceed MLTermCriteria
(``ac_term_overloads`` = 5; valve +1 per pass after ``valve_after_iters``),
PercentResize forced to ``ac_percent_resize`` (0.01), scenario column 0
dropped (the duplicate planted by ReactivePowerPlacement).

Envelope semantics (DV-K, D-7 ruling 2026-07-19): the legacy parfor
convention gave non-converged scenarios ZERO flow — on real CATS state
that under-reports utilization, feeds the (random, Downsize=1) downsize
path and degrades AC solvability further (P7 measured 54%→0% over 46
passes; at 0 converged the MATLAB while-loop exits as if SUCCESSFUL).
Default ``resize.ac_dc_substitute`` replaces a non-converged scenario's
contribution with its DC |MW| flows; False restores the faithful
zero-fill.

Exit semantics (same ruling): normal termination returns the current
state; a 0-converged pass ("collapse") or the ``max_passes`` cap returns
the BEST-converged pass's entry state with a warning — never a hard
raise, never the legacy silent success. The log frame's ``attrs`` carry
``{exit, returned_pass, best_converged_fraction}``.

⚠ DV-F: the committed MiddleLoopAC.m targets SGG's 3-arg
TransformerImpedances / struct-arg AssignConductors and references an
undefined ``Sub`` — it CANNOT run against GridWeld's own functions. This
port targets one signature set with all inputs explicit. DV-H: singular
Jacobian solves raise inside ACEngine (the parfor path silently discarded
the flag). DV-J: final stats are computed from the CURRENT solve (legacy
read the previous scenario's SNetCalc — off by one).

Sign/units: ``snet`` arrives as complex net injections MW+jMVar (units are
explicit — the legacy positive-load flip and the ×20000 per-unit sniff,
quirk Q-i, are dropped).

Phase: P3 (property/unit tests); D-7 (DEBUGGING) revised the envelope and
exit semantics after the first real-state runs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import WeldModel
from ..power.ac import ACEngine
from ..power.dc import solve_dc_scenarios

log = logging.getLogger(__name__)


def _ac_envelope(br: pd.DataFrame, n_buses: int, snet: np.ndarray,
                 slack: int, pv: np.ndarray, pq: np.ndarray,
                 vset: np.ndarray, cfg: WeldConfig
                 ) -> tuple[np.ndarray, float, int]:
    """Max |MVA| per branch across scenario solves; converged scenarios
    contribute AC |MVA|, non-converged ones their DC |MW| flows when
    ``resize.ac_dc_substitute`` (DV-K) — zero otherwise (the faithful
    MATLAB ACPFlowsConverge convention). Returns (envelope, converged
    fraction, DC-substituted count). Warm-starts serially from the last
    converged solve; a singular Jacobian raises inside ACEngine (DV-H); a
    disconnected network raises in the DC substitution solve (DV-I)."""
    model = WeldModel(
        buses=pd.DataFrame({"bus_id": np.arange(n_buses, dtype=np.int64)}),
        branches=pd.DataFrame({
            "from_bus": br["From"].to_numpy(np.int64) - 1,
            "to_bus": br["To"].to_numpy(np.int64) - 1,
            "tap": br["Tap"].to_numpy(float),
            "r_pu": br["R"].to_numpy(float),
            "x_pu": br["X"].to_numpy(float),
            "b_pu": br["B"].to_numpy(float),
        }))
    eng = ACEngine(model, s_base=cfg.system.s_base_mva,
                   long_line=cfg.powerflow.long_line,
                   tol=cfg.powerflow.ac_tol,
                   max_iter=cfg.powerflow.ac_max_iter)
    env = np.zeros(len(br))
    v0 = None
    ok = 0
    failed: list[int] = []
    for i in range(snet.shape[1]):
        res = eng.solve(snet[:, i], slack, pv, pq, v0=v0, v_setpoints=vset)
        if res.converged:
            env = np.maximum(env, np.abs(res.flow_mva))
            v0 = res.v_mag_pu * np.exp(1j * np.radians(res.v_angle_deg))
            ok += 1
        else:
            failed.append(i)   # zero contribution unless DC-substituted
    n_dc = 0
    if failed and cfg.resize.ac_dc_substitute:
        for res in solve_dc_scenarios(model, snet[:, failed],
                                      s_base=cfg.system.s_base_mva):
            env = np.maximum(env, np.abs(res.flow_mva))
        n_dc = len(failed)
    return env, ok / snet.shape[1], n_dc


def run_ac_resize(branch_df: pd.DataFrame, snet: np.ndarray, n_buses: int,
                  slack: int, pv: np.ndarray, pq: np.ndarray,
                  vset: np.ndarray, xfdata: pd.DataFrame, xfxr: pd.DataFrame,
                  catalog_crucible: pd.DataFrame, conductor_catalog,
                  gmd_tables, cfg: WeldConfig, rng: np.random.Generator,
                  *, max_passes: int = 100
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MiddleLoopAC.m over a MATLAB-named branch frame. ``snet`` INCLUDES
    the duplicated column 0 (dropped here, as the original does). Returns
    (branch frame, per-pass log frame) — the frame is the terminating
    pass's state, or the best-converged pass's entry state on collapse/cap
    (D-7 ruling; see module docstring)."""
    from gridcrucible.network.conductors import resize_branches

    from ..params.conductors import _CRUCIBLE_RENAME, assign_conductors
    from ..params.impedance import compute_line_impedances
    from ..params.transformers import transformer_impedances
    from .dc_loop import step_circuits

    br = branch_df.copy()
    scen = snet[:, 1:]                        # drop the duplicated column
    term = cfg.resize.ac_term_overloads
    rows = []
    best_br, best_frac, best_pass = br, -1.0, 0
    exit_reason = "cap"
    it = 0
    for it in range(1, max_passes + 1):
        env, frac, n_dc = _ac_envelope(br, n_buses, scen, slack, pv, pq,
                                       vset, cfg)
        br["MaxACPF"] = env
        rate = br["RateA"].to_numpy(float)
        cond = br["CondIndex"].to_numpy(float)
        cmax = br["MaxCondIndex"].to_numpy(float)
        over = (rate - env * cfg.resize.conductor_margin < 0) & (cond < cmax)
        eligible = over & ((cmax - cond) > 1)
        rows.append({"iter": it, "eligible_overloads": int(eligible.sum()),
                     "converged_fraction": frac, "dc_substituted": n_dc,
                     "total_rate_mva": float(rate.sum())})
        log.info("AC resize pass %d: %d eligible overloads (%.0f%% conv, "
                 "%d DC-filled)", it, int(eligible.sum()), 100 * frac, n_dc)
        if frac >= best_frac:             # ties → latest (more work applied)
            best_br, best_frac, best_pass = br.copy(), frac, it
        if frac == 0.0:
            # collapse — checked BEFORE the termination valve: an empty (or
            # DC-only) envelope must never read as convergence
            exit_reason = "collapse"
            break
        if int(eligible.sum()) <= term:
            exit_reason = "converged"
            best_br, best_pass = br, it   # normal exit → the current state
            break

        view = br.rename(columns=_CRUCIBLE_RENAME)
        if "keep" not in view.columns:
            view["keep"] = False
        resize_branches(
            view, env,
            percent_resize=cfg.resize.ac_percent_resize,   # forced 0.01
            percent_change=cfg.resize.percent_change,
            conductor_margin=cfg.resize.conductor_margin,
            downsize_threshold=cfg.resize.downsize_threshold,
            upsize=cfg.resize.upsize, downsize=cfg.resize.downsize,
            downsize_keep=cfg.resize.downsize_keep,
            random_resize=cfg.resize.random_resize, rng=rng)
        step_circuits(view, max_circuits=cfg.resize.max_circuits)
        br = view.rename(columns={v: k for k, v in _CRUCIBLE_RENAME.items()})

        tl = br["Tap"].to_numpy(float) == 0
        tl_part = assign_conductors(
            br.loc[tl], catalog_crucible,
            change_voltages=cfg.resize.ac_change_voltages)
        rxb_in = tl_part.rename(columns={
            "Length": "length_mi", "GMD": "gmd_ft",
            "CondIndex": "conductor_idx", "kVFrom": "voltage_kv_from",
            "Circuits": "n_circuits", "RateA": "capacity_mva", "FID": "fid"})
        rxb_out = compute_line_impedances(rxb_in, conductor_catalog,
                                          gmd_tables, cfg, matlab_compat=True)
        tl_part["R"] = rxb_out["r_pu"].to_numpy()
        tl_part["X"] = rxb_out["x_pu"].to_numpy()
        tl_part["B"] = rxb_out["b_pu"].to_numpy()
        tl_part["RateA"] = rxb_out["capacity_mva"].to_numpy()
        br.loc[tl_part.index, tl_part.columns] = tl_part
        xf_part = transformer_impedances(br.loc[~tl], xfdata, xfxr,
                                         s_base=cfg.system.s_base_mva)
        br.loc[xf_part.index, :] = xf_part

        if it + 1 > cfg.resize.valve_after_iters:
            term += 1
    if exit_reason != "converged":
        log.warning(
            "AC resize %s at pass %d; best converged fraction %.0f%% at "
            "pass %d — returning that pass's entry state (D-7 ruling "
            "2026-07-19); AC-side outputs are NOT credible for this run",
            "collapsed (0 scenarios converged)" if exit_reason == "collapse"
            else f"hit the {max_passes}-pass cap",
            it, 100 * max(best_frac, 0.0), best_pass)
    log_df = pd.DataFrame(rows)
    log_df.attrs.update({"exit": exit_reason, "returned_pass": best_pass,
                         "best_converged_fraction": max(best_frac, 0.0)})
    return best_br, log_df
