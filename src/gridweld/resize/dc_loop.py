"""DC resize loop: DCPF sweeps alternating with resize passes.

Origin (re-verified vs sources 2026-07-17): PSERCMain.m §"Sequential DC
power flow" (lines 349-408) + ``ResizeBranches.m``. Loop: envelope MaxDCPF
over all SNetFull scenarios → ResizeBranches (upsize utilization >
1/margin, downsize < downsize_threshold, ceil(×/÷ percent_change) steps,
ceil(percent_resize·TotalPaths) cap on each side, randperm subset — DV3) →
circuit stepping for the at-bound flags (UNCONDITIONAL in ResizeBranches.m
lines 44-52: RateA rebased per circuit, ±1, clamp 1..8, RateA restored
post-clamp — :func:`step_circuits`, the gridcrucible port lacks it) →
AssignConductors + RXB on lines, TransformerImpedances on transformers →
terminate when ``nnz(RateA − MaxDCPF·margin < 0) ≤ TermCriteria``
(starts 0; +1 per pass after iteration 20 — the only guaranteed
termination).

Reuse (D5): gridcrucible ``resize_branches`` (sorts INDICES, fixing the
DV-D sortrows desync of the legacy non-random path). Do NOT reuse
TUNELoop.run_dc control flow (it ports SGG's MiddleLoop.m — different
constants: term 5, valve 15). ⚠ gridcrucible defaults differ
(percent_change 1.3, downsize_threshold 0.5) — always passed explicitly
from config. Y-bus refactorizes per pass by construction here (X changes
every pass); within a pass the factorization is shared across scenarios
(the headline win vs MATLAB's per-scenario refactorization).

Units: flows/ratings MW-MVA; utilization fraction.
Phase: P3. Statistical validation vs s7_dcpf_iteration_log (D9b/DV3).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import WeldModel

log = logging.getLogger(__name__)


def step_circuits(branch_view: pd.DataFrame, *, max_circuits: int = 8) -> int:
    """ResizeBranches.m lines 44-52 on the gridcrucible-named view
    (overloaded/undersized/capacity_mva/n_circuits/resize): rebase
    capacity per circuit, Circuits +1 where overloaded / −1 where
    undersized, clamp [1, max], capacity restored with the CLAMPED count.
    Returns rows touched. Mutates in place."""
    over = branch_view["overloaded"].to_numpy(bool)
    under = branch_view["undersized"].to_numpy(bool)
    uoo = over | under
    if not uoo.any():
        return 0
    cap = branch_view["capacity_mva"].to_numpy(float).copy()
    circ = branch_view["n_circuits"].to_numpy(float).copy()
    cap[uoo] = cap[uoo] / circ[uoo]
    circ[over] += 1
    circ[under] -= 1
    circ = np.clip(circ, 1, max_circuits)
    cap[uoo] = cap[uoo] * circ[uoo]
    branch_view["capacity_mva"] = cap
    branch_view["n_circuits"] = circ
    resize = branch_view["resize"].to_numpy(bool)
    branch_view["resize"] = resize | uoo
    return int(uoo.sum())


def resize_reparam_pass(br: pd.DataFrame, max_dcpf: np.ndarray,
                        xfdata: pd.DataFrame, xfxr: pd.DataFrame,
                        catalog_crucible: pd.DataFrame, conductor_catalog,
                        gmd_tables, cfg: WeldConfig,
                        rng: np.random.Generator) -> pd.DataFrame:
    """One resize + reparameterization pass over the MATLAB-named branch
    frame: ResizeBranches → step_circuits → AssignConductors + RXB on lines
    → TransformerImpedances on transformers. This chain is VERBATIM-shared
    between the S4 DC loop (PSERCMain.m:392-395) and the S5 DCOPF loop
    (PSERCMain_v2.m:463-466) — same options (Upsize=Downsize=1,
    changevoltages=0), same order. Expects ``Branch.MaxDCPF`` [MW] already
    set on ``br``; returns the updated frame (input not mutated)."""
    from ..params.conductors import _CRUCIBLE_RENAME, assign_conductors
    from ..params.impedance import compute_line_impedances
    from ..params.transformers import transformer_impedances
    from gridcrucible.network.conductors import resize_branches

    view = br.rename(columns=_CRUCIBLE_RENAME)
    if "keep" not in view.columns:
        view["keep"] = False               # legacy read Keep nowhere (DV-G)
    resize_branches(
        view, max_dcpf,
        percent_resize=cfg.resize.percent_resize,
        percent_change=cfg.resize.percent_change,
        conductor_margin=cfg.resize.conductor_margin,
        downsize_threshold=cfg.resize.downsize_threshold,
        upsize=cfg.resize.upsize, downsize=cfg.resize.downsize,
        downsize_keep=cfg.resize.downsize_keep,
        random_resize=cfg.resize.random_resize, rng=rng)
    step_circuits(view, max_circuits=cfg.resize.max_circuits)
    br = view.rename(columns={v: k for k, v in _CRUCIBLE_RENAME.items()})

    tl = br["Tap"].to_numpy(float) == 0
    tl_part = assign_conductors(br.loc[tl], catalog_crucible,
                                change_voltages=cfg.resize.dc_change_voltages)
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
    return br


def run_dc_resize(branch_df: pd.DataFrame, snet_full: np.ndarray,
                  n_buses: int, xfdata: pd.DataFrame, xfxr: pd.DataFrame,
                  catalog_crucible: pd.DataFrame, conductor_catalog,
                  gmd_tables, cfg: WeldConfig, rng: np.random.Generator,
                  *, max_passes: int = 200
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The PSERCMain DC loop over a MATLAB-named branch frame (post
    SizeTransformers state). Raises SingularNetworkError on disconnection
    (DV-H). Returns (branch frame, iteration log frame with columns
    iter/overloads/total_rate_mva/max_dcpf — the s7 log contract)."""
    from ..power.dc import dc_envelope

    br = branch_df.copy()
    term = cfg.resize.dc_term_overloads
    log_rows = []
    for it in range(1, max_passes + 1):
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
        max_dcpf, max_angle = dc_envelope(model, snet_full,
                                          s_base=cfg.system.s_base_mva)
        br["MaxDCPF"] = max_dcpf
        br["MaxAngleDiff"] = max_angle
        br = resize_reparam_pass(br, max_dcpf, xfdata, xfxr,
                                 catalog_crucible, conductor_catalog,
                                 gmd_tables, cfg, rng)

        overloads = int((br["RateA"].to_numpy(float)
                         - br["MaxDCPF"].to_numpy(float)
                         * cfg.resize.conductor_margin < 0).sum())
        log_rows.append({"iter": it, "overloads": overloads,
                         "total_rate_mva": float(br["RateA"].sum()),
                         "max_dcpf": float(br["MaxDCPF"].max())})
        log.info("DC resize pass %d: %d overloads (term %d)",
                 it, overloads, term)
        if overloads <= term:
            break
        if it + 1 > cfg.resize.valve_after_iters:
            term += 1                          # relaxation valve
    return br, pd.DataFrame(log_rows)
