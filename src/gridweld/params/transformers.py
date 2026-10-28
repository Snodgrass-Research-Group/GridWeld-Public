"""Transformer impedances + duty-based sizing (S3, P3b).

Origin (re-verified against the MATLAB sources 2026-07-17):
``TransformerImpedances.m`` and ``SizeTransformers.m``. X/R from the
empirical XFData (Pri/Sec kV → XHL % on own rating) and XFXR (MVA → X/R)
tables (MATLAB ``table`` objects; B1-resolved parquet in gridweld/data/).

⚠ DV-P3-9 REVISED (fixture evidence, s6 dump): the ``kVFrom == kVTo``
"stale ZIndex" path is NOT dead code — **353 of CATS's 680 transformers are
138/138 after the Pri/Sec snap** and carry REAL impedances computed from
whatever (Pri,Sec) pair the PREVIOUS successfully-resolved row left behind
(implied XHL ∈ {10.9, 14.6, 17.3} = the XHLs of the 138/230, 230/500,
138/500 pairs — confirmed against the fixture). ``matlab_compat=True``
(default) reproduces the stale-carry exactly (a same-kV row with no prior
resolved pair is SKIPPED, keeping its placeholder X — the legacy first-row
try/catch); ``False`` raises on same-kV rows.

TransformerImpedances.m semantics, in order (rows = the tap≠0 subset):
1. ``RateA < MinCondIndex`` → clamp up, flag Undersized (ALL rows).
2. For Resize-flagged rows: snap kVFrom to unique(Pri) and kVTo to
   unique(Sec) (interp1 nearest/extrap ≡ nearest-value snap); X/R from
   XFXR — 25 below 50 MVA, per-unit interp above 2000 MVA with
   ``Circuits = ceil(RateA/2000)``; (Pri,Sec) pair looked up direct then
   swapped; ``Zpu = (100/RateA)·XHL/100``; ``X = √(Zpu²/(1+1/XR²))``;
   ``R = X/XR``. Any error → row silently skipped (legacy try/catch).
3. ALL rows: ``CondIndex = ceil(RateA/Circuits)`` (⚠ field overload:
   per-unit MVA of one parallel transformer, not a catalog index);
   Resize cleared. (The parallel-count adjustment FIXME was never
   implemented upstream — not ported.)

Reuse note (D5): gridcrucible ``transformer_impedances`` ports steps 2-3
but raises on missing pairs and lacks the clamp + stale-carry — the CATS
fixtures REQUIRE the legacy semantics, so this is a faithful WRITE-NEW with
the gridcrucible port cited for cross-checking.

Units: MVA ratings (RateA = TOTAL across parallel units); kV; Zpu/X/R
per-unit on 100 MVA (``system.s_base_mva``).
Phase: P3. Validated vs s6_branch_post_sizetransformers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import SingularNetworkError, WeldModel


def transformer_impedances(branch_df: pd.DataFrame, xfdata: pd.DataFrame,
                           xfxr: pd.DataFrame, *, s_base: float = 100.0,
                           matlab_compat: bool = True) -> pd.DataFrame:
    """TransformerImpedances.m over a MATLAB-named frame (tap≠0 rows only —
    a mixed frame is subset like the original). Returns the updated copy of
    the SUBSET (caller reassigns rows, mirroring ``Branch(XF,:) = ...``)."""
    xf_mask = branch_df["Tap"].to_numpy(float) != 0
    br = branch_df.loc[xf_mask].copy() if not xf_mask.all() else branch_df.copy()

    # step 1 — MinCondIndex clamp (all rows, not just Resize)
    rate = br["RateA"].to_numpy(float)
    floor = br["MinCondIndex"].to_numpy(float)
    low = rate < floor
    rate[low] = floor[low]
    br["RateA"] = rate
    br.loc[low, "Undersized"] = 1.0

    pri = np.sort(xfdata["Pri"].unique().astype(float))
    sec = np.sort(xfdata["Sec"].unique().astype(float))
    mva_grid = xfxr["MVA"].to_numpy(float)
    xr_grid = xfxr["XR"].to_numpy(float)

    def _snap(val: float, grid: np.ndarray) -> float:
        # interp1(...,'nearest','extrap'): midpoint ties take the HIGHER
        # sample; no such ties occur on the CATS grids
        return float(grid[np.abs(grid - val).argmin()])

    stale_xhl: float | None = None            # the carried ZIndex → XHL value
    resize_rows = br.index[br["Resize"].to_numpy(float) != 0]
    for r in resize_rows:
        kv_f = _snap(float(br.at[r, "kVFrom"]), pri)
        kv_t = _snap(float(br.at[r, "kVTo"]), sec)
        br.at[r, "kVFrom"] = kv_f
        br.at[r, "kVTo"] = kv_t
        rate_r = float(br.at[r, "RateA"])
        if rate_r < 50:
            xr = 25.0
        elif rate_r > 2000:
            circ = math.ceil(rate_r / 2000)
            br.at[r, "Circuits"] = float(circ)
            xr = float(np.interp(rate_r / circ, mva_grid, xr_grid))
        else:
            xr = float(np.interp(rate_r, mva_grid, xr_grid))

        if kv_f == kv_t:
            # DV-P3-9: stale-carry (load-bearing for 353/680 CATS rows)
            if not matlab_compat:
                raise ValueError(f"same-kV transformer at row {r} "
                                 f"({kv_f}/{kv_t}) — matlab_compat=False")
            xhl = stale_xhl
        else:
            pair = xfdata[(xfdata["Pri"] == kv_f) & (xfdata["Sec"] == kv_t)]
            if pair.empty:
                pair = xfdata[(xfdata["Sec"] == kv_f) & (xfdata["Pri"] == kv_t)]
            xhl = float(pair["XHL"].iloc[0]) if len(pair) else stale_xhl
            if len(pair):
                stale_xhl = xhl
        if xhl is None:
            continue                          # legacy: error under try/catch
        z_pu = (s_base / rate_r) * (xhl / 100.0)
        x = math.sqrt(z_pu ** 2 / (1.0 + 1.0 / xr ** 2))
        br.at[r, "X"] = x
        br.at[r, "R"] = x / xr

    br["CondIndex"] = np.ceil(br["RateA"].to_numpy(float)
                              / br["Circuits"].to_numpy(float))
    br["Resize"] = 0.0
    return br


def size_transformers(branch_df: pd.DataFrame, snet_complete: np.ndarray,
                      n_buses: int, xfdata: pd.DataFrame, xfxr: pd.DataFrame,
                      catalog_matlab: pd.DataFrame,
                      catalog_crucible: pd.DataFrame, cfg: WeldConfig,
                      rng: np.random.Generator, gmd_tables,
                      conductor_catalog) -> pd.DataFrame:
    """SizeTransformers.m: give the placeholder bus-tie transformers real
    ratings from DC duty across the deduplicated scenario tier.

    Steps (order kept exactly, incl. the impedance/rating inconsistency of
    step 2 running before ratings settle — doc 03 quirk):
    1. XF rows: RateA = cfg.params.xf_initial_mva (2000);
       MaxCondIndex = xf_max_index_factor × max(XFXR.MVA); MinCondIndex =
       xf_min_mva (20); TransformerImpedances.
    2. DCPF sweep over ``snet_complete`` columns (net-injection MW; /s_base
       inside) → MaxDCPF envelope; disconnected → SingularNetworkError.
    3. RateA(XF) = MaxDCPF(XF); Resize(XF) = 1; TransformerImpedances.
    4. Second sweep with the new X → ResizeBranches (Upsize=Downsize=1,
       stochastic — DV3 rng) → TransformerImpedances(XF) +
       AssignConductors(TL) + RXB(TL).
    5. The low-flow debug tail (MaxDCPF < 0.1 MW) is logged, not exported.

    Returns the updated MATLAB-named frame (all rows).
    """
    from gridcrucible.network.conductors import resize_branches

    from ..params.conductors import assign_conductors
    from ..params.impedance import compute_line_impedances
    from ..power.dc import dc_envelope

    br = branch_df.copy()
    xf = br["Tap"].to_numpy(float) != 0
    br.loc[xf, "RateA"] = cfg.params.xf_initial_mva
    br.loc[xf, "MaxCondIndex"] = cfg.params.xf_max_index_factor * \
        xfxr["MVA"].max()
    br.loc[xf, "MinCondIndex"] = cfg.params.xf_min_mva
    br.loc[xf, "Resize"] = 1.0
    sub = transformer_impedances(br.loc[xf], xfdata, xfxr,
                                 s_base=cfg.system.s_base_mva)
    br.loc[sub.index, :] = sub

    def _sweep() -> np.ndarray:
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
        try:
            max_dcpf, _ = dc_envelope(model, snet_complete,
                                      s_base=cfg.system.s_base_mva)
        except SingularNetworkError as exc:
            raise SingularNetworkError(
                "Network has become disconnected. Check code and try again"
            ) from exc
        return max_dcpf

    br["MaxDCPF"] = _sweep()
    br.loc[xf, "RateA"] = br.loc[xf, "MaxDCPF"].to_numpy()
    br.loc[xf, "Resize"] = 1.0
    sub = transformer_impedances(br.loc[xf], xfdata, xfxr,
                                 s_base=cfg.system.s_base_mva)
    br.loc[sub.index, :] = sub

    max_dcpf = _sweep()

    from ..params.conductors import _CRUCIBLE_RENAME
    view = br.rename(columns=_CRUCIBLE_RENAME)
    if "keep" not in view.columns:
        view["keep"] = False                 # legacy read Keep nowhere (DV-G)
    resize_branches(
        view, max_dcpf,
        percent_resize=cfg.resize.percent_resize,
        percent_change=cfg.resize.percent_change,
        conductor_margin=cfg.resize.conductor_margin,
        downsize_threshold=cfg.resize.downsize_threshold,
        upsize=True, downsize=True,
        downsize_keep=cfg.resize.downsize_keep,
        random_resize=cfg.resize.random_resize, rng=rng)
    from ..resize.dc_loop import step_circuits
    step_circuits(view, max_circuits=cfg.resize.max_circuits)
    br = view.rename(columns={v: k for k, v in _CRUCIBLE_RENAME.items()})

    xf = br["Tap"].to_numpy(float) != 0
    sub = transformer_impedances(br.loc[xf], xfdata, xfxr,
                                 s_base=cfg.system.s_base_mva)
    br.loc[sub.index, :] = sub
    tl = ~xf
    tl_part = assign_conductors(br.loc[tl], catalog_crucible,
                                change_voltages=False)
    rxb_in = tl_part.rename(columns={
        "Length": "length_mi", "GMD": "gmd_ft", "CondIndex": "conductor_idx",
        "kVFrom": "voltage_kv_from", "Circuits": "n_circuits",
        "RateA": "capacity_mva", "FID": "fid"})
    rxb_out = compute_line_impedances(rxb_in, conductor_catalog, gmd_tables,
                                      cfg, matlab_compat=True)
    tl_part["R"] = rxb_out["r_pu"].to_numpy()
    tl_part["X"] = rxb_out["x_pu"].to_numpy()
    tl_part["B"] = rxb_out["b_pu"].to_numpy()
    tl_part["RateA"] = rxb_out["capacity_mva"].to_numpy()
    br.loc[tl_part.index, tl_part.columns] = tl_part

    low = br["MaxDCPF"].to_numpy(float) < 0.1
    if low.any():
        import logging
        logging.getLogger(__name__).info(
            "SizeTransformers QA: %d branches with MaxDCPF < 0.1 MW "
            "(%d transformers)", int(low.sum()), int((low & xf).sum()))
    return br
