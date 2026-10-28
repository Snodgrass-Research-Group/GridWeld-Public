"""Per-unit line impedances from conductor + tower geometry (RXB.m port).

Origin: ``legacy/Jonathan_Code/Matlab Functions/RXB.m`` (algorithm doc 03
§RXB — re-verified against the source 2026-07-16):

1. Per kV class, conductor unit parameters via pooled ``CondIndex``:
   ``r = Ra`` [Ω/mi]; ``x = Xa + Xd(GMD)`` [Ω/mi];
   ``y = j/((XaP + XdPrime(GMD))·1e6)`` [S/mi] (XaP in MΩ·mi); Xd/XdPrime
   indexed DIRECTLY by GMD in integer feet (1-based; UG lines' GMD = 1 ft →
   both spacing factors 0). The inverse-CDF GMDTable draw is commented out —
   GridWeld GMD always comes from TowerData.
2. Long-line correction: ``Z = z·L·sinh(γL)/(γL)`` [Ω],
   ``B = Im(y·L·tanh(γL/2)/(γL/2))`` [S]; L in statute miles.
3. Per-unit on ``Zbase = kVFrom²/100`` (literal 100 ≡ ``system.s_base_mva``):
   R,X ÷ Zbase; B × Zbase.
4. Circuits: R,X ÷ n_circuits; ⚠ **B ÷ n_circuits in matlab_compat mode**
   (DV-P3-2 — parallel circuits physically MULTIPLY shunt charging;
   ``matlab_compat=False`` applies the fix).
5. Stability re-cap: ``capacity_mva = min(capacity_mva,
   s_base·sin(stability_angle_deg)/X)``.

Reuse (D5, doc 03 verdict REUSE-WITH-EXTENSION): steps 1–3 are
``gridforge.network.impedance.compute_rxb`` verbatim (incl. the small-γL
guard — legacy relied on 0/0=NaN plus later row deletion for zero-length
lines; zero lengths never occur post-Form 1 because PSERCMain floors them at
``min_length_mi``). Extensions: real GIS lengths in MILES (gridforge's
``length_km/1.61`` is bypassed exactly by pre-multiplying with its own
constant), TowerData ``gmd_ft`` mandatory (zero/NaN raises up front, listing
offenders), CATS catalog via ``gridweld.params.conductors.load_cats_catalog``,
and steps 4–5 here.

Units: lengths mi; GMD ft; Ra/Xa Ω/mi; XaP MΩ·mi; R/X/B per-unit on
``s_base_mva``; ratings MVA (total across parallel circuits).

Phase: P2b. Validated vs ``s2_branch_post_rxb.csv`` at rtol 1e-9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gridforge.network.conductors import ConductorCatalog
from gridforge.network.impedance import MILES_TO_KM, GMDTables, compute_rxb

from ..config import WeldConfig

#: Committed CAISO Xd/XdPrime npz (P2a artifact from GMDDataCAISO.m; the
#: gmd_table plane is a zeros placeholder — never drawn from).
DEFAULT_GMD_NPZ = Path(__file__).resolve().parents[1] / "data" / "gmd_tables.npz"


def load_gmd_tables(path: str | Path = DEFAULT_GMD_NPZ) -> GMDTables:
    """gridforge ``GMDTables`` over the CAISO npz (P2a artifact)."""
    return GMDTables(path)


def compute_line_impedances(branch_df: pd.DataFrame, catalog: ConductorCatalog,
                            gmd_tables: GMDTables, cfg: WeldConfig, *,
                            matlab_compat: bool = True) -> pd.DataFrame:
    """RXB.m for a LINE table (transformer impedances live in
    ``gridweld.params.transformers``). Returns a copy with ``r_pu``/
    ``x_pu``/``b_pu`` recomputed and the stability cap applied to
    ``capacity_mva``.

    Required columns: ``length_mi``, ``gmd_ft`` (integer feet ≥ 1),
    ``conductor_idx`` (1-based pooled), ``voltage_kv_from`` (∈ catalog kV
    set), ``n_circuits``, ``capacity_mva``.
    """
    out = branch_df.copy()
    gmd_ft = out["gmd_ft"].to_numpy(dtype=float)
    bad = ~np.isfinite(gmd_ft) | (gmd_ft < 1)
    if bad.any():
        fids = (out.loc[bad, "fid"].tolist()[:20] if "fid" in out
                else np.flatnonzero(bad)[:20].tolist())
        raise ValueError("RXB: lines with missing/zero GMD (no TowerData "
                         f"match upstream): {fids}")
    idx = out["conductor_idx"].to_numpy(dtype=np.int64)
    kv = out["voltage_kv_from"].to_numpy(dtype=np.int64)
    length_mi = out["length_mi"].to_numpy(dtype=float)

    # Zbase check: RXB.m uses kVFrom²; compute_rxb uses the pooled row's kv —
    # identical by construction (CondIndex chosen inside the kV block).
    pooled_kv = catalog.pooled["kv"].to_numpy()[idx - 1].astype(np.int64)
    if not np.array_equal(pooled_kv, kv):
        off = np.flatnonzero(pooled_kv != kv)
        raise ValueError(f"RXB: conductor_idx outside its kV block at rows "
                         f"{off.tolist()[:20]} (Zbase would disagree)")

    # gridforge divides by its own MILES_TO_KM — pre-multiplying by the same
    # constant hands it the true statute miles (≤1 ulp noise, doc 03 §RXB).
    r_pu, x_pu, b_pu, _, _ = compute_rxb(
        length_mi * MILES_TO_KM, np.zeros(len(out), dtype=np.int64), gmd_ft,
        idx, kv, catalog, gmd_tables, per_voltage=False,
        s_base=cfg.system.s_base_mva)

    circuits = out["n_circuits"].to_numpy(dtype=float)
    r_pu = r_pu / circuits
    x_pu = x_pu / circuits
    b_pu = b_pu / circuits if matlab_compat else b_pu * circuits   # DV-P3-2

    out["r_pu"] = r_pu
    out["x_pu"] = x_pu
    out["b_pu"] = b_pu
    stability = (cfg.system.s_base_mva
                 * np.sin(np.deg2rad(cfg.params.stability_angle_deg)) / x_pu)
    out["capacity_mva"] = np.minimum(
        out["capacity_mva"].to_numpy(dtype=float), stability)
    return out
