"""Generator cost-curve assignment.

Origin: ``generate_grid_script.jl`` lines 71-205 (algorithm doc 02 §costs).
Coefficient bookkeeping: the CSV's Final A/B/C are ASCENDING (A = constant)
→ renamed c0/c1/c2; the MATPOWER cost vector is DESCENDING [c2, c1, c0].
Here costs live in separate c2/c1/c0 columns (the julia fixture dumps them
as cost_c2/cost_c1/cost_c0).

Pass structure — order is load-bearing and replicated exactly:

1. ONE sequential loop over generators (script lines 151-182): renewables →
   [0,0,0]; Nuclear → 7.027 $/MWh linear; Conventional Steam Coal → FIRST
   coal row of the cost CSV; then, if still NaN, the nearest-capacity
   fallback scan over ALL generators of the same FuelType. ⚠ The scan reads
   each donor's CURRENT cost state: donors earlier in the table may already
   carry special-case/fallback-assigned costs, later renewables are still
   NaN — sequential processing is semantic, not an optimization miss.
2. A SECOND full loop (lines 185-205) for the donor-type map
   (Other Natural Gas ← Natural Gas Steam Turbine, Municipal Solid Waste ←
   Landfill Gas) with the same comparator — it runs even for generators
   pass 1 already filled (overwrites them).

⚠ DV-P2c-1 comparator drift bug (replicated when
``cfg.costs.legacy_nearest_capacity`` is True, the default): the accept
condition compares |pmax_i − closest| > |pmax_j − closest| where
``closest`` is the LAST ACCEPTED donor's capacity, not pmax_i — the chosen
donor is scan-order dependent and not the true nearest. False = true
argmin |pmax_i − pmax_j| over join-costed donors.

Units: c2 $/MWh², c1 $/MWh, c0 $.
Phase: P2c. Validated vs p1_generators_clean (exact in compat mode).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WeldConfig


def _legacy_scan(pmax: np.ndarray, c2: np.ndarray, c1: np.ndarray,
                 c0: np.ndarray, i: int, donor_mask: np.ndarray) -> bool:
    """The verbatim comparator loop: scan donors j in row order; accept when
    closest is unset OR |pmax_i − closest| > |pmax_j − closest|; on accept,
    closest ← pmax_j and cost_i ← cost_j (current state). Returns True if
    any donor accepted."""
    closest = None
    hit = False
    for j in np.flatnonzero(donor_mask):
        if closest is None or abs(pmax[i] - closest) > abs(pmax[j] - closest):
            closest = pmax[j]
            c2[i], c1[i], c0[i] = c2[j], c1[j], c0[j]
            hit = True
    return hit


def assign_costs(gen_df: pd.DataFrame, cost_csv: str | Path,
                 cfg: WeldConfig) -> pd.DataFrame:
    """Attach model-2 3-coefficient polynomial costs to every generator.

    Adds columns model(=2), ncost(=3), startup, shutdown, c2, c1, c0.
    ``types_without_matches`` (script lines 136-149) is computed-but-unused
    upstream — not ported (dead code, doc 02 step 7).
    """
    cost = pd.read_csv(cost_csv, low_memory=False)
    cost = cost[cost["Plant Code"].notna()].rename(
        columns={"Plant Code": "PlantCode", "Generator ID": "GenID",
                 "Final A": "c0", "Final B": "c1", "Final C": "c2"})

    merged = gen_df.merge(
        cost[["PlantCode", "GenID", "c0", "c1", "c2", "Technology"]],
        on=["PlantCode", "GenID"], how="left", sort=False,
        suffixes=("", "_cost"), indicator=True)
    if len(merged) != len(gen_df):
        raise ValueError("cost-curve join fanned out: duplicate "
                         "(PlantCode, GenID) rows in the cost CSV")
    # DataFrames.jl leftjoin row layout (verified against the p1 fixture:
    # one descent at row 754): key-MATCHED rows first in left order, then
    # unmatched in left order. The sequential cost scans below are
    # order-sensitive (DV-P2c-1), so this layout is semantic.
    matched = merged["_merge"] == "both"
    merged = pd.concat([merged[matched], merged[~matched]],
                       ignore_index=True).drop(columns="_merge")
    out = merged.drop(columns=["Technology"])
    out["model"] = 2
    out["ncost"] = 3
    out["startup"] = 0.0
    out["shutdown"] = 0.0

    # join state: [c2, c1, c0] if c0 present, else all-NaN (script line 124)
    have = out["c0"].notna().to_numpy()
    c2 = np.where(have, out["c2"].to_numpy(dtype=float), np.nan)
    c1 = np.where(have, out["c1"].to_numpy(dtype=float), np.nan)
    c0 = np.where(have, out["c0"].to_numpy(dtype=float), np.nan)

    fuel = out["FuelType"].to_numpy(dtype=object)
    pmax = out["pmax"].to_numpy(dtype=float)
    renewables = set(cfg.costs.renewables)
    coal = cost[cost["Technology"] == "Conventional Steam Coal"]

    join_c2 = c2.copy()          # join-state snapshot (fixed-mode donor pool)

    # --- pass 1: special cases + same-fuel fallback (sequential) ---
    for i in range(len(out)):
        if fuel[i] in renewables:
            c2[i] = c1[i] = c0[i] = 0.0
        elif fuel[i] == "Nuclear":
            c2[i], c0[i] = 0.0, 0.0
            c1[i] = cfg.costs.nuclear_linear_usd_per_mwh
        elif fuel[i] == "Conventional Steam Coal":
            c0[i] = coal["c0"].iloc[0]
            c1[i] = coal["c1"].iloc[0]
            c2[i] = coal["c2"].iloc[0]
        if np.isnan(c2[i]):
            donor = (fuel == fuel[i]) & ~np.isnan(c2)
            donor[i] = False
            if cfg.costs.legacy_nearest_capacity:
                _legacy_scan(pmax, c2, c1, c0, i, donor)
            elif donor.any():
                j = np.flatnonzero(donor)[
                    np.abs(pmax[np.flatnonzero(donor)] - pmax[i]).argmin()]
                c2[i], c1[i], c0[i] = c2[j], c1[j], c0[j]

    # --- pass 2: donor-type map (runs for every generator, overwrites) ---
    for i in range(len(out)):
        donor_type = cfg.costs.fallback_donor_map.get(str(fuel[i]))
        if donor_type is None:
            continue
        donor = (fuel == donor_type) & ~np.isnan(c2)
        if cfg.costs.legacy_nearest_capacity:
            _legacy_scan(pmax, c2, c1, c0, i, donor)
        elif donor.any():
            cand = np.flatnonzero(donor & ~np.isnan(join_c2))
            if len(cand) == 0:
                cand = np.flatnonzero(donor)
            j = cand[np.abs(pmax[cand] - pmax[i]).argmin()]
            c2[i], c1[i], c0[i] = c2[j], c1[j], c0[j]

    out["c2"], out["c1"], out["c0"] = c2, c1, c0
    return out
