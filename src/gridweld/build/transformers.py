"""Multi-voltage bus splitting + ideal transformer insertion.

Origin: ``create_transformers!`` / ``copy_bus!`` in ``helper_functions.jl``
lines 476-604 (algorithm doc 02 §create_transformers!). Where branches of
multiple (remapped) kV meet at one bus: split into one bus per level;
generators ride to the MAXIMUM kV, loads to the MINIMUM kV; ideal
transformer branches (tap=1, shift=0, transformer=True — no impedances,
Q7: sized in S3/S4) connect ADJACENT sorted levels only.

Determinism (DV-P2c-2): buses and voltage levels iterate in sorted order —
split-bus/transformer ids are deterministic here but differ from any
particular Julia run (Dict-hash order); validation is topological.
Jitter: split-bus coordinates wiggle by the seeded rng (DV3, amplitude
cfg.build.jitter_deg); excluded from exact coordinate comparison.

Semantics replicated exactly:
* every (bus, kv) level present gets a NEW bus id (the original id is
  deleted even though one level could have kept it);
* gens/loads at buses with NO attached branch are skipped (empty voltage
  dict) — such buses keep their id and get base_kv = 0;
* single-voltage buses keep their id, base_kv = that voltage;
* transformer endpoint Lat/Lon are copied from the (jittered) split buses;
* f_bus = higher-kV split, t_bus = lower-kV split.

Units: kV; coordinates degrees.
Phase: P2c. Validated vs p5 fixtures (counts + voltage-pair multisets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import WeldConfig


def create_transformers(bus_df: pd.DataFrame, branch_df: pd.DataFrame,
                        gen_df: pd.DataFrame, load_df: pd.DataFrame,
                        cfg: WeldConfig, rng: np.random.Generator
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                   pd.DataFrame]:
    """Split multi-voltage buses; returns updated (bus, branch, gen, load).

    Input frames need: bus index/Lat/Lon(+provenance), branch
    index/f_bus/t_bus/kV, gen gen_bus, load load_bus.
    """
    bus = bus_df.copy()
    branch = branch_df.copy()
    gen = gen_df.copy()
    load = load_df.copy()

    # --- registration maps [helper_functions.jl:485-517] ---
    kv_at_bus: dict[int, dict[float, dict]] = {
        int(b): {} for b in bus["index"]}
    f_arr = branch["f_bus"].to_numpy(np.int64)
    t_arr = branch["t_bus"].to_numpy(np.int64)
    kv_arr = branch["kV"].to_numpy(dtype=float)
    for row, (f, t, kv) in enumerate(zip(f_arr, t_arr, kv_arr)):
        for b in (int(f), int(t)):
            kv_at_bus[b].setdefault(float(kv), {"branch": []})
            kv_at_bus[b][float(kv)]["branch"].append(row)
    for gi, gb in enumerate(gen["gen_bus"].to_numpy(np.int64)):
        levels = kv_at_bus[int(gb)]
        if levels:
            levels[max(levels)].setdefault("gen", []).append(gi)
    for li, lb in enumerate(load["load_bus"].to_numpy(np.int64)):
        levels = kv_at_bus[int(lb)]
        if levels:
            levels[min(levels)].setdefault("load", []).append(li)

    base_kv = pd.Series(np.nan, index=bus["index"].to_numpy(np.int64))
    bus_by_id = bus.set_index("index", drop=False)
    next_bus_id = int(bus["index"].max())
    next_branch_id = int(branch["index"].max())
    new_bus_rows: list[dict] = []
    new_branch_rows: list[dict] = []
    drop_bus_ids: list[int] = []
    gen_bus_col = gen["gen_bus"].to_numpy(np.int64).copy()
    load_bus_col = load["load_bus"].to_numpy(np.int64).copy()

    for b in sorted(kv_at_bus):                       # DV-P2c-2 sorted order
        levels = kv_at_bus[b]
        if len(levels) <= 1:
            base_kv.loc[b] = next(iter(levels)) if levels else 0.0
            continue
        split_id: dict[float, int] = {}
        split_coord: dict[float, tuple[float, float]] = {}
        src = bus_by_id.loc[b]
        for kv in sorted(levels):
            next_bus_id += 1
            lat = float(src["Lat"]) + (rng.random() - 0.5) * cfg.build.jitter_deg
            lon = float(src["Lon"]) + (rng.random() - 0.5) * cfg.build.jitter_deg
            row = src.to_dict()
            row.update(index=next_bus_id, Lat=lat, Lon=lon, base_kv=kv)
            new_bus_rows.append(row)
            split_id[kv] = next_bus_id
            split_coord[kv] = (lat, lon)
            comp = levels[kv]
            for br_row in comp["branch"]:
                if f_arr[br_row] == b:
                    f_arr[br_row] = next_bus_id
                if t_arr[br_row] == b:
                    t_arr[br_row] = next_bus_id
            for gi in comp.get("gen", []):
                gen_bus_col[gi] = next_bus_id
            for li in comp.get("load", []):
                load_bus_col[li] = next_bus_id
        drop_bus_ids.append(b)

        sorted_kv = sorted(levels)
        for hv, lv in zip(sorted_kv[1:], sorted_kv[:-1]):
            next_branch_id += 1
            new_branch_rows.append({
                "index": next_branch_id,
                "f_bus": split_id[hv], "t_bus": split_id[lv],
                "tap": cfg.build.transformer_tap,
                "shift": cfg.build.transformer_shift,
                "transformer": True,
                "Lat1": split_coord[hv][0], "Lon1": split_coord[hv][1],
                "Lat2": split_coord[lv][0], "Lon2": split_coord[lv][1],
            })

    branch["f_bus"] = f_arr
    branch["t_bus"] = t_arr
    gen["gen_bus"] = gen_bus_col
    load["load_bus"] = load_bus_col

    bus["base_kv"] = base_kv.loc[bus["index"]].to_numpy()
    bus = bus[~bus["index"].isin(drop_bus_ids)]
    if new_bus_rows:
        bus = pd.concat([bus, pd.DataFrame(new_bus_rows)], ignore_index=True)
    bus = bus.sort_values("index", ignore_index=True)
    if new_branch_rows:
        branch = pd.concat([branch, pd.DataFrame(new_branch_rows)],
                           ignore_index=True)
    if "transformer" not in branch.columns:
        branch["transformer"] = False
    branch["transformer"] = branch["transformer"].map(
        lambda v: bool(v) if pd.notna(v) else False)
    return bus, branch, gen, load
