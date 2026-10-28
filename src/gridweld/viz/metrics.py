"""Diagnostic metrics: MILP assignment quality + per-kV realism statistics.

Origin: ``Analysis/load_metrics.jl`` (algorithm doc 02 §print_load_metrics)
and ``NetworkStatistics.m`` (algorithm doc 03 §NetworkStatistics).

Ported fixes: load_metrics reads its PARAMETER, not the global grid (the
legacy comprehensions read global ``grid`` — subtly wrong for largest_grid;
spec-side fix, diagnostic only, DV-P2c-3); ``leaf_nodes_gen_load`` counts
leaves with NEITHER gen NOR load despite its name (kept, documented).

NetworkStatistics (P3): the GVA·mi sentinel row will use an explicit HV
total row (DV-P3-7) — matlab_compat reproduces the 765 kV clobber.

Units: lengths mi; GVA·mi = Σ Length·RateA/1000.
Phase: P2c (load_metrics) / P3 (network stats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd



def load_metrics(grid) -> pd.DataFrame:
    """Leaf/load/gen coverage counts split by Substation vs AddedNode.

    ``grid`` is a build.S2Grid (Julia-named frames). 8 metric rows ×
    (all, substation, added-node) count columns — the legacy 'count/total'
    strings are formatting only; counts compared in validation.
    """
    from ..build.loads import leaf_buses

    bus, branch = grid.bus, grid.branch
    leaves = set(leaf_buses(bus["index"].to_numpy(),
                            branch["f_bus"].to_numpy(),
                            branch["t_bus"].to_numpy()).tolist())
    with_load = set(grid.load["load_bus"].astype(int).tolist())
    with_gen = set(grid.gen["gen_bus"].astype(int).tolist())
    subs = set(bus.loc[bus["Type"] == "Substation", "index"].astype(int))
    added = set(bus.loc[bus["Type"] == "AddedNode", "index"].astype(int))

    metrics = {
        "leaf_nodes": leaves,
        "load_buses": with_load,
        "gen_buses": with_gen,
        "leaf_no_load": leaves - with_load,
        "leaf_with_load": leaves & with_load,
        "leaf_no_gen": leaves - with_gen,
        "leaf_with_gen": leaves & with_gen,
        "leaf_no_gen_or_load": (leaves - with_gen) - with_load,
    }
    rows = [{"metric": name, "all": len(ids),
             "substation": len(ids & subs), "added_node": len(ids & added)}
            for name, ids in metrics.items()]
    return pd.DataFrame(rows)


#: NetworkStatistics.m hard-codes these regardless of options.VoltageLevels
#: (999 = the HV-total sentinel row of the LENGTH table only).
_STATS_KV = [66, 115, 230, 345, 500, 765]


def network_statistics(branch_df: pd.DataFrame, bus_df: pd.DataFrame,
                       incidence, *, matlab_compat: bool = True
                       ) -> dict[str, object]:
    """NetworkStatistics.m (re-verified vs source 2026-07-17) over the
    MATLAB-named S3 frames.

    * ``length``: per-kV Total/Mean/Std/Quantile10/Median/Quantile90 of
      Branch.Length [mi], rounded to 2 dp (NaN→0); sentinel row 999 =
      Σ Total over 230..765 (the intended HV total).
    * ``branch``: per-kV % of line count (3 dp) and GVA·mi =
      Σ Length·RateA/1000 (rounded then divided, as the source does).
      ⚠ DV-P3-7: the source's ``GVAMiles(end) = Σ(3:6)`` CLOBBERS the
      765 kV row (the 6-row table has no sentinel). ``matlab_compat``
      reproduces the clobber; False appends an explicit ``hv_total`` row.
    * ``bus``: counts/% per unique Bus.kV.
    * ``node_degree`` / ``node_degree_dist``: diag(A·Aᵀ) + histogram.
    * quantiles use MATLAB's (i−0.5)/N positions — numpy method='hazen'
      (the numpy DEFAULT 'linear' differs).
    """
    ln = branch_df["Length"].to_numpy(float)
    kv_from = branch_df["kVFrom"].to_numpy(float)
    rate = branch_df["RateA"].to_numpy(float)

    rows = []
    for kv in _STATS_KV:
        x = ln[kv_from == kv]
        if len(x):
            rows.append({
                "volt": kv, "total": round(float(x.sum()), 2),
                "mean": round(float(x.mean()), 2),
                "std": round(float(np.std(x, ddof=1)), 2) if len(x) > 1 else 0.0,
                "quantile10": round(float(np.quantile(x, 0.1, method="hazen")), 2),
                "median": round(float(np.median(x)), 2),
                "quantile90": round(float(np.quantile(x, 0.9, method="hazen")), 2),
            })
        else:
            rows.append({"volt": kv, "total": 0.0, "mean": 0.0, "std": 0.0,
                         "quantile10": 0.0, "median": 0.0, "quantile90": 0.0})
    hv_total = round(sum(r["total"] for r in rows
                         if r["volt"] in (230, 345, 500, 765)), 2)
    rows.append({"volt": 999, "total": hv_total, "mean": 0.0, "std": 0.0,
                 "quantile10": 0.0, "median": 0.0, "quantile90": 0.0})
    length = pd.DataFrame(rows)

    n_branch = len(branch_df)
    brows = []
    for kv in _STATS_KV:
        mask = kv_from == kv
        brows.append({
            "volt": kv,
            "volt_percent": round(100.0 * mask.sum() / n_branch, 3),
            "gva_miles": round(float((ln[mask] * rate[mask]).sum()), 2) / 1000.0,
        })
    hv_gva = sum(b["gva_miles"] for b in brows
                 if b["volt"] in (230, 345, 500, 765))
    if matlab_compat:
        brows[-1]["gva_miles"] = hv_gva        # DV-P3-7: clobbers the 765 row
    branch_stats = pd.DataFrame(brows)
    if not matlab_compat:
        branch_stats = pd.concat(
            [branch_stats, pd.DataFrame([{"volt": 0, "volt_percent": np.nan,
                                          "gva_miles": hv_gva}])],
            ignore_index=True)

    kv_bus = bus_df["kV"].to_numpy(float)
    bus_stats = pd.DataFrame([
        {"voltage": kv, "number": int((kv_bus == kv).sum()),
         "percent": round(100.0 * (kv_bus == kv).sum() / len(bus_df), 3)}
        for kv in np.unique(kv_bus)])

    a = incidence.tocsr()
    degree = np.asarray((a @ a.T).diagonal()).ravel()
    counts = np.bincount(degree.astype(int))[1:]
    return {
        "length": length, "branch": branch_stats, "bus": bus_stats,
        "node_degree": degree,
        "node_degree_dist": counts / counts.sum() if counts.sum() else counts,
        "bus_branch_ratio": n_branch / len(bus_df),
        "line_bus_ratio": float((branch_df["Tap"].to_numpy(float) == 0).sum())
        / len(bus_df),
    }
