"""Load ingestion + load→bus assignment (Gurobi MILP / exact reduction).

Origin: load cleaning (``generate_grid_script.jl`` lines 209-210) +
``add_loads!`` and the ACTIVE ``assign_loads!`` MILP in
``prototyping_code/optim_load_assign.jl`` lines 105-177 (algorithm doc 02
§assign_loads!). The nearest-bus assign_loads! in helper_functions.jl is
commented out / DORMANT — not ported.

Q5/DV6 RESOLVED (D-1/D-6): the default apportionment is now the corrected
loadsmith CAISO file (Σ pd = 47,257 MW peak anchor; config default flipped
2026-07-20). The legacy "P load is INCORRECT" file (a ~×1.25 overstatement)
is retired as default but stays pinned in the P2c parser fixture test.

Leaf semantics (matches the PowerModelsGraph/Graphs.jl SimpleGraph the
julia rerun used): leaf ⇔ the bus has exactly ONE distinct graph neighbor,
where a self-loop branch contributes the bus itself as a neighbor
(pre-MILP, self-loops still exist — removed later). required = gen-free
leaves; available = substations ∪ required.

MILP [optim_load_assign.jl]: binary x[i,j] over loads × available;
Σ_j x[i,j] = 1; Σ_i x[i,j] ≥ 1 for required j; min Σ d_ij·x[i,j]
(haversine km). Solution read with x > 0.5 (the Julia ``== 1`` float
compare is not replicated).

DV-P2c-4 — ``solver="assignment"`` solves the SAME optimum exactly without
Gurobi: in any optimal solution, each load sits at its nearest available
node unless it is the designated cover of a required leaf, and no leaf
needs two designated loads (dropping a second cover never increases cost).
So optimum = Σ_i d_nearest(i) + min over injective φ: required → loads of
Σ_j (d(φ(j), j) − d_nearest(φ(j))) — a rectangular assignment problem
(scipy linear_sum_assignment). Objectives agree to solver tolerance;
per-load argmin may differ at degenerate optima (doc 02: validate on the
objective, never per-load).

Units: pd MW, qd MVAr; distances km.
Phase: P2c. Validated vs p4_post_assign_loads_load (objective 1e-6 rel).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import MilpResult
from .snapping import haversine_km

log = logging.getLogger(__name__)


def load_loads(path: str | Path) -> pd.DataFrame:
    """Load-apportionment CSV → frame (index/Lat/Lon/pd/qd)."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"id": "index", "MW Load": "pd",
                            "Mvar Load": "qd"})
    return df[["index", "Lat", "Lon", "pd", "qd"]].copy()


def leaf_buses(bus_ids: np.ndarray, f_bus: np.ndarray,
               t_bus: np.ndarray) -> np.ndarray:
    """Bus ids with exactly one DISTINCT neighbor (self-loops count the bus
    itself — Graphs.jl SimpleGraph neighbor semantics)."""
    neighbors: dict[int, set[int]] = {int(b): set() for b in bus_ids}
    for f, t in zip(f_bus.astype(np.int64), t_bus.astype(np.int64)):
        neighbors[int(f)].add(int(t))
        neighbors[int(t)].add(int(f))
    return np.array(sorted(b for b, ns in neighbors.items() if len(ns) == 1),
                    dtype=np.int64)


def _problem_data(load_df, bus_df, branch_df, gen_df, cfg):
    """(available node ids sorted, required ids sorted, cost matrix km)."""
    leaves = leaf_buses(bus_df["index"].to_numpy(),
                        branch_df["f_bus"].to_numpy(),
                        branch_df["t_bus"].to_numpy())
    gen_buses = set(gen_df["gen_bus"].astype(np.int64).tolist())
    required = np.array(sorted(set(leaves.tolist()) - gen_buses),
                        dtype=np.int64)
    subs = bus_df.loc[bus_df["Type"] == "Substation", "index"].to_numpy(np.int64)
    available = np.array(sorted(set(subs.tolist()) | set(required.tolist())),
                         dtype=np.int64)
    pos = bus_df.set_index("index")
    cost = haversine_km(
        load_df["Lat"].to_numpy()[:, None], load_df["Lon"].to_numpy()[:, None],
        pos.loc[available, "Lat"].to_numpy()[None, :],
        pos.loc[available, "Lon"].to_numpy()[None, :],
        cfg.geo.earth_radius_km)
    return available, required, cost


def _solve_reduction(cost: np.ndarray, available: np.ndarray,
                     required: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact optimum via nearest-node + min-regret required-leaf cover
    (DV-P2c-4). Returns (assigned available-column per load, objective)."""
    from scipy.optimize import linear_sum_assignment
    nearest_col = cost.argmin(axis=1)
    nearest_d = cost[np.arange(len(cost)), nearest_col]
    req_cols = np.searchsorted(available, required)
    # regret[j, i] = d(load i → required j) − d(load i → its nearest)
    regret = cost[:, req_cols].T - nearest_d[None, :]
    rows, cols = linear_sum_assignment(regret)
    assign = nearest_col.copy()
    assign[cols] = req_cols[rows]
    objective = float(cost[np.arange(len(cost)), assign].sum())
    return assign, objective


def _solve_gurobi(cost: np.ndarray, available: np.ndarray,
                  required: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The faithful MILP (optim_load_assign.jl model, gurobipy matrix API).
    Returns (assigned column per load, objective, MIP gap)."""
    import gurobipy as gp
    from gurobipy import GRB
    n_load, n_avail = cost.shape
    req_cols = np.searchsorted(available, required)
    with gp.Env() as env, gp.Model("load_assign", env=env) as m:
        x = m.addMVar((n_load, n_avail), vtype=GRB.BINARY)
        m.addConstr(x.sum(axis=1) == 1)
        if len(req_cols):
            m.addConstr(x[:, req_cols].sum(axis=0) >= 1)
        m.setObjective((cost * x).sum(), GRB.MINIMIZE)
        m.optimize()
        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"load MILP failed: gurobi status {m.Status}")
        sol = x.X > 0.5
        assign = sol.argmax(axis=1)
        return assign, float(m.ObjVal), float(m.MIPGap)


def assign_loads(load_df: pd.DataFrame, bus_df: pd.DataFrame,
                 branch_df: pd.DataFrame, gen_df: pd.DataFrame,
                 cfg: WeldConfig, *, solver: str | None = None
                 ) -> tuple[pd.DataFrame, MilpResult]:
    """assign_loads!: MILP load→bus assignment over substations ∪ gen-free
    leaves. ``solver`` overrides cfg.loads.solver ("gurobi" | "assignment").
    Adds ``load_bus`` / ``load_bus_distance`` (warn > cfg.loads.milp_warn_km,
    NOT the 15 km snapping threshold)."""
    solver = solver or cfg.loads.solver
    available, required, cost = _problem_data(load_df, bus_df, branch_df,
                                              gen_df, cfg)
    if solver == "assignment":
        assign, objective = _solve_reduction(cost, available, required)
        gap = 0.0
    elif solver == "gurobi":
        assign, objective, gap = _solve_gurobi(cost, available, required)
    else:
        raise ValueError(f"unknown loads solver {solver!r}")

    out = load_df.copy()
    dist = cost[np.arange(len(out)), assign]
    out["load_bus"] = available[assign]
    out["load_bus_distance"] = dist
    for i in np.flatnonzero(dist > cfg.loads.milp_warn_km):
        log.warning("Closest bus to load %s is %.3f km away",
                    out["index"].iloc[i], dist[i])

    covered = set(out["load_bus"].tolist())
    uncovered = [int(j) for j in required if int(j) not in covered]
    if uncovered:
        raise RuntimeError(f"required leaves left uncovered: {uncovered[:10]}")
    return out, MilpResult(objective_km=objective,
                           assignment=out["load_bus"].to_numpy(np.int64),
                           distance_km=dist, gap=gap)
