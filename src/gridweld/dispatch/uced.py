"""UCED / UCCD: economic + complementary dispatch (gurobipy).

Origin (re-verified against the MATLAB sources 2026-07-17): ``UCED.m`` /
``UCCD.m``. QP dispatch + crude commitment: UCED decommits the 3 HIGHEST
marginal-cost units per round until spinning reserve ≤ reserve; UCCD forces
expensive units online by turning off fully-dispatched cheap units (5%
headroom test — the comment says 3%, the CODE says 5%), then drops the 3
LOWEST average-cost units.

⚠ Hydro matching (source fact, correcting doc 03): BOTH files use
``strcmpi(FuelType,'Hydro')`` — EXACT case-insensitive equality. CATS fuel
strings ('Conventional Hydroelectric', 'Hydroelectric Pumped Storage')
never equal 'Hydro', so the HydroScaling argument is a NO-OP for CATS in
both dispatchers (the thirds schedule only bites through CreateSNet's
``contains``-based capacity guard). Replicated exactly.

UCED.m is ported faithfully HERE (sequencing differs from gridforge
``solve_uced`` — its lb=0 ranking solves would change the decommit order,
and the committed-set signatures drive the SNet tier widths); UCCD is
gridcrucible ``data.dispatch.uccd`` REUSE-AS-IS (step-for-step port), with
its ``_solve_qp`` reused for the QP so both dispatchers share one kernel.
½C QP convention kept (DV-P3-5, cplexqp uncompensated); CPLEX→Gurobi
degenerate ties are DV-P3-6.

Fleet contract (gridcrucible's): columns MinMW/MaxMW/A/B/C/FuelType/BusNum;
the ORIGINAL frame index is preserved on the surviving fleet.

Units: MW; reserve fraction; A $/h, B $/MWh, C $/MWh².
Phase: P3. Validated vs s5_snetfull within solver tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gridcrucible.data.dispatch import _solve_qp
from gridcrucible.data.dispatch import uccd as uccd  # noqa: F401  (re-export)


def marginal_cost(fleet: pd.DataFrame, x_mw: np.ndarray) -> np.ndarray:
    """True marginal cost B + 2C·x (UCED's decommit ranking — NOT ½C)."""
    return (fleet["B"].to_numpy(float)
            + 2 * fleet["C"].to_numpy(float) * np.asarray(x_mw, float))


def _qp(gens: pd.DataFrame, lb: np.ndarray, total: float,
        env=None) -> np.ndarray | None:
    return _solve_qp(gens["C"].to_numpy(float), gens["B"].to_numpy(float),
                     lb, gens["MaxMW"].to_numpy(float), total, env)


def uced(fleet: pd.DataFrame, hydro_scaling: float, load_mw, reserve: float,
         gurobi_env=None, max_iterations: int = 500
         ) -> tuple[np.ndarray, pd.DataFrame]:
    """UCED.m, step for step. ``load_mw``: array (only the sum is used) or
    scalar total. Returns (x, surviving_fleet) — original index preserved.

    Sequence [UCED.m line refs]:
    * hydro scaling via EXACT 'Hydro' match [5-7] (no-op for CATS);
    * solve at lb=MinMW [11]; if reserve excessive: lb=0 solve, DROP
      x<1e-3 units, re-solve at MinMW, lb=ones fallback on infeasible,
      rank marginal cost B+2Cx → top-3 [12-26];
    * decommit loop while cap·(1−R) > load and n>5: stop if dropping the 3
      would strand load; delete; lb = ones if the PRE-LOOP solve was
      infeasible else MinMW (the fval state is FROZEN before the loop —
      quirk kept); re-solve, re-rank [29-49];
    * final re-solve at MinMW; on infeasible: MinMW→0, drop x<1e-3 [50-57];
    * cleanup: drop x ≤ 0.1 MW, rescale survivors to the load [58-64].
    """
    gens = fleet.copy()
    hydro = gens["FuelType"].astype(str).str.strip().str.casefold() == "hydro"
    gens.loc[hydro, "MaxMW"] *= hydro_scaling
    gens.loc[hydro, "MinMW"] *= hydro_scaling
    total = float(np.asarray(load_mw, float).sum())

    x = _qp(gens, gens["MinMW"].to_numpy(float), total, gurobi_env)
    fval_empty = x is None
    worst = None
    if gens["MaxMW"].sum() * (1 - reserve) > total:
        x0 = _qp(gens, np.zeros(len(gens)), total, gurobi_env)
        if x0 is not None:
            gens = gens[x0 >= 1e-3]
        x = _qp(gens, gens["MinMW"].to_numpy(float), total, gurobi_env)
        fval_empty = x is None
        if fval_empty:
            x = _qp(gens, np.ones(len(gens)), total, gurobi_env)
        if x is None:
            raise RuntimeError("UCED: QP infeasible even at lb=1 MW")
        mc = marginal_cost(gens, x)
        worst = np.argsort(mc)[-3:]

    iters = 0
    while (gens["MaxMW"].sum() * (1 - reserve) > total and len(gens) > 5
           and worst is not None and iters < max_iterations):
        if gens["MaxMW"].sum() - gens["MaxMW"].to_numpy(float)[worst].sum() \
                <= total:
            break
        gens = gens.drop(index=gens.index[worst])
        lb = (np.ones(len(gens)) if fval_empty
              else gens["MinMW"].to_numpy(float))
        x = _qp(gens, lb, total, gurobi_env)
        if x is None:
            break                       # MATLAB would crash ranking on empty
        worst = np.argsort(marginal_cost(gens, x))[-3:]
        iters += 1

    x = _qp(gens, gens["MinMW"].to_numpy(float), total, gurobi_env)
    if x is None:
        gens = gens.assign(MinMW=0.0)
        x = _qp(gens, np.zeros(len(gens)), total, gurobi_env)
        if x is None:
            raise RuntimeError("UCED: final QP infeasible even at lb=0")
        keep = x >= 1e-3
        gens = gens[keep]
        x = x[keep]

    low = x <= 0.1
    if low.any():
        gens = gens[~low]
        x = x[~low]
        x = x * (1 + (total - x.sum()) / x.sum())
    return x, gens
