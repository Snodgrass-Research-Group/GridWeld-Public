"""DCOPF feasibility loop (gurobipy).

Origin: ``solve_dcopf.m`` + ``decision_variables.m`` + ``dcopf_par.m`` +
the scenario/resize loop of ``PSERCMain_v2.m`` (lines 427-497) — algorithm
doc 05, formulation verbatim: min 0.95·Σdelp + 0.05·(pu-space quadratic
cost) + 0.0·Σdelta s.t. DC balance, setpoint deviation band, unit limits,
slack-relaxed line limits, θ(slack)=0.

Kept quirks (doc 05): λ3=0 makes line limits VACUOUS — overloads surface
afterwards as MaxDCPF > RateA and drive ResizeBranches (resize-to-flow
philosophy; the knob is exposed as w_violation so it's honest); cost is
evaluated on per-unit Pg (implicit 1/SBase² scaling — do NOT "fix" without
a DV row); Σ Gen.A constant kept so objective VALUES match Mosek dumps;
`Puc = PucNew` stays commented out (setpoints anchored to original UCED);
Bus.Vm/Va/PNet/QNet hold the LAST scenario only (Va in RADIANS — the Vpol
convention); the pre-loop GenCommit bookkeeping indexes Gen.BusNum with a
unique-bus-space logical (MATLAB length-mismatch indexing — replicated).
The legacy parfor path is broken (indeterminate temporaries, DV-P4-1) —
``sweep_scenarios`` implements the INTENT with per-scenario results kept.

``decision_variables.m``'s overwritten-sdpvar dance collapses to the sparse
gen→bus incidence Cg (n×g) so Pg_full = Cg @ Pg. BuildANet/YBusCalcDC are
hoisted out of the per-scenario solve (recomputed per OUTER iteration only
— X fixed within one; pure speedup, doc 05).

DV1: gurobipy replaces YALMIP+Mosek (NumericFocus/BarHomogeneous hardening
from the dump run); validate objective (rtol 1e-6), never argmin.
DV-P4-2: non-optimal scenario raises DCOPFInfeasibleError (legacy:
`disp(sol.info); pause` — an interactive hang).

Frame contract (MATLAB-named, the S4→S5 state): ``buses`` Num (contiguous
1-based)/Type (3=slack); ``branches`` From/To (1-based)/X [pu]/RateA [MVA]/
Tap; ``gens`` BusNum (1-based)/MinMW/MaxMW [MW]/A [$/h]/B [$/MWh]/
C [$/MWh²].

Units: everything per-unit on s_base inside the model; MW at the interface.
Phase: P4. Validated vs validation/fixtures/dcopf/ (3 Mosek scenarios).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gurobipy import GRB

from ..config import WeldConfig
from ..model import DCOPFInfeasibleError, DCOPFResult, WeldModel
from ..power.ybus import incidence

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
log = logging.getLogger(__name__)


@dataclass
class DCOPFKernel:
    """One built gurobipy DCOPF model (X fixed) + the per-scenario handles.

    ``c_balance``/``c_dev_lo``/``c_dev_hi`` RHS are the only things that
    change between scenarios (Pd and the UCED setpoints)."""

    model: gp.Model
    theta: object                     # MVar (n,) bus angles [rad]
    pg: object                        # MVar (g,) unit output [pu]
    delp: object                      # MVar (g,) setpoint deviation [pu]
    delta: object                     # MVar (l,) line-limit slack [pu]
    c_balance: object                 # MConstr (n,) RHS = Pd/SBase
    c_dev_lo: object                  # MConstr (g,) pg+delp ≥ Puc/SBase
    c_dev_hi: object                  # MConstr (g,) pg−delp ≤ Puc/SBase
    a_inc: sp.csc_matrix              # (n,l) signed incidence (BuildANet)
    b_prim: np.ndarray                # (l,) 1/X primitive [pu]
    bbus: sp.csc_matrix               # (n,n) A·diag(1/X)·Aᵀ
    s_base: float                     # MVA


def build_dcopf(branches: pd.DataFrame, gens: pd.DataFrame,
                buses: pd.DataFrame, cfg: WeldConfig, *,
                env: gp.Env | None = None) -> DCOPFKernel:
    """Build the gurobipy model once per outer iteration (X fixed): vars
    theta/pg/delp/delta, sparse gen→bus incidence Cg (collapses
    decision_variables.m's overwritten sdpvar dance), balance rows with
    placeholder RHS, verbatim QP objective incl. the λ2·ΣA constant
    (solve_dcopf.m lines 16-54). Unit limits are variable BOUNDS (same
    feasible set as the MATLAB double inequality)."""
    fs = cfg.feasibility
    s_base = cfg.system.s_base_mva
    n, g, n_lines = len(buses), len(gens), len(branches)

    from_b = branches["From"].to_numpy(np.int64) - 1
    to_b = branches["To"].to_numpy(np.int64) - 1
    x_pu = branches["X"].to_numpy(float)
    rate_pu = branches["RateA"].to_numpy(float) / s_base
    busnum0 = gens["BusNum"].to_numpy(np.int64) - 1

    a = incidence(from_b, to_b, n)
    b_prim = 1.0 / x_pu
    bbus = (a @ sp.diags(b_prim) @ a.T).tocsc()
    cg = sp.csc_matrix((np.ones(g), (busnum0, np.arange(g))), shape=(n, g))
    slack_rows = np.flatnonzero(buses["Type"].to_numpy(float) == 3)

    m = gp.Model("dcopf", env=env)
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = fs.numeric_focus
    if fs.bar_homogeneous:
        m.Params.BarHomogeneous = 1

    th_lb = np.full(n, -GRB.INFINITY)
    th_ub = np.full(n, GRB.INFINITY)
    th_lb[slack_rows] = 0.0                    # theta(slack_bus) == 0
    th_ub[slack_rows] = 0.0
    theta = m.addMVar(n, lb=th_lb, ub=th_ub, name="theta")
    pg = m.addMVar(g, lb=gens["MinMW"].to_numpy(float) / s_base,
                   ub=gens["MaxMW"].to_numpy(float) / s_base, name="pg")
    delp = m.addMVar(g, lb=0.0, name="delp")
    delta = m.addMVar(n_lines, lb=0.0, name="delta")

    c_balance = m.addConstr(cg @ pg - bbus @ theta == np.zeros(n),
                            name="balance")
    c_dev_lo = m.addConstr(pg + delp >= np.zeros(g), name="dev_lo")
    c_dev_hi = m.addConstr(pg - delp <= np.zeros(g), name="dev_hi")
    flow_map = (sp.diags(b_prim) @ a.T).tocsr()       # f = Bprim·Aᵀ·θ [pu]
    m.addConstr(flow_map @ theta + delta >= -rate_pu, name="lim_lo")
    m.addConstr(flow_map @ theta - delta <= rate_pu, name="lim_hi")

    quad = sp.diags(gens["C"].to_numpy(float)).tocsr()
    cost = (pg @ quad @ pg + gens["B"].to_numpy(float) @ pg
            + float(gens["A"].sum()))
    m.setObjective(fs.w_deviation * delp.sum() + fs.w_cost * cost
                   + fs.w_violation * delta.sum(), GRB.MINIMIZE)
    return DCOPFKernel(model=m, theta=theta, pg=pg, delp=delp, delta=delta,
                       c_balance=c_balance, c_dev_lo=c_dev_lo,
                       c_dev_hi=c_dev_hi, a_inc=a, b_prim=b_prim, bbus=bbus,
                       s_base=s_base)


def solve_scenario(kernel: DCOPFKernel, pd_mw: np.ndarray,
                   pg_set_mw: np.ndarray, cfg: WeldConfig, *,
                   scenario: int | None = None) -> DCOPFResult:
    """Update balance + deviation-band RHS in place, re-optimize (warm
    start). ``pd_mw`` may be complex (real part taken — solve_dcopf.m:20).
    Raises DCOPFInfeasibleError on any non-optimal status (DV-P4-2)."""
    s_base = kernel.s_base
    pd_pu = np.real(np.asarray(pd_mw)) / s_base
    puc_pu = np.asarray(pg_set_mw, dtype=float) / s_base
    kernel.c_balance.setAttr("RHS", pd_pu)
    kernel.c_dev_lo.setAttr("RHS", puc_pu)
    kernel.c_dev_hi.setAttr("RHS", puc_pu)
    m = kernel.model
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        # DV1-hardening family: barrier occasionally exits SUBOPTIMAL
        # (status 13) on these QPs — one deterministic dual-simplex retry
        # before the DV-P4-2 raise (first seen: D-1 corrected-load run,
        # scenario 26)
        log.warning("DCOPF scenario %s: status %d — retrying with dual "
                    "simplex", scenario, m.Status)
        method0 = m.Params.Method
        m.Params.Method = 1
        m.reset()
        m.optimize()
        m.Params.Method = method0
    if m.Status != GRB.OPTIMAL:
        label = "?" if scenario is None else str(scenario)
        raise DCOPFInfeasibleError(
            f"DCOPF scenario {label} did not solve to optimality: gurobi "
            f"status {m.Status} (legacy behavior was disp+pause, DV-P4-2)")

    theta = np.asarray(kernel.theta.X)
    angle_diff = kernel.a_inc.T @ theta               # ANet'·Va [rad]
    flows = s_base * kernel.b_prim * angle_diff       # SBase·Bprim·Δθ [MW]
    v_rect = np.exp(1j * theta)                       # pol2cart(Va, 1.0)
    snet = s_base * v_rect * np.conj(kernel.bbus @ v_rect)   # Q9 convention
    return DCOPFResult(
        theta_rad=theta, pg_mw=np.asarray(kernel.pg.X) * s_base,
        delp_pu=np.asarray(kernel.delp.X), delta_pu=np.asarray(kernel.delta.X),
        flow_mw=flows, angle_diff_rad=angle_diff, snet_mva=snet,
        objective=float(m.ObjVal), status=int(m.Status))


def _resolve_workers(cfg: WeldConfig, n_scenarios: int) -> int:
    """max_workers=0 → cpu_count; capped by parallel.worker_cap (32, the
    MATLAB parpool cap) and the scenario count. The RAM gate is dropped
    (doc 05 config table)."""
    w = cfg.feasibility.max_workers or (os.cpu_count() or 1)
    return max(1, min(w, cfg.parallel.worker_cap, n_scenarios))


def _solve_chunk(branches: pd.DataFrame, gens: pd.DataFrame,
                 buses: pd.DataFrame, cfg: WeldConfig, pd_cols: np.ndarray,
                 puc_cols: np.ndarray, indices: list[int]) -> list[DCOPFResult]:
    """Worker: build one kernel, solve a contiguous scenario chunk
    (chunking amortizes the build; gurobipy models are not picklable)."""
    kernel = build_dcopf(branches, gens, buses, cfg)
    return [solve_scenario(kernel, pd_cols[:, j], puc_cols[:, j], cfg,
                           scenario=k)
            for j, k in enumerate(indices)]


def sweep_scenarios(branches: pd.DataFrame, gens: pd.DataFrame,
                    buses: pd.DataFrame, pd_mw: np.ndarray,
                    pg_set_mw: np.ndarray, cfg: WeldConfig, *,
                    gurobi_env: gp.Env | None = None) -> list[DCOPFResult]:
    """All scenarios (``pd_mw`` (n,K) MW, may be complex; ``pg_set_mw``
    (g,K) MW). One process per chunk of scenarios (DV-P4-1: the parallel
    INTENT of the broken dcopf_par.m parfor); serial with a single warm-
    started kernel when one worker resolves. Results are scenario-ordered."""
    pd_mw = np.asarray(pd_mw)
    pg_set_mw = np.asarray(pg_set_mw, dtype=float)
    n_scen = pd_mw.shape[1]
    if pg_set_mw.shape[1] != n_scen:
        raise ValueError(f"pg_set_mw has {pg_set_mw.shape[1]} columns, "
                         f"pd_mw has {n_scen}")
    workers = _resolve_workers(cfg, n_scen)
    if workers == 1:
        kernel = build_dcopf(branches, gens, buses, cfg, env=gurobi_env)
        return [solve_scenario(kernel, pd_mw[:, k], pg_set_mw[:, k], cfg,
                               scenario=k)
                for k in range(n_scen)]
    chunks = [c for c in np.array_split(np.arange(n_scen), workers) if len(c)]
    results: list[DCOPFResult] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_solve_chunk, branches, gens, buses, cfg,
                             pd_mw[:, c], pg_set_mw[:, c], c.tolist())
                   for c in chunks]
        for fut in futures:                    # chunks are contiguous+ordered
            results.extend(fut.result())
    return results


def _commitment_bookkeeping(bus_df: pd.DataFrame, snet_full: np.ndarray,
                            gen_busnum1: np.ndarray) -> pd.DataFrame:
    """PSERCMain_v2.m:431-434, faithful: maxscenario = argmax column sum of
    |real(SNetFull)|; GenCommit = real(SNetFull(unique buses, max)) > 0
    (MATLAB compares REAL parts of complex); ⚠ GenCommit (unique-bus-space,
    length |u|) then logically indexes the RAW Gen.BusNum — a length-
    mismatch quirk replicated verbatim: positions of True within the first
    |u| generator ROWS. Bus.GenMWSet at the resulting buses gets
    |real(SNetFull(bus, maxscenario))|."""
    bus = bus_df.copy()
    max_scen = int(np.abs(np.real(snet_full)).sum(axis=0).argmax())
    u = np.unique(gen_busnum1)
    commit = np.real(snet_full[u - 1, max_scen]) > 0
    gen_buses = np.unique(gen_busnum1[np.flatnonzero(commit)])
    gen_mw_set = (bus["GenMWSet"].to_numpy(float).copy()
                  if "GenMWSet" in bus.columns else np.zeros(len(bus)))
    gen_mw_set[gen_buses - 1] = np.abs(
        np.real(snet_full[gen_buses - 1, max_scen]))
    bus["GenMWSet"] = gen_mw_set
    return bus


def run_dcopf_upgrade(bus_df: pd.DataFrame, branch_df: pd.DataFrame,
                      gen_df: pd.DataFrame, snet_full: np.ndarray,
                      load_scenarios: np.ndarray, pg_uc: np.ndarray,
                      xfdata: pd.DataFrame, xfxr: pd.DataFrame,
                      catalog_crucible: pd.DataFrame, conductor_catalog,
                      gmd_tables, cfg: WeldConfig, rng: np.random.Generator,
                      *, max_passes: int = 200,
                      gurobi_env: gp.Env | None = None,
                      compute_stats: bool = True
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict,
                                 pd.DataFrame]:
    """The S5 while-loop (PSERCMain_v2.m:427-497) over MATLAB-named frames:
    sweep DCOPF → MaxDCPF/MaxAngleDiff envelopes → the shared resize +
    reparameterization chain (resize_reparam_pass — verbatim-identical to
    the S4 DC loop's) → overload count vs TermCriteria (seed
    feasibility.term_overloads = 2, +1 per pass after term_relax_after).
    Setpoints ``pg_uc`` stay anchored to the ORIGINAL UCED dispatch every
    pass (`Puc = PucNew` is commented out upstream — kept).

    ``load_scenarios`` (n,K) complex MW is horizontally duplicated to match
    ``snet_full``'s column count (v2:398 `Pd = [LoadScenarios,
    LoadScenarios]`); ``pg_uc`` is (g, totalscenarios) MW.

    Returns (bus frame, branch frame, updates dict, iteration log). Updates:
    ``pg`` (g×K MW, the OPF dispatch replacing UCED — PF.Pg), ``snet_full_opf``
    (n×K complex — the v2:485-491 recompute incl. the ScenarioIdx wrap),
    ``dead_lines`` (QA, v2:499-500), ``dc_stats`` (DCPowerFlowFull pass).
    Log columns: iter/overloads/total_rate_mva/max_dcpf (s7 contract)."""
    from ..resize.dc_loop import resize_reparam_pass

    n = len(bus_df)
    total = snet_full.shape[1]
    k_ls = load_scenarios.shape[1]
    if total > 2 * k_ls:
        raise ValueError(f"SNetFull has {total} scenario columns but "
                         f"[LoadScenarios, LoadScenarios] only {2 * k_ls}")
    if pg_uc.shape != (len(gen_df), total):
        raise ValueError(f"pg_uc shape {pg_uc.shape} != "
                         f"({len(gen_df)}, {total})")
    if not np.array_equal(bus_df["Num"].to_numpy(np.int64),
                          np.arange(1, n + 1)):
        raise ValueError("Bus.Num must be contiguous 1..n (post-ReadMatpower "
                         "invariant; positional GenMWSet/Va writes rely on it)")

    busnum1 = gen_df["BusNum"].to_numpy(np.int64)
    bus = _commitment_bookkeeping(bus_df, snet_full, busnum1)
    br = branch_df.copy()
    pd_mat = np.hstack([load_scenarios, load_scenarios])[:, :total]

    term = cfg.feasibility.term_overloads
    log_rows = []
    puc_new = np.empty((len(gen_df), total))
    for it in range(1, max_passes + 1):
        results = sweep_scenarios(br, gen_df, bus, pd_mat, pg_uc, cfg,
                                  gurobi_env=gurobi_env)
        flows = np.column_stack([r.flow_mw for r in results])
        angles = np.column_stack([r.angle_diff_rad for r in results])
        puc_new = np.column_stack([r.pg_mw for r in results])
        last = results[-1]                    # last-scenario-only quirk
        bus["Vm"] = 1.0
        bus["Va"] = last.theta_rad            # RADIANS (Vpol convention)
        bus["PNet"] = np.real(last.snet_mva)
        bus["QNet"] = np.imag(last.snet_mva)

        max_dcpf = np.abs(flows).max(axis=1)
        br["MaxDCPF"] = max_dcpf
        br["MaxAngleDiff"] = np.abs(angles).max(axis=1)     # radians here
        br = resize_reparam_pass(br, max_dcpf, xfdata, xfxr,
                                 catalog_crucible, conductor_catalog,
                                 gmd_tables, cfg, rng)
        overloads = int((br["RateA"].to_numpy(float)
                         - br["MaxDCPF"].to_numpy(float)
                         * cfg.resize.conductor_margin < 0).sum())
        log_rows.append({"iter": it, "overloads": overloads,
                         "total_rate_mva": float(br["RateA"].sum()),
                         "max_dcpf": float(br["MaxDCPF"].max())})
        if overloads <= term:
            break
        if it + 1 > cfg.feasibility.term_relax_after:
            term += 1

    snet_opf = np.empty((n, total), dtype=complex)
    scenario_idx = 0                          # the v2:485-491 wrap, faithful
    for k in range(1, total + 1):
        inj = np.bincount(busnum1 - 1, weights=puc_new[:, k - 1],
                          minlength=n)
        snet_opf[:, k - 1] = inj - load_scenarios[:, k - 1 - scenario_idx]
        if k >= k_ls:
            scenario_idx = k_ls

    updates: dict = {"pg": puc_new, "snet_full_opf": snet_opf,
                     "dead_lines": dead_line_report(br, cfg)}
    if compute_stats:
        from ..power.dc import dc_stats, solve_dc_scenarios
        pf_model = WeldModel(
            buses=pd.DataFrame({"bus_id": np.arange(n, dtype=np.int64)}),
            branches=pd.DataFrame({
                "from_bus": br["From"].to_numpy(np.int64) - 1,
                "to_bus": br["To"].to_numpy(np.int64) - 1,
                "tap": br["Tap"].to_numpy(float),
                "r_pu": br["R"].to_numpy(float),
                "x_pu": br["X"].to_numpy(float),
                "b_pu": br["B"].to_numpy(float),
                "capacity_mva": br["RateA"].to_numpy(float)}))
        res = solve_dc_scenarios(pf_model, snet_full,
                                 s_base=cfg.system.s_base_mva)
        updates["dc_stats"] = dc_stats(pf_model, res, snet_full,
                                       s_base=cfg.system.s_base_mva)
    return bus, br, updates, pd.DataFrame(log_rows)


class DCOPFUpgrader:
    """The pipeline-facing S5 stage ('dcopf'). Thin adapter over
    :func:`run_dcopf_upgrade`; catalogs/tables default-load from
    ``gridweld/data`` when not injected.

    WeldModel contract (provisional until P5 pins the checkpoint schema):
    ``model.buses/branches/generators`` hold the MATLAB-named S4-state
    frames; ``model.scenarios.tiers['full']`` is SNetFull (n×K complex MW);
    ``model.meta['load_scenarios']`` (n×K_LS complex MW) and
    ``model.meta['pg_uc']`` (g×K MW, PF.Pg from UCED) must be present.
    After upgrade: frames replaced, ``meta['pg']`` / ``meta['snet_full_opf']``
    / ``meta['dcopf_log']`` / ``meta['dead_lines']`` / ``meta['dc_stats']``
    written (the S5 contract for S6/S7)."""

    def __init__(self, config: WeldConfig, catalog=None, *,
                 catalog_crucible: pd.DataFrame | None = None,
                 xfdata: pd.DataFrame | None = None,
                 xfxr: pd.DataFrame | None = None,
                 gmd_tables=None) -> None:
        from ..params.conductors import crucible_catalog, load_cats_catalog
        from ..params.impedance import load_gmd_tables
        self.config = config
        self.catalog = catalog if catalog is not None else load_cats_catalog()
        self.catalog_crucible = (catalog_crucible if catalog_crucible
                                 is not None else crucible_catalog())
        self.xfdata = (xfdata if xfdata is not None
                       else pd.read_parquet(_DATA_DIR / "xfdata.parquet"))
        self.xfxr = (xfxr if xfxr is not None
                     else pd.read_parquet(_DATA_DIR / "xfxr.parquet"))
        self.gmd_tables = (gmd_tables if gmd_tables is not None
                           else load_gmd_tables())

    def upgrade(self, model: WeldModel,
                rng: np.random.Generator | None = None, *,
                max_passes: int = 200, compute_stats: bool = True
                ) -> WeldModel:
        if rng is None:                        # D10: normally threaded in
            rng = np.random.default_rng(self.config.system.rng_seed)
        bus, br, updates, log_df = run_dcopf_upgrade(
            model.buses, model.branches, model.generators,
            model.scenarios.tiers["full"], model.meta["load_scenarios"],
            model.meta["pg_uc"], self.xfdata, self.xfxr,
            self.catalog_crucible, self.catalog, self.gmd_tables,
            self.config, rng, max_passes=max_passes,
            compute_stats=compute_stats)
        model.buses = bus
        model.branches = br
        model.meta["pg"] = updates["pg"]
        model.meta["snet_full_opf"] = updates["snet_full_opf"]
        model.meta["dead_lines"] = updates["dead_lines"]
        if compute_stats:
            model.meta["dc_stats"] = updates["dc_stats"]
        model.meta["dcopf_log"] = log_df
        return model


def dead_line_report(branch_df: pd.DataFrame,
                     cfg: WeldConfig) -> pd.DataFrame:
    """QA diagnostic (PSERCMain_v2.m:499-500): branches with
    kV ≥ qa.deadline_kv_min AND MaxDCPF < qa.deadline_flow_mw (dead
    high-voltage lines). Returns the offending rows (copy)."""
    qa = cfg.feasibility.qa
    kv = branch_df["kV"].to_numpy(float)
    max_dcpf = branch_df["MaxDCPF"].to_numpy(float)
    return branch_df.loc[(kv >= qa.deadline_kv_min)
                         & (max_dcpf < qa.deadline_flow_mw)].copy()
