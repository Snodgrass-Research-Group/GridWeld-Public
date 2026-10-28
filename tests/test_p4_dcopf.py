"""P4 validation: the DCOPF kernel vs the 3 Mosek-solved fixture scenarios
(DV1 — objective, never argmin at degenerate optima), fixture-internal
consistency + fixed-dispatch reconstruction (incidence/B-matrix/sign
conventions, solver-free), serial/parallel sweep parity (DV-P4-1), the
commitment-bookkeeping quirk, the S5 upgrade loop (slow), and the
assigning_pmax_pmin port (re-application must reproduce the fixture Gen's
limits — the fixture Gen already went through the MATLAB original)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from gridweld.build.generators import assign_gen_limits
from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.feasibility.dcopf import (_commitment_bookkeeping, build_dcopf,
                                        dead_line_report, run_dcopf_upgrade,
                                        solve_scenario, sweep_scenarios)
from gridweld.model import DCOPFInfeasibleError
from gridweld.power.ybus import incidence

FIX = REPO_ROOT / "validation" / "fixtures" / "dcopf"
DATA = REPO_ROOT / "src" / "gridweld" / "data"
LEGACY_EIA = (REPO_ROOT / "legacy" / "Jonathan_Code" / "Aditya_Code"
              / "Data Files" / "EIA_Generators_Combined_2010.xlsx")
needs_dcopf = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                 reason="dcopf fixtures not extracted")
needs_eia = pytest.mark.skipif(not LEGACY_EIA.exists(),
                               reason="legacy EIA workbook not present")

S_BASE = 100.0


# ---------------------------------------------------------------------------
# toy networks (kernel semantics, solver-cheap)
# ---------------------------------------------------------------------------


def _toy_frames(rate_a: float = 250.0):
    buses = pd.DataFrame({"Num": [1, 2, 3], "Type": [3, 2, 1]})
    branches = pd.DataFrame({
        "From": [1, 2, 1], "To": [2, 3, 3],
        "X": [0.1, 0.2, 0.25], "RateA": [rate_a] * 3, "Tap": [0.0] * 3})
    gens = pd.DataFrame({
        "BusNum": [1, 2], "MinMW": [0.0, 0.0], "MaxMW": [300.0, 200.0],
        "A": [5.0, 7.0], "B": [10.0, 14.0], "C": [0.02, 0.035]})
    return branches, gens, buses


def test_kernel_balance_and_deviation_semantics():
    branches, gens, buses = _toy_frames()
    cfg = WeldConfig()
    kernel = build_dcopf(branches, gens, buses, cfg)
    pd_mw = np.array([0.0, 0.0, 120.0])
    puc = np.array([80.0, 40.0])
    res = solve_scenario(kernel, pd_mw, puc, cfg, scenario=0)

    busnum0 = gens["BusNum"].to_numpy(np.int64) - 1
    inj_pu = np.bincount(busnum0, weights=res.pg_mw, minlength=3) / S_BASE
    residual = inj_pu - kernel.bbus @ res.theta_rad - pd_mw / S_BASE
    assert np.abs(residual).max() < 1e-7          # DC balance rows hold
    assert res.theta_rad[0] == 0.0                # slack pinned
    assert abs(res.pg_mw.sum() - 120.0) < 1e-5    # lossless DC
    # delp is exactly the L1 deviation at the optimum (lambda_1 > 0)
    assert np.allclose(res.delp_pu, np.abs(res.pg_mw - puc) / S_BASE,
                       atol=1e-7)


def test_objective_analytic_single_gen():
    """2-bus forced dispatch: objective is fully determined —
    0.95·|Pd−Puc|/SBase + 0.05·(C·pg² + B·pg + A) with pg in pu
    (the verbatim pu-space cost quirk, doc 05)."""
    buses = pd.DataFrame({"Num": [1, 2], "Type": [3, 1]})
    branches = pd.DataFrame({"From": [1], "To": [2], "X": [0.1],
                             "RateA": [500.0], "Tap": [0.0]})
    gens = pd.DataFrame({"BusNum": [1], "MinMW": [0.0], "MaxMW": [400.0],
                         "A": [3.0], "B": [12.0], "C": [0.01]})
    cfg = WeldConfig()
    kernel = build_dcopf(branches, gens, buses, cfg)
    res = solve_scenario(kernel, np.array([0.0, 150.0]), np.array([100.0]),
                         cfg)
    pg_pu = 1.5
    expected = (0.95 * 0.5
                + 0.05 * (0.01 * pg_pu**2 + 12.0 * pg_pu + 3.0))
    assert np.isclose(res.objective, expected, rtol=1e-6)
    assert np.isclose(res.flow_mw[0], 150.0, atol=1e-4)   # 1→2 positive
    assert np.isclose(res.angle_diff_rad[0], 0.15, atol=1e-7)


def test_line_limits_vacuous_with_zero_weight():
    """λ3 = 0 (doc 05 quirk): a 1-MVA rating does not bind — the overload
    surfaces in the flow and is left for ResizeBranches."""
    branches, gens, buses = _toy_frames(rate_a=1.0)
    cfg = WeldConfig()
    kernel = build_dcopf(branches, gens, buses, cfg)
    res = solve_scenario(kernel, np.array([0.0, 0.0, 120.0]),
                         np.array([80.0, 40.0]), cfg)
    assert np.abs(res.flow_mw).max() > 1.0        # solved anyway, overloaded


def test_infeasible_raises():
    branches, gens, buses = _toy_frames()
    cfg = WeldConfig()
    kernel = build_dcopf(branches, gens, buses, cfg)
    with pytest.raises(DCOPFInfeasibleError):
        solve_scenario(kernel, np.array([0.0, 0.0, 900.0]),  # > ΣMaxMW
                       np.array([80.0, 40.0]), cfg, scenario=7)


def test_sweep_parallel_matches_serial():
    """DV-P4-1 parity: process-pool sweep ≡ serial warm-started sweep on a
    strictly convex toy (unique optimum — exact comparison is fair)."""
    branches, gens, buses = _toy_frames()
    loads = np.array([100.0, 120.0, 140.0, 160.0])
    pd_mw = np.zeros((3, 4))
    pd_mw[2, :] = loads
    puc = np.column_stack([[60.0, 40.0], [70.0, 50.0],
                           [80.0, 60.0], [90.0, 70.0]])
    cfg_serial = WeldConfig()
    cfg_serial.feasibility.max_workers = 1
    cfg_par = WeldConfig()
    cfg_par.feasibility.max_workers = 2
    serial = sweep_scenarios(branches, gens, buses, pd_mw, puc, cfg_serial)
    par = sweep_scenarios(branches, gens, buses, pd_mw, puc, cfg_par)
    assert len(serial) == len(par) == 4
    for s, p in zip(serial, par):
        assert np.isclose(s.objective, p.objective, rtol=1e-8)
        assert np.allclose(s.pg_mw, p.pg_mw, atol=1e-5)
        assert np.allclose(s.flow_mw, p.flow_mw, atol=1e-5)


def test_commitment_bookkeeping_quirk():
    """v2:431-434: the unique-bus-space logical indexes RAW Gen.BusNum —
    with gens at buses [2,3,3,5] and commit pattern [T,F,T] over unique
    buses [2,3,5], the selected buses are {2,3} (positions 0,2 of the RAW
    list), NOT {2,5}."""
    bus = pd.DataFrame({"Num": np.arange(1, 6), "GenMWSet": np.zeros(5)})
    snet = np.zeros((5, 2), dtype=complex)
    snet[:, 0] = [1.0, 10.0, -5.0, 2.0, 7.0]      # column 0 is maxscenario
    snet[:, 1] = [0.1, 0.2, 0.1, 0.1, 0.1]
    out = _commitment_bookkeeping(bus, snet, np.array([2, 3, 3, 5]))
    expected = np.zeros(5)
    expected[1] = 10.0                             # bus 2
    expected[2] = 5.0                              # bus 3 (|−5| — the quirk)
    assert np.allclose(out["GenMWSet"].to_numpy(), expected)


def test_dead_line_report():
    cfg = WeldConfig()
    br = pd.DataFrame({"kV": [230.0, 66.0, 115.0],
                       "MaxDCPF": [1e-3, 1e-3, 5.0]})
    dead = dead_line_report(br, cfg)
    assert list(dead.index) == [0]


# ---------------------------------------------------------------------------
# fixture validation (DV1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fx():
    bus = pd.read_csv(FIX / "in_bus.csv", low_memory=False)
    br = pd.read_csv(FIX / "in_branch.csv", low_memory=False)
    gen = pd.read_csv(FIX / "in_gen.csv", low_memory=False)
    ls = (pd.read_csv(FIX / "in_loadscenarios_real.csv",
                      header=None).to_numpy()
          + 1j * pd.read_csv(FIX / "in_loadscenarios_imag.csv",
                             header=None).to_numpy())
    pg_uc = pd.read_csv(FIX / "in_pg.csv", header=None).to_numpy()
    status = pd.read_csv(FIX / "out_scenario_status.csv",
                         header=None).to_numpy(int)
    ok = status[status[:, 1] == 0, 0] - 1          # 0-based Pd/Puc columns
    return SimpleNamespace(
        bus=bus, br=br, gen=gen, ls=ls, pg_uc=pg_uc, scenarios=ok.tolist(),
        flows=pd.read_csv(FIX / "out_flows.csv", header=None).to_numpy(),
        angles=pd.read_csv(FIX / "out_anglediff.csv",
                           header=None).to_numpy(),
        pucnew=pd.read_csv(FIX / "out_pucnew.csv", header=None).to_numpy(),
        snet_real=pd.read_csv(FIX / "out_snetcalc_real.csv",
                              header=None).to_numpy(),
        vbuspolar=pd.read_csv(FIX / "out_vbuspolar_first_ok.csv",
                              header=None).to_numpy(),
    )


def _network_arrays(fx):
    n = len(fx.bus)
    a = incidence(fx.br["From"].to_numpy(np.int64) - 1,
                  fx.br["To"].to_numpy(np.int64) - 1, n)
    b_prim = 1.0 / fx.br["X"].to_numpy(float)
    bbus = (a @ sp.diags(b_prim) @ a.T).tocsc()
    busnum0 = fx.gen["BusNum"].to_numpy(np.int64) - 1
    slack = int(np.flatnonzero(fx.bus["Type"].to_numpy(float) == 3)[0])
    return n, a, b_prim, bbus, busnum0, slack


@needs_dcopf
def test_fixture_internal_consistency(fx):
    """Solver-free convention checks against the Mosek dump: flows =
    SBase·Bprim·Δθ, Δθ = ANetᵀ·Va (incidence orientation), and the balance
    rows hold at the fixture's own dispatch (validates our B-matrix)."""
    n, a, b_prim, bbus, busnum0, _ = _network_arrays(fx)
    # flows/angle-diff identity, every fixture scenario
    assert np.allclose(fx.flows, S_BASE * b_prim[:, None] * fx.angles,
                       atol=1e-6)
    # incidence orientation + balance vs the first-OK scenario's Va
    va = fx.vbuspolar[:, 1]
    assert np.allclose(fx.vbuspolar[:, 0], 1.0)
    assert np.allclose(a.T @ va, fx.angles[:, 0], atol=1e-8)
    k = fx.scenarios[0]
    inj_pu = (np.bincount(busnum0, weights=fx.pucnew[:, 0], minlength=n)
              - np.real(fx.ls[:, k])) / S_BASE
    # 1e-4: Mosek's own primal feasibility residual (measured 4.5e-5 pu)
    assert np.abs(bbus @ va - inj_pu).max() < 1e-4


@needs_dcopf
def test_fixture_fixed_dispatch_reconstruction(fx):
    """Doc 05 check: with dispatch FIXED to MATLAB's PucNew, our network
    machinery reproduces the dumped flows/angles/SNetCalc (linear solve —
    no optimizer in the loop)."""
    n, a, b_prim, bbus, busnum0, slack = _network_arrays(fx)
    keep = np.flatnonzero(np.arange(n) != slack)
    lu = splu(bbus[keep][:, keep].tocsc())
    for i, k in enumerate(fx.scenarios):
        inj_pu = (np.bincount(busnum0, weights=fx.pucnew[:, i], minlength=n)
                  - np.real(fx.ls[:, k])) / S_BASE
        theta = np.zeros(n)
        theta[keep] = lu.solve(inj_pu[keep])
        flows = S_BASE * b_prim * (a.T @ theta)
        # doc 05 atol 1e-4 pu = 1e-2 MW (measured 3.7e-3 MW; the noise is
        # Mosek's interior-point residual amplified through the LU solve)
        assert np.allclose(flows, fx.flows[:, i], atol=1e-2), \
            f"scenario {k}: max flow diff {np.abs(flows - fx.flows[:, i]).max():.3e} MW"
        assert np.allclose(a.T @ theta, fx.angles[:, i], atol=1e-5)
        v_rect = np.exp(1j * theta)
        snet = np.real(S_BASE * v_rect * np.conj(bbus @ v_rect))
        assert np.allclose(snet, fx.snet_real[:, i], atol=1e-2)


@pytest.fixture(scope="module")
def solved(fx):
    """Our gurobipy solves of the three fixture scenarios (one kernel,
    warm-started — the production serial path)."""
    cfg = WeldConfig()
    kernel = build_dcopf(fx.br, fx.gen, fx.bus, cfg)
    return [solve_scenario(kernel, fx.ls[:, k], fx.pg_uc[:, k], cfg,
                           scenario=k)
            for k in fx.scenarios]


@needs_dcopf
def test_fixture_objective(fx, solved):
    """DV1: objective (never argmin). The dump has no objective file, so
    the Mosek objective is EVALUATED from its own dispatch (delp = L1
    deviation at any optimum; theta contributes nothing at λ3 = 0)."""
    a_sum = float(fx.gen["A"].sum())
    b_lin = fx.gen["B"].to_numpy(float)
    c_quad = fx.gen["C"].to_numpy(float)
    for i, res in enumerate(solved):
        pg_pu = fx.pucnew[:, i] / S_BASE
        delp = np.abs(fx.pucnew[:, i] - fx.pg_uc[:, fx.scenarios[i]]) / S_BASE
        obj_mosek = (0.95 * delp.sum()
                     + 0.05 * (pg_pu @ (c_quad * pg_pu) + b_lin @ pg_pu
                               + a_sum))
        assert np.isclose(res.objective, obj_mosek, rtol=1e-6), \
            (f"scenario {fx.scenarios[i]}: gurobi {res.objective:.9f} vs "
             f"mosek {obj_mosek:.9f} "
             f"(rel {abs(res.objective - obj_mosek) / obj_mosek:.2e})")


@needs_dcopf
def test_fixture_unit_dispatch_where_unique(fx, solved):
    """Doc 05's bus-level atol-1e-3-pu check assumed 'aggregation is
    usually non-degenerate' — FALSE for this model: with λ3 = 0 the network
    imposes NO coupling on the objective (θ absorbs any bus distribution),
    so placement among cost-tied units is arbitrary (measured: up to
    ~285 MW moved between C==0 units at identical objective, biggest at the
    slack bus). What IS provably unique at every optimum is pg of each
    C > 0 unit (strictly convex coordinate of a separable objective on a
    convex optimum set) and the deviation/cost term split — compared here.
    Measured 2026-07-18: C>0 max |ΔPg| ≤ 1.1e-5 MW, term split ≤ 8e-8 rel."""
    c_quad = fx.gen["C"].to_numpy(float)
    b_lin = fx.gen["B"].to_numpy(float)
    a_sum = float(fx.gen["A"].sum())
    strict = c_quad > 0
    assert strict.sum() > 1000                     # 1211 of 2229 for CATS
    for i, res in enumerate(solved):
        diff = np.abs(res.pg_mw - fx.pucnew[:, i])
        assert diff[strict].max() <= 1e-3, \
            (f"scenario {fx.scenarios[i]}: max C>0 unit diff "
             f"{diff[strict].max():.3e} MW")
        assert abs(res.pg_mw.sum() - fx.pucnew[:, i].sum()) <= 0.1
        delp_mosek = np.abs(fx.pucnew[:, i]
                            - fx.pg_uc[:, fx.scenarios[i]]).sum() / S_BASE
        pg_pu = fx.pucnew[:, i] / S_BASE
        cost_mosek = pg_pu @ (c_quad * pg_pu) + b_lin @ pg_pu + a_sum
        pg_pu_g = res.pg_mw / S_BASE
        cost_ours = pg_pu_g @ (c_quad * pg_pu_g) + b_lin @ pg_pu_g + a_sum
        assert np.isclose(res.delp_pu.sum(), delp_mosek, rtol=1e-6)
        assert np.isclose(cost_ours, cost_mosek, rtol=1e-6)


# ---------------------------------------------------------------------------
# the S5 loop (fixture-scale; ~4 s a test — not slow-marked)
# ---------------------------------------------------------------------------


@needs_dcopf
def test_upgrade_loop_fixture(fx):
    """Two passes of the v2:427-497 loop on the CATS state (2 scenarios):
    wiring, bookkeeping, PF.Pg/SNetFullOPF contract, iteration log."""
    from gridweld.params.conductors import crucible_catalog, load_cats_catalog
    from gridweld.params.impedance import load_gmd_tables

    cfg = WeldConfig()
    cfg.feasibility.max_workers = 1
    rng = np.random.default_rng(cfg.system.rng_seed)
    n = len(fx.bus)
    # SNetFull (injections) is only consumed by maxscenario/GenCommit and
    # dc_stats here — build it as UCED gen aggregation minus load:
    busnum0 = fx.gen["BusNum"].to_numpy(np.int64) - 1
    snet_full = np.empty((n, 2), dtype=complex)
    for c in range(2):
        snet_full[:, c] = (np.bincount(busnum0, weights=fx.pg_uc[:, c],
                                       minlength=n)
                           - fx.ls[:, c])
    bus, br, updates, log_df = run_dcopf_upgrade(
        fx.bus, fx.br, fx.gen, snet_full, fx.ls[:, :2], fx.pg_uc[:, :2],
        pd.read_parquet(DATA / "xfdata.parquet"),
        pd.read_parquet(DATA / "xfxr.parquet"),
        crucible_catalog(), load_cats_catalog(), load_gmd_tables(),
        cfg, rng, max_passes=2)

    assert list(log_df.columns) == ["iter", "overloads", "total_rate_mva",
                                    "max_dcpf"]
    assert 1 <= len(log_df) <= 2
    assert (br["RateA"].to_numpy(float) > 0).all()
    assert updates["pg"].shape == (len(fx.gen), 2)
    # SNetFullOPF column identity (v2:485-491): gen aggregation − load
    expected0 = (np.bincount(busnum0, weights=updates["pg"][:, 0],
                             minlength=n) - fx.ls[:, 0])
    assert np.allclose(updates["snet_full_opf"][:, 0], expected0)
    assert bus["Vm"].eq(1.0).all()
    assert np.abs(bus["Va"].to_numpy(float)).max() > 0   # radians written
    assert (bus["GenMWSet"].to_numpy(float) > 0).any()
    assert len(updates["dc_stats"]) == 2
    assert (updates["dead_lines"]["kV"].to_numpy(float) >= 100).all()


@needs_dcopf
def test_upgrader_class_meta_contract(fx):
    """DCOPFUpgrader over the provisional WeldModel meta contract."""
    from gridcrucible.model import ScenarioSet

    from gridweld.feasibility.dcopf import DCOPFUpgrader
    from gridweld.model import WeldModel

    cfg = WeldConfig()
    cfg.feasibility.max_workers = 1
    n = len(fx.bus)
    busnum0 = fx.gen["BusNum"].to_numpy(np.int64) - 1
    snet_full = np.empty((n, 1), dtype=complex)
    snet_full[:, 0] = (np.bincount(busnum0, weights=fx.pg_uc[:, 0],
                                   minlength=n) - fx.ls[:, 0])
    model = WeldModel(buses=fx.bus, branches=fx.br, generators=fx.gen,
                      scenarios=ScenarioSet(tiers={"full": snet_full}),
                      meta={"load_scenarios": fx.ls[:, :1],
                            "pg_uc": fx.pg_uc[:, :1]})
    out = DCOPFUpgrader(cfg).upgrade(model, max_passes=1,
                                     compute_stats=False)
    assert out.meta["pg"].shape == (len(fx.gen), 1)
    assert out.meta["snet_full_opf"].shape == (n, 1)
    assert "dcopf_log" in out.meta and "dead_lines" in out.meta


# ---------------------------------------------------------------------------
# assigning_pmax_pmin (S5 prep)
# ---------------------------------------------------------------------------


def test_assign_gen_limits_semantics():
    """Hand-built case covering the order-dependent quirks: multi-unit
    broadcast to BUS mates (marking them done — a bus-mate's OWN EIA row is
    never consulted), the per-row final clamp, case-insensitive
    numeric-vs-string keys, and the fuel-type mean fallback."""
    gens = pd.DataFrame({
        "BusNum": [1, 1, 2],
        "PlantCode": [10, 11, 12],
        "GenID": ["1", "2", "X"],
        "FuelType": ["Gas", "Gas", "Solar"],
        "MaxMW": [50.0, 5.0, 100.0],
        "MinMW": [np.nan, np.nan, np.nan],
        "MinMVar": [0.0, 0.0, 0.0],
        "MaxMVar": [0.0, 0.0, 0.0]})
    eia = pd.DataFrame({
        "State": ["CA", "CA", "CA", "CA"],
        "PlantCode": ["10", "11", "900", "901"],   # str keys vs int gen codes
        "GeneratorID": ["1", "2", "A", "B"],
        "NumberOfUnits": [2.0, 1.0, 1.0, 1.0],
        "PlantMinimumLoad_MW_": [20.0, 3.0, 6.0, 10.0],
        "MinMVar": [8.0, 1.0, 2.0, 4.0],
        "MaxMVar": [16.0, 2.0, 3.0, 5.0],
        "Technology": ["Gas", "Gas", "Solar", "solar"]})
    out = assign_gen_limits(gens, eia)
    # gen0: multi-unit 20/2=10 < 50 → broadcast 10 to bus-1 mates
    assert out.at[0, "MinMW"] == 10.0
    assert out.at[0, "MinMVar"] == 4.0 and out.at[0, "MaxMVar"] == 8.0
    # gen1: bus-mate of gen0 → NEVER matched against its own EIA row
    # (plant 11 says 3.0); broadcast 10 ≥ MaxMW 5 → final clamp zeroes it
    assert out.at[1, "MinMW"] == 0.0
    assert out.at[1, "MinMVar"] == 4.0             # broadcast MVars stay
    # gen2: no plant match → strcmpi fuel-type mean over BOTH Solar rows
    assert out.at[2, "MinMW"] == 8.0               # mean(6, 10)
    assert out.at[2, "MinMVar"] == 3.0 and out.at[2, "MaxMVar"] == 4.0


@needs_dcopf
@needs_eia
def test_assign_gen_limits_reproduces_fixture():
    """The fixture Gen already carries the MATLAB original's output —
    re-applying the port must be a fixed point (rows with neither a plant
    nor a fuel-type match keep their values by construction and therefore
    pass trivially; everything else is exercised for real)."""
    gen = pd.read_csv(FIX / "in_gen.csv", low_memory=False)
    eia = pd.read_excel(LEGACY_EIA)
    out = assign_gen_limits(gen, eia)
    for col in ("MinMW", "MinMVar", "MaxMVar"):
        diff = np.abs(out[col].to_numpy(float) - gen[col].to_numpy(float))
        assert np.nanmax(diff) < 1e-6, \
            (f"{col}: {int((diff > 1e-6).sum())} rows differ, "
             f"max {np.nanmax(diff):.3e} (first at row "
             f"{int(np.nanargmax(diff))})")
