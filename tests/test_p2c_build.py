"""P2c validation: the S2 build stage vs the julia dump fixtures, phase by
phase (doc 02 / D9a). Deterministic phases compare exactly (coords 1e-12,
costs 1e-9, snap ids exact / distances 1e-9); the MILP compares on the
OBJECTIVE (degenerate optima — doc 02); jittered coordinates compare within
the jitter amplitude (DV3); split-bus ids compare topologically (DV-P2c-2).

⚠ DV6/Q5: load pd values inherit the "P load is INCORRECT" defect — these
tests validate the CONVERSION against the current fixtures; the fixtures
themselves are not frozen until the author rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridweld.build import run_build
from gridweld.build.costs import assign_costs
from gridweld.build.generators import load_generators
from gridweld.build.loads import load_loads
from gridweld.build.reduce import clean_transmission
from gridweld.build.substations import load_added_nodes, load_substations
from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.viz.metrics import load_metrics

FIX = REPO_ROOT / "validation" / "fixtures" / "julia"
needs_julia = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                 reason="julia fixtures not extracted")

pytestmark = needs_julia


@pytest.fixture(scope="module")
def cfg() -> WeldConfig:
    return WeldConfig()


@pytest.fixture(scope="module")
def built(cfg):
    """One full S2 run shared by the phase tests (exact reduction solver —
    DV-P2c-4; the gurobipy MILP path is unit-tested separately)."""
    rng = np.random.default_rng(cfg.system.rng_seed)
    return run_build(cfg, rng, solver="assignment")


# ---------------------------------------------------------------------------
# Phase 1 — read + clean
# ---------------------------------------------------------------------------

def test_p1_substations(cfg):
    subs = load_substations(cfg.paths.resolve(cfg.paths.substations))
    fix = pd.read_csv(FIX / "p1_substations_clean.csv")
    assert len(subs) == len(fix) == 3249
    assert np.array_equal(subs["index"], fix["index"])
    assert np.allclose(subs["Lat"], fix["Lat"]) and np.allclose(subs["Lon"], fix["Lon"])
    assert (subs["Type"] == "Substation").all()
    assert (subs["Import"].astype(str) == fix["Import"].astype(str)).all()


def test_p1_added_nodes(cfg):
    nodes = load_added_nodes(cfg.paths.resolve(cfg.paths.added_nodes), 3249)
    fix = pd.read_csv(FIX / "p1_new_nodes_clean.csv")
    assert len(nodes) == len(fix) == 5727
    assert np.array_equal(nodes["index"], fix["index"])
    assert np.allclose(nodes["Lat"], fix["Lat"]) and np.allclose(nodes["Lon"], fix["Lon"])


def test_p1_generators_with_costs(cfg):
    gens = load_generators(cfg.paths.resolve(cfg.paths.eia_generators),
                           cfg.paths.resolve(cfg.paths.eia_plants), cfg)
    gens = assign_costs(gens, cfg.paths.resolve(cfg.paths.cost_curves), cfg)
    fix = pd.read_csv(FIX / "p1_generators_clean.csv")
    assert len(gens) == len(fix) == 2551
    # row order replicates the DataFrames.jl matched-first join layout
    assert np.array_equal(gens["index"], fix["index"])
    assert (gens["FuelType"] == fix["FuelType"]).all()
    assert np.allclose(gens["pmax"], fix["pmax"])
    assert np.allclose(gens["Lat"], fix["Lat"], rtol=1e-12)
    for py, jl in (("c2", "cost_c2"), ("c1", "cost_c1"), ("c0", "cost_c0")):
        assert np.allclose(gens[py], fix[jl], rtol=1e-9), (py, jl)


def test_p1_loads_legacy_parser():
    """load_loads reproduces the legacy Julia read EXACTLY (the fixture
    embodies the legacy apportionment file — this pins the parser even now
    that the default is the corrected file, DV6/D-1)."""
    legacy = REPO_ROOT / "legacy" / "Data" / "California_load_approtionment.csv"
    loads = load_loads(legacy)
    fix = pd.read_csv(FIX / "p1_loads_clean.csv")
    assert len(loads) == len(fix) == 8034
    assert np.array_equal(loads["index"], fix["index"])
    assert np.allclose(loads["pd"], fix["pd"]) and np.allclose(loads["qd"], fix["qd"])


def test_p1_loads_default_is_corrected(cfg):
    """DV6/D-1 default flip (2026-07-20): the shipped default apportionment
    is the corrected loadsmith file (Σ pd = 47,257 MW peak anchor). Its
    index/Lat/Lon are IDENTICAL to the legacy file, so the MILP assignment
    and all downstream topology are invariant — only pd/qd magnitudes move."""
    corr = load_loads(cfg.paths.resolve(cfg.paths.load_apportionment))
    legacy = load_loads(REPO_ROOT / "legacy" / "Data"
                        / "California_load_approtionment.csv")
    assert len(corr) == 8034
    assert np.array_equal(corr["index"], legacy["index"])
    assert np.allclose(corr["Lat"], legacy["Lat"])
    assert np.allclose(corr["Lon"], legacy["Lon"])
    assert corr["pd"].sum() == pytest.approx(47257.0, rel=1e-6)
    assert not np.allclose(corr["pd"], legacy["pd"])       # magnitudes moved


def test_p1_transmission(cfg):
    lines, tiny = clean_transmission(cfg.paths.resolve(cfg.paths.transmission), cfg)
    fix = pd.read_csv(FIX / "p1_transmission_clean.csv")
    assert len(tiny) == 0                       # p1_tiny_lines.csv is empty
    assert len(lines) == len(fix) == 10398
    assert np.array_equal(lines["index"], fix["index"])
    assert np.array_equal(lines["FID"], fix["FID"])
    assert np.allclose(lines["kV"], fix["kV"])   # remap, ties → lowest
    assert np.allclose(lines["Lat1"], fix["Lat1"], rtol=1e-12)


# ---------------------------------------------------------------------------
# Phases 2-3 — snapping + import generators
# ---------------------------------------------------------------------------

def _by_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("index", ignore_index=True)


def test_p2_generator_snap(cfg):
    """Recomputed at the phase-2 point (substations only — the p2 fixture
    is PRE-node/PRE-split; run_build's final gen table has post-split ids)."""
    from gridweld.build.snapping import snap_points
    from gridweld.build.substations import make_bus_table
    gens = load_generators(cfg.paths.resolve(cfg.paths.eia_generators),
                           cfg.paths.resolve(cfg.paths.eia_plants), cfg)
    gens = assign_costs(gens, cfg.paths.resolve(cfg.paths.cost_curves), cfg)
    subs = load_substations(cfg.paths.resolve(cfg.paths.substations))
    snapped = snap_points(gens, make_bus_table(subs), "gen",
                          warn_km=cfg.snapping.warn_km,
                          earth_radius_km=cfg.geo.earth_radius_km)
    fix = _by_index(pd.read_csv(FIX / "p2_post_assign_generators_gen.csv"))
    mine = _by_index(snapped)
    assert len(mine) == len(fix) == 2551
    assert np.array_equal(mine["gen_bus"], fix["gen_bus"])
    assert np.allclose(mine["gen_bus_distance"], fix["gen_bus_distance"],
                       rtol=1e-9)


def test_p3_branch_snap(cfg):
    """Recomputed at the phase-3 point (pre-split endpoint ids).

    ⚠ 65 endpoints sit at EXACT coordinate ties (co-located substation +
    added-node pairs, identical raw CSV strings — verified 2026-07-16).
    Julia's Dict-hash tie-breaks are unreproducible (the dump manifest says
    so itself); ours is sorted-id (DV-P2c-2). Ids must match EXCEPT at
    provable ties, where the distances must agree instead."""
    from gridweld.build.snapping import snap_branch_endpoints
    from gridweld.build.substations import make_bus_table
    subs = load_substations(cfg.paths.resolve(cfg.paths.substations))
    nodes = load_added_nodes(cfg.paths.resolve(cfg.paths.added_nodes), len(subs))
    lines, _ = clean_transmission(cfg.paths.resolve(cfg.paths.transmission), cfg)
    bus = make_bus_table(subs, nodes)
    snapped = snap_branch_endpoints(lines, bus,
                                    warn_km=cfg.snapping.warn_km,
                                    earth_radius_km=cfg.geo.earth_radius_km)
    fix = _by_index(pd.read_csv(FIX / "p3_post_assign_branches_branch.csv"))
    mine = _by_index(snapped)
    assert len(mine) == len(fix) == 10398
    assert np.array_equal(mine["FID"], fix["FID"])
    pos = bus.set_index("index")
    for end, latc, lonc in (("f_bus", "Lat1", "Lon1"), ("t_bus", "Lat2", "Lon2")):
        m = mine[end].to_numpy()
        f = fix[end].to_numpy()
        # atol 1e-11 km = 10 nm: Julia's atan(√a,√(1−a)) vs arctan2 differ
        # by a few ulp on sub-metre distances
        assert np.allclose(mine[f"{end}_distance"], fix[f"{end}_distance"],
                           rtol=1e-9, atol=1e-11)
        bad = np.flatnonzero(m != f)
        assert len(bad) <= 80, f"{end}: too many mismatches ({len(bad)})"
        for i in bad:      # every mismatch must be a genuine coordinate tie
            assert pos.at[m[i], "Lat"] == pos.at[f[i], "Lat"], (end, i)
            assert pos.at[m[i], "Lon"] == pos.at[f[i], "Lon"], (end, i)
    agree = (mine["f_bus"].to_numpy() == fix["f_bus"].to_numpy()).mean()
    assert agree > 0.995


def test_p3_import_generators(built, cfg):
    full, _, _ = built
    fix = _by_index(pd.read_csv(FIX / "p3_post_assign_branches_gen.csv"))
    mine = _by_index(full.gen)
    assert len(mine) == len(fix) == 2582
    imports = mine["FuelType"] == "IMPORT"
    fimports = fix["FuelType"] == "IMPORT"
    assert imports.sum() == fimports.sum() == 31
    assert np.array_equal(mine.loc[imports, "index"], fix.loc[fimports, "index"])
    assert np.allclose(mine.loc[imports, "pmax"], cfg.costs.import_pmax_mw)
    fix_cost = fix.loc[fimports, "cost"].map(
        lambda s: [float(x) for x in s.strip("[]").split(",")]).tolist()
    assert np.allclose(np.asarray(fix_cost)[:, 1],
                       cfg.costs.import_linear_usd_per_mwh)
    assert np.allclose(mine.loc[imports, "c1"],
                       cfg.costs.import_linear_usd_per_mwh)


def test_p3_bus_count(built):
    full, _, _ = built
    fix = pd.read_csv(FIX / "p3_post_assign_branches_bus.csv")
    assert len(fix) == 8976                      # 3249 subs + 5727 nodes


# ---------------------------------------------------------------------------
# Fixture-graph replay: the 65 coordinate-tie endpoints make my snapped
# graph differ microscopically from the julia run's (DV-P2c-2), which
# shifts leaves/components. Downstream logic is validated EXACTLY by
# replaying it on the FIXTURE's own graph (p3 endpoints + p4 loads).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_frames(cfg):
    subs = load_substations(cfg.paths.resolve(cfg.paths.substations))
    nodes = load_added_nodes(cfg.paths.resolve(cfg.paths.added_nodes), len(subs))
    from gridweld.build.substations import make_bus_table
    bus = make_bus_table(subs, nodes)
    branch = _by_index(pd.read_csv(FIX / "p3_post_assign_branches_branch.csv"))
    gen = _by_index(pd.read_csv(FIX / "p3_post_assign_branches_gen.csv"))
    load = _by_index(pd.read_csv(FIX / "p4_post_assign_loads_load.csv"))
    return bus, branch, gen, load


def test_p4_milp_objective_on_fixture_graph(cfg, fixture_frames):
    """Leaf detection + the exact reduction (DV-P2c-4) on the fixture's own
    graph must reproduce the fixture MILP objective to 1e-6."""
    from gridweld.build.loads import assign_loads
    bus, branch, gen, load = fixture_frames
    fresh = load.drop(columns=["load_bus", "load_bus_distance"])
    assigned, milp = assign_loads(fresh, bus, branch, gen, cfg,
                                  solver="assignment")
    fixture_objective = load["load_bus_distance"].sum()
    assert milp.objective_km == pytest.approx(fixture_objective, rel=1e-6)
    assert milp.gap == 0.0


def test_p4_own_run_valid(built):
    full, _, artifacts = built
    assert full.load["load_bus"].notna().all()
    assert (full.load["load_bus_distance"] >= 0).all()
    # own-graph objective sits within a hair of the fixture's (tie noise)
    fix = pd.read_csv(FIX / "p4_post_assign_loads_load.csv")
    assert artifacts["milp"].objective_km == pytest.approx(
        fix["load_bus_distance"].sum(), rel=0.01)


@pytest.fixture(scope="module")
def replayed(cfg, fixture_frames):
    """Downstream stages (transformers → finalize → self-loops → largest)
    on the fixture graph — deterministic, so p5 must match EXACTLY."""
    from gridweld.build.reduce import (finalize_frames, largest_component,
                                       remove_self_loops)
    from gridweld.build.transformers import create_transformers
    bus, branch, gen, load = fixture_frames
    rng = np.random.default_rng(cfg.system.rng_seed)
    bus, branch, gen, load = create_transformers(bus, branch, gen, load,
                                                 cfg, rng)
    finalize_frames(bus, branch, gen, load)
    branch, self_loops = remove_self_loops(branch)
    lb, lbr, lg, ll = largest_component(bus, branch, gen, load)
    return (bus, branch, gen, load), (lb, lbr, lg, ll), self_loops


def test_p5_full_grid_counts(replayed):
    (bus, branch, gen, load), _, _ = replayed
    assert len(bus) == 9647
    assert len(branch) == 11057
    assert len(gen) == 2582
    assert len(load) == 8034


def test_p5_self_loops(replayed):
    _, _, self_loops = replayed
    fix = pd.read_csv(FIX / "p5_self_loops.csv")
    assert set(self_loops["FID"]) == set(fix["FID"])
    assert len(self_loops) == 12


def test_p5_transformer_topology(replayed):
    (bus, branch, _, _), _, _ = replayed
    fix = pd.read_csv(FIX / "p5_final_grid_branch.csv")
    fix_xf = fix[fix["transformer"] == True]  # noqa: E712
    mine_xf = branch[branch["transformer"]]
    assert len(mine_xf) == len(fix_xf)
    kv = bus.set_index("index")["base_kv"]
    pairs = sorted(zip(kv.loc[mine_xf["f_bus"]].to_numpy(),
                       kv.loc[mine_xf["t_bus"]].to_numpy()))
    fkv = pd.read_csv(FIX / "p5_final_grid_bus.csv").set_index("index")["base_kv"]
    fpairs = sorted(zip(fkv.loc[fix_xf["f_bus"]].to_numpy(),
                        fkv.loc[fix_xf["t_bus"]].to_numpy()))
    assert pairs == fpairs
    assert all(hv > lv for hv, lv in pairs)      # f_bus = higher kV


def test_p5_bus_kv_multiset(replayed):
    (bus, _, _, _), _, _ = replayed
    fix = pd.read_csv(FIX / "p5_final_grid_bus.csv")
    mine = bus.groupby(["Type", "base_kv"]).size().sort_index()
    theirs = fix.groupby(["Type", "base_kv"]).size().sort_index()
    assert mine.equals(theirs)


def test_p5_largest_grid_exact_on_replay(replayed):
    _, (lb, lbr, lg, ll), _ = replayed
    assert len(lb) == 8870
    assert len(lbr) == 10823
    assert len(lg) == 2149
    assert len(ll) == 7315


def test_p2c_renormalize_loads_conserves_totals():
    """DV-P2c-5 (D-6): renormalize_loads scales kept pd/qd to the pre-cut
    apportionment totals exactly, preserving relative shares, without
    mutating the input or dropping rows."""
    from gridweld.build.reduce import renormalize_loads
    kept = pd.DataFrame({"pd": [10.0, 30.0, 60.0], "qd": [4.0, 6.0, 10.0]})
    out = renormalize_loads(kept, target_pd=200.0, target_qd=40.0)
    assert out["pd"].sum() == pytest.approx(200.0)          # exact conservation
    assert out["qd"].sum() == pytest.approx(40.0)
    assert np.allclose(out["pd"], [20.0, 60.0, 120.0])      # pd ×2 (shares kept)
    assert np.allclose(out["qd"], [8.0, 12.0, 20.0])        # qd ×2
    assert kept["pd"].sum() == 100.0 and len(out) == 3      # input untouched


def test_p2c_renormalize_conserves_apportionment_on_replay(replayed):
    """D-6 conservation convention on the real replayed graph: the
    largest-component cut drops load, and renormalize_loads lifts the kept
    totals back to the full pre-cut apportionment."""
    from gridweld.build.reduce import renormalize_loads
    (_, _, _, load_full), (_, _, _, ll), _ = replayed
    target_pd = float(load_full["pd"].sum())
    target_qd = float(load_full["qd"].sum())
    assert float(ll["pd"].sum()) < target_pd               # cut dropped load
    out = renormalize_loads(ll, target_pd, target_qd)
    assert out["pd"].sum() == pytest.approx(target_pd)
    assert out["qd"].sum() == pytest.approx(target_qd)
    assert len(out) == len(ll)                             # scales, never drops


def test_p5_own_run_close(built):
    """End-to-end own-graph run: counts within tie-break noise of the
    fixture run (the graph differs at 65 tied endpoints)."""
    full, largest, _ = built
    assert abs(len(full.bus) - 9647) <= 60
    assert abs(len(full.branch) - 11057) <= 60
    assert len(full.gen) == 2582 and len(full.load) == 8034
    assert abs(len(largest.bus) - 8870) <= 80
    assert abs(len(largest.gen) - 2149) <= 40
    assert abs(len(largest.load) - 7315) <= 80


def test_load_metrics_runs(built):
    full, _, _ = built
    df = load_metrics(full)
    assert len(df) == 8
    assert (df["all"] >= df["substation"]).all()


# ---------------------------------------------------------------------------
# gurobipy MILP path (tiny synthetic case — free size-limited license)
# ---------------------------------------------------------------------------

def test_gurobi_milp_matches_reduction_on_toy():
    gp = pytest.importorskip("gurobipy")
    from gridweld.build.loads import _solve_gurobi, _solve_reduction
    rng = np.random.default_rng(7)
    n_load, n_avail = 12, 6
    cost = rng.random((n_load, n_avail)) * 100
    available = np.arange(1, n_avail + 1)
    required = np.array([2, 5])
    a_assign, a_obj = _solve_reduction(cost, available, required)
    try:
        g_assign, g_obj, _ = _solve_gurobi(cost, available, required)
    except gp.GurobiError as e:
        pytest.skip(f"gurobi unavailable: {e}")
    assert g_obj == pytest.approx(a_obj, rel=1e-9)
    req_cols = np.searchsorted(available, required)
    for cols in (a_assign, g_assign):
        assert set(req_cols) <= set(cols.tolist())
