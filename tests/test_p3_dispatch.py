"""P3c validation: bus roles vs s5_pf.mat (exact), the deterministic LOAD
side of SNetFull (exact), and the full ED+CD scenario build vs the s5
fixtures within CPLEX→Gurobi tolerance (DV-P3-6: objectives/commitments,
never per-unit argmins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.dispatch.snet import (build_snet, bus_fleet_aggregates,
                                    bus_roles, sine_load_scenarios)
from gridweld.dispatch.uced import uccd, uced

FIX = REPO_ROOT / "validation" / "fixtures" / "matlab"
needs_matlab = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                  reason="matlab fixtures not extracted")
pytestmark = needs_matlab


@pytest.fixture(scope="module")
def cfg() -> WeldConfig:
    return WeldConfig()


@pytest.fixture(scope="module")
def pf() -> dict:
    return sio.loadmat(FIX / "s5_pf.mat", simplify_cells=True)["PF"]


@pytest.fixture(scope="module")
def s3_bus() -> pd.DataFrame:
    return pd.read_csv(FIX / "s3_bus_post_readmatpower.csv", low_memory=False)


@pytest.fixture(scope="module")
def s3_gen() -> pd.DataFrame:
    g = pd.read_csv(FIX / "s3_gen_post_readmatpower.csv", low_memory=False)
    g["FuelType"] = g["FuelType"].astype(str)
    return g


@pytest.fixture(scope="module")
def snetfull_fix() -> np.ndarray:
    re = pd.read_csv(FIX / "s5_snetfull_real.csv", header=None).to_numpy()
    im = pd.read_csv(FIX / "s5_snetfull_imag.csv", header=None).to_numpy()
    return re + 1j * im


def test_bus_roles_match_pf(cfg, pf, s3_bus, s3_gen):
    """matlab_compat=True: the GenList(2:end) quirk vs s5_pf.mat (DV-P3-12
    — the CORRECTED behavior is the default since the author's 2026-07-19
    ruling; this test pins the legacy fixture semantics)."""
    slack, gen_list, load_list, vset = bus_roles(
        bus_fleet_aggregates(s3_bus, s3_gen), s3_gen, cfg,
        matlab_compat=True)
    assert slack == int(pf["SlackList"]) == 2011
    assert np.array_equal(gen_list, np.asarray(pf["GenList"]).astype(int))
    assert np.array_equal(load_list, np.asarray(pf["LoadList"]).astype(int))
    assert np.allclose(vset, np.asarray(pf["VBusSet"], float))


def test_bus_roles_corrected_default(cfg, s3_bus, s3_gen):
    """The DEFAULT (corrected, DV-P3-12 ruling 2026-07-19): the smallest
    gen bus STAYS in GenList; the TRUE slack is removed from it."""
    slack, gen_list, load_list, _ = bus_roles(
        bus_fleet_aggregates(s3_bus, s3_gen), s3_gen, cfg)
    gen_buses = np.sort(s3_gen["BusNum"].astype(int).unique())
    assert slack == 2011
    assert gen_buses[0] in gen_list           # bus 6 no longer mis-dropped
    assert slack not in gen_list
    assert gen_buses[0] not in load_list


def test_load_side_of_snetfull_exact(cfg, s3_bus, s3_gen, snetfull_fix):
    """Buses with no generators: SNetFull column = −(scaled sine load) —
    fully deterministic, validates the sine profile, maxscenario pick, and
    the guard scaling exactly (incl. the imag part everywhere the fixture
    has no dispatch)."""
    loads = sine_load_scenarios(bus_fleet_aggregates(s3_bus, s3_gen), cfg)
    cap = float(s3_gen["MaxMW"].sum())
    hydro_cap = float(s3_gen.loc[s3_gen["FuelType"]
                                 .str.contains("Hydro", case=False),
                                 "MaxMW"].sum())
    max_scen = int(loads.real.sum(axis=0).argmax())
    peak = float(loads.real[:, max_scen].sum())
    if cap - (2 / 3) * hydro_cap <= 1.01 * peak:
        loads = loads * (cap / (1.01 * peak))
    elif float(s3_gen["MinMW"].sum()) >= 0.99 * peak:
        loads = loads * (1.01 * float(s3_gen["MinMW"].sum()) / peak)

    gen_buses = s3_gen["BusNum"].astype(int).unique()
    no_gen = ~np.isin(np.arange(1, len(s3_bus) + 1), gen_buses)
    n_scen = loads.shape[1]
    for block in range(snetfull_fix.shape[1] // n_scen):
        cols = slice(block * n_scen, (block + 1) * n_scen)
        assert np.allclose(snetfull_fix[no_gen, cols], -loads[no_gen, :],
                           rtol=1e-9, atol=1e-9), f"block {block}"
    # Q side is pure −load EVERYWHERE (dispatch is real-valued)
    assert np.allclose(snetfull_fix.imag[:, :n_scen], -loads.imag,
                       rtol=1e-9, atol=1e-9)


@pytest.fixture(scope="module")
def built_snet(cfg, s3_bus, s3_gen):
    # bus_roles_compat=True: tier-signature rows follow the legacy quirk
    # the CPLEX-era fixtures embody (DV-P3-12 corrected-default ruling)
    return build_snet(s3_bus, s3_gen, cfg, bus_roles_compat=True)


def test_snet_energy_balance(built_snet, snetfull_fix):
    full = built_snet.tiers["full"]
    assert full.shape == snetfull_fix.shape == (8822, 48)
    col_sums = np.abs(full.real.sum(axis=0))
    assert (col_sums < 1e-3 * np.abs(full.real).sum(axis=0)).all()
    fix_sums = np.abs(snetfull_fix.real.sum(axis=0))
    assert (fix_sums < 1e-3 * np.abs(snetfull_fix.real).sum(axis=0)).all()


def test_snet_commitments_close_to_fixture(built_snet, snetfull_fix, pf):
    """Committed-plant signatures at [slack; GenList] rows vs the CPLEX
    fixture (DV-P3-6: many units share IDENTICAL donor cost curves, so the
    3-at-a-time decommit ranking is tie-ridden and trajectories diverge).

    Thresholds calibrated from the 2026-07-17 measurement run:
    Jaccard min/mean/max = 0.754/0.958/1.0; committed-count rel diff
    ≤ 0.047; median gen-row |ΔP| = 0.0 MW."""
    gen_rows = np.concatenate([[int(pf["SlackList"])],
                               np.asarray(pf["GenList"]).astype(int)]) - 1
    mine = built_snet.tiers["full"].real[gen_rows, :] > 0.1
    fix = snetfull_fix.real[gen_rows, :] > 0.1
    jac = np.array([
        (mine[:, i] & fix[:, i]).sum() / max((mine[:, i] | fix[:, i]).sum(), 1)
        for i in range(mine.shape[1])])
    assert jac.min() > 0.70, f"worst column Jaccard {jac.min():.3f}"
    assert jac.mean() > 0.93, f"mean Jaccard {jac.mean():.3f}"
    count_diff = np.abs(mine.sum(0) - fix.sum(0)) / fix.sum(0)
    assert count_diff.max() < 0.10


def test_snet_tier_widths_close(built_snet, pf):
    """Tier widths vs the CPLEX run (36/32/31/28 from 48) — commitment
    signatures shift a little under Gurobi ties, so ±4 columns."""
    expected = {"complete": np.asarray(pf["SNetComplete"]).shape[1],
                "advanced": np.asarray(pf["SNetAdvanced"]).shape[1],
                "intermediate": np.asarray(pf["SNetIntermediate"]).shape[1],
                "basic": np.asarray(pf["SNetBasic"]).shape[1]}
    for tier, want in expected.items():
        got = built_snet.tiers[tier].shape[1]
        assert abs(got - want) <= 4, (tier, got, want)
    widths = [built_snet.tiers[t].shape[1]
              for t in ("complete", "advanced", "intermediate", "basic")]
    assert widths == sorted(widths, reverse=True)


def test_uced_uccd_anticorrelate(cfg, s3_gen):
    """The CD dispatch must stress the fleet oppositely to ED (doc 03
    validation property): rank correlation of dispatches < 0 on shared
    committed units."""
    total = 0.5 * float(s3_gen["MaxMW"].sum())
    x_ed, f_ed = uced(s3_gen, 1.0, total, cfg.dispatch.reserve)
    x_cd, f_cd = uccd(s3_gen, 1.0, np.array([total]), cfg.dispatch.reserve)
    # doc 03: anti-correlation ACROSS THE FLEET (uncommitted units count 0)
    ed = pd.Series(0.0, index=s3_gen.index)
    ed.loc[f_ed.index] = x_ed
    cd = pd.Series(0.0, index=s3_gen.index)
    cd.loc[f_cd.index] = x_cd
    rho = np.corrcoef(ed.rank(), cd.rank())[0, 1]
    assert rho < 0, f"ED/CD dispatches should anti-correlate (rho={rho:.2f})"
    # and CD must run FEWER, more expensive units than ED
    assert len(f_cd) < len(f_ed)
    mean_b_ed = float(f_ed["B"].mean())
    mean_b_cd = float(f_cd["B"].mean())
    assert mean_b_cd > mean_b_ed
