"""P2b validation: incidence/B-matrix vs s4 triplet fixtures, RXB vs the
s2 branch fixtures (rtol 1e-9), the DC stack vs the deterministic
iteration-1 MaxDCPF envelope, and AC kernel self-consistency.

The iteration-1 envelope is deterministic: it is computed from the branch
table exactly as it leaves SizeTransformers (s6 fixture) and the full
SNetFull scenario set (s5 fixtures) BEFORE any randperm resize touches the
table — the first stochastic draw happens after the dump of
s7_maxdcpf_iter01.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.model import SingularNetworkError, WeldModel
from gridweld.params.conductors import load_cats_catalog
from gridweld.params.impedance import compute_line_impedances, load_gmd_tables
from gridweld.power.ac import ACEngine, check_connected
from gridweld.power.dc import dc_envelope, dc_stats, solve_dc, solve_dc_scenarios
from gridweld.power.ybus import b_matrix, incidence, ybus

FIX = REPO_ROOT / "validation" / "fixtures" / "matlab"
needs_matlab = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                  reason="matlab fixtures not extracted")

BRANCH_RENAME = {
    "FID": "fid", "kVFrom": "voltage_kv_from", "kVTo": "voltage_kv_to",
    "Length": "length_mi", "Circuits": "n_circuits",
    "CondPerPhase": "n_cond_per_phase", "RateA": "capacity_mva",
    "GMD": "gmd_ft", "R": "r_pu", "X": "x_pu", "B": "b_pu",
    "CondIndex": "conductor_idx", "From": "from_bus", "To": "to_bus",
    "Tap": "tap",
}


def _load_branch_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, low_memory=False).rename(columns=BRANCH_RENAME)


@pytest.fixture(scope="module")
def s3_branches() -> pd.DataFrame:
    return _load_branch_fixture("s3_branch_post_readmatpower.csv")


@pytest.fixture(scope="module")
def s6_model() -> WeldModel:
    br = _load_branch_fixture("s6_branch_post_sizetransformers.csv")
    branches = pd.DataFrame({
        "from_bus": br["from_bus"].to_numpy(np.int64) - 1,
        "to_bus": br["to_bus"].to_numpy(np.int64) - 1,
        "tap": br["tap"],
        "r_pu": br["r_pu"], "x_pu": br["x_pu"], "b_pu": br["b_pu"],
        "capacity_mva": br["capacity_mva"],
    })
    buses = pd.DataFrame({"bus_id": np.arange(8822, dtype=np.int64)})
    return WeldModel(buses=buses, branches=branches)


@pytest.fixture(scope="module")
def snetfull() -> np.ndarray:
    re = pd.read_csv(FIX / "s5_snetfull_real.csv", header=None).to_numpy()
    im = pd.read_csv(FIX / "s5_snetfull_imag.csv", header=None).to_numpy()
    assert re.shape == (8822, 48)
    return re + 1j * im


# ---------------------------------------------------------------------------
# Incidence + B matrix vs MATLAB triplets
# ---------------------------------------------------------------------------

@needs_matlab
def test_incidence_matches_anet_triplets(s3_branches):
    trip = pd.read_csv(FIX / "s4_anet_triplets.csv", header=None,
                       names=["i", "j", "v"])
    a_ml = sp.coo_matrix((trip["v"], (trip["i"] - 1, trip["j"] - 1)),
                         shape=(8822, 10854)).tocsc()
    a_py = incidence(s3_branches["from_bus"].to_numpy(np.int64) - 1,
                     s3_branches["to_bus"].to_numpy(np.int64) - 1, 8822)
    assert (a_py != a_ml).nnz == 0


@needs_matlab
def test_b_matrix_matches_bbus_triplets(s3_branches):
    trip = pd.read_csv(FIX / "s4_bbus_triplets.csv", header=None,
                       names=["i", "j", "v"])
    b_ml = sp.coo_matrix((trip["v"], (trip["i"] - 1, trip["j"] - 1)),
                         shape=(8822, 8822)).tocsc()
    b_py = b_matrix(s3_branches["from_bus"].to_numpy(np.int64) - 1,
                    s3_branches["to_bus"].to_numpy(np.int64) - 1,
                    s3_branches["x_pu"].to_numpy(float), 8822)
    diff = abs(b_py - b_ml).max()
    assert diff <= 1e-12 * abs(b_ml).max()


# ---------------------------------------------------------------------------
# RXB vs s2 fixture
# ---------------------------------------------------------------------------

@needs_matlab
def test_rxb_matches_matlab(s3_branches):
    fix = _load_branch_fixture("s2_branch_post_rxb.csv")
    cfg = WeldConfig()
    catalog = load_cats_catalog()
    gmd = load_gmd_tables()
    out = compute_line_impedances(fix, catalog, gmd, cfg, matlab_compat=True)
    for col in ("r_pu", "x_pu", "b_pu"):
        assert np.allclose(out[col], fix[col], rtol=1e-9), col
    # fixture RateA is already post-cap → the cap must be idempotent
    assert np.allclose(out["capacity_mva"], fix["capacity_mva"], rtol=1e-9)


@needs_matlab
def test_rxb_fixed_mode_multiplies_b(s3_branches):
    fix = _load_branch_fixture("s2_branch_post_rxb.csv")
    cfg = WeldConfig()
    out = compute_line_impedances(fix, load_cats_catalog(), load_gmd_tables(),
                                  cfg, matlab_compat=False)
    circ = fix["n_circuits"].to_numpy(float)
    multi = circ > 1
    assert multi.any()
    # DV-P3-2: fixed mode = compat mode × circuits²
    assert np.allclose(out.loc[multi, "b_pu"],
                       fix.loc[multi, "b_pu"] * circ[multi] ** 2, rtol=1e-9)


# ---------------------------------------------------------------------------
# DC stack vs deterministic iteration-1 envelope
# ---------------------------------------------------------------------------

@needs_matlab
def test_dc_envelope_matches_iter01(s6_model, snetfull):
    max_dcpf, _ = dc_envelope(s6_model, snetfull, s_base=100.0)
    fix = pd.read_csv(FIX / "s7_maxdcpf_iter01.csv", header=None)[0].to_numpy()
    assert max_dcpf.shape == fix.shape == (10854,)
    assert np.allclose(max_dcpf, fix, rtol=1e-8, atol=1e-8)


@needs_matlab
def test_dc_scenarios_share_factorization(s6_model, snetfull):
    results = solve_dc_scenarios(s6_model, snetfull[:, :3])
    stats = dc_stats(s6_model, results, snetfull[:, :3])
    assert len(stats) == 3
    assert (stats["losses_mw"] > 0).all()
    assert (stats["gen_mw"] > 0).all() and (stats["load_mw"] > 0).all()


def test_dc_exact_angles_small_case():
    """2-bus sanity: θ = X·P exactly; angles are EXACT degrees (Q9), not
    the arctan-compressed gridforge default."""
    model = WeldModel(
        buses=pd.DataFrame({"bus_id": [0, 1]}),
        branches=pd.DataFrame({"from_bus": [0], "to_bus": [1], "tap": [0.0],
                               "r_pu": [0.01], "x_pu": [0.5], "b_pu": [0.0]}),
    )
    res = solve_dc(model, np.array([-100.0, 100.0]), s_base=100.0)
    theta1 = 0.5 * 1.0                       # B=1/X=2 → θ = P/B = 1·0.5 rad
    assert res.v_angle_deg[1] == pytest.approx(np.degrees(theta1), rel=1e-12)
    assert res.flow_mva[0] == pytest.approx(-100.0, rel=1e-12)
    assert res.angle_diff_deg[0] == pytest.approx(np.degrees(theta1), rel=1e-12)


def test_dc_disconnected_raises():
    model = WeldModel(
        buses=pd.DataFrame({"bus_id": [0, 1, 2, 3]}),
        branches=pd.DataFrame({"from_bus": [0, 2], "to_bus": [1, 3],
                               "tap": [0.0, 0.0], "r_pu": [0.01, 0.01],
                               "x_pu": [0.1, 0.1], "b_pu": [0.0, 0.0]}),
    )
    with pytest.raises(SingularNetworkError):
        solve_dc(model, np.array([1.0, -1.0, 1.0, -1.0]))
    with pytest.raises(SingularNetworkError):
        check_connected(model)


# ---------------------------------------------------------------------------
# AC kernel: equation residuals + gridforge equivalence
# ---------------------------------------------------------------------------

def _toy_model() -> WeldModel:
    branches = pd.DataFrame({
        "from_bus": [0, 0, 1, 2], "to_bus": [1, 2, 2, 3],
        "tap": [0.0] * 4,
        "r_pu": [0.01, 0.02, 0.015, 0.01],
        "x_pu": [0.06, 0.11, 0.09, 0.07],
        "b_pu": [0.03, 0.05, 0.04, 0.02],
    })
    buses = pd.DataFrame({"bus_id": range(4)})
    gens = pd.DataFrame({"gen_id": [0, 1], "bus": [0, 1]})
    return WeldModel(buses=buses, branches=branches, generators=gens,
                     meta={"slack_bus": 0})


def test_ac_engine_satisfies_power_flow_equations():
    model = _toy_model()
    snet = np.array([0 + 0j, 40 + 0j, -80 - 30j, -50 - 20j])   # MW+jMVar
    vset = np.array([1.02, 1.01, 1.0, 1.0])
    eng = ACEngine(model, s_base=100.0)
    res = eng.solve(snet, slack=0, pv=np.array([1]),
                    pq=np.array([2, 3]), v_setpoints=vset)
    assert res.converged and res.iterations < 10
    assert res.v_mag_pu[0] == pytest.approx(1.02)
    assert res.v_mag_pu[1] == pytest.approx(1.01)
    # solved injections must match specification at PV (P) and PQ (P&Q) buses
    calc = res.snet_calc_mva
    assert calc[1].real == pytest.approx(40.0, abs=1e-5)
    assert calc[2] == pytest.approx(-80 - 30j, abs=1e-5)
    assert calc[3] == pytest.approx(-50 - 20j, abs=1e-5)
    assert res.losses_mw > 0


def test_ac_engine_matches_gridforge_kernel():
    """Same kernel pieces, same typing → identical solution to gridforge's
    ACPowerFlow (which derives pv/pq from generators + meta)."""
    from gridforge.scale.power_flow import ACPowerFlow
    model = _toy_model()
    snet = np.array([0 + 0j, 40 + 0j, -80 - 30j, -50 - 20j])
    vset = np.ones(4)
    ours = ACEngine(model, s_base=100.0).solve(
        snet, slack=0, pv=np.array([1]), pq=np.array([2, 3]),
        v_setpoints=vset)
    # gridforge wants LOAD-convention pu injections
    theirs = ACPowerFlow(model, s_base=100.0).solve(-snet / 100.0, vset)
    assert theirs.converged
    assert np.allclose(ours.v_mag_pu, theirs.voltages_pu, atol=1e-10)
    assert np.allclose(ours.v_angle_deg, theirs.angles_deg, atol=1e-10)
    assert np.allclose(ours.flow_mva, theirs.flows_mva, atol=1e-8)


def test_ac_engine_warm_start_and_cache():
    model = _toy_model()
    snet = np.array([0 + 0j, 40 + 0j, -80 - 30j, -50 - 20j])
    eng = ACEngine(model)
    r1 = eng.solve(snet, 0, np.array([1]), np.array([2, 3]))
    v0 = r1.v_mag_pu * np.exp(1j * np.radians(r1.v_angle_deg))
    r2 = eng.solve(snet, 0, np.array([1]), np.array([2, 3]), v0=v0)
    assert r2.iterations <= r1.iterations
    assert np.allclose(r1.v_mag_pu, r2.v_mag_pu, atol=1e-9)


def test_ac_engine_rejects_bad_partition():
    model = _toy_model()
    eng = ACEngine(model)
    with pytest.raises(ValueError, match="partition"):
        eng.solve(np.zeros(4, complex), 0, np.array([1]), np.array([2]))


# ---------------------------------------------------------------------------
# ybus wrapper spot check
# ---------------------------------------------------------------------------

def test_ybus_long_line_shunts():
    ys = ybus(np.array([0]), np.array([1]), np.array([0.01]),
              np.array([0.1]), np.array([0.04]), 2)
    d = (ys.b_long - ys.b_short).diagonal()
    assert np.allclose(d, [0.02, 0.02])       # B/2 at each terminal
    assert ys.y_prim.shape == (1, 1)
