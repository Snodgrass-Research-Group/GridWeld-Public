"""P3d validation: circuit stepping (deterministic unit tests), the DC
resize loop vs the s7 iteration log (STATISTICAL — randperm draws differ by
construction, DV3/D9b), and unit/property tests for the AC-side pieces
(no MATLAB fixtures exist: the dump's AC stages crashed upstream)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.resize.dc_loop import run_dc_resize, step_circuits
from gridweld.resize.reactive import (correct_power_factors,
                                      duplicate_max_scenario)
from gridweld.viz.metrics import network_statistics

FIX = REPO_ROOT / "validation" / "fixtures" / "matlab"
DATA = REPO_ROOT / "src" / "gridweld" / "data"
needs_matlab = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                  reason="matlab fixtures not extracted")


def test_step_circuits_semantics():
    """ResizeBranches.m:44-52: rebase per circuit, ±1, clamp, restore."""
    view = pd.DataFrame({
        "overloaded": [True, False, False, True],
        "undersized": [False, True, False, False],
        "capacity_mva": [300.0, 400.0, 500.0, 800.0],
        "n_circuits": [2.0, 2.0, 3.0, 8.0],
        "resize": [False, False, False, False],
    })
    n = step_circuits(view)
    assert n == 3
    # row0: 300/2=150 → circuits 3 → 450
    assert view.at[0, "capacity_mva"] == 450.0 and view.at[0, "n_circuits"] == 3
    # row1: 400/2=200 → circuits 1 → 200
    assert view.at[1, "capacity_mva"] == 200.0 and view.at[1, "n_circuits"] == 1
    # row2 untouched
    assert view.at[2, "capacity_mva"] == 500.0 and not view.at[2, "resize"]
    # row3: clamp at 8 → capacity restored unchanged (800/8*8)
    assert view.at[3, "capacity_mva"] == 800.0 and view.at[3, "n_circuits"] == 8
    assert view.loc[[0, 1, 3], "resize"].all()


def test_step_circuits_downsize_floor():
    view = pd.DataFrame({
        "overloaded": [False], "undersized": [True],
        "capacity_mva": [100.0], "n_circuits": [1.0], "resize": [False]})
    step_circuits(view)
    assert view.at[0, "n_circuits"] == 1          # clamp floor
    assert view.at[0, "capacity_mva"] == 100.0


@needs_matlab
@pytest.mark.slow
def test_dc_resize_loop_statistical(tmp_path):
    """Full DC loop from the s6 state: must terminate in the same regime as
    the MATLAB run (38 iterations, 18 overloads at exit, s7 log) — the
    randperm subsets differ (DV3), so trajectory/aggregate comparisons only
    (D9b)."""
    from gridweld.params.conductors import (crucible_catalog,
                                            load_cats_catalog)
    from gridweld.params.impedance import load_gmd_tables

    cfg = WeldConfig()
    rng = np.random.default_rng(cfg.system.rng_seed)
    br = pd.read_csv(FIX / "s6_branch_post_sizetransformers.csv",
                     low_memory=False)
    re = pd.read_csv(FIX / "s5_snetfull_real.csv", header=None).to_numpy()
    im = pd.read_csv(FIX / "s5_snetfull_imag.csv", header=None).to_numpy()
    out, log = run_dc_resize(
        br, re + 1j * im, 8822,
        pd.read_parquet(DATA / "xfdata.parquet"),
        pd.read_parquet(DATA / "xfxr.parquet"),
        crucible_catalog(), load_cats_catalog(), load_gmd_tables(),
        cfg, rng)

    fix_log = pd.read_csv(FIX / "s7_dcpf_iteration_log.csv", header=None,
                          names=["iter", "overloads", "total_rate", "max_dcpf"])
    assert len(log) <= 120                        # same regime as 38 (valve!)
    assert log["overloads"].iloc[-1] <= max(
        0, len(log) - cfg.resize.valve_after_iters)
    # overload count must decrease strongly from the initial state
    assert log["overloads"].iloc[0] == pytest.approx(
        fix_log["overloads"].iloc[0], rel=0.1)    # deterministic first pass
    assert log["overloads"].iloc[-1] < 0.05 * log["overloads"].iloc[0]
    # total capacity lands in the MATLAB run's neighborhood
    assert log["total_rate_mva"].iloc[-1] == pytest.approx(
        fix_log["total_rate"].iloc[-1], rel=0.15)


def test_correct_power_factors_bands():
    p = np.array([100.0, 100.0, 100.0, 100.0])
    # pf: 0.85 → 0.97; 0.93 → 0.98; 0.955 → 0.99; 0.99 → unchanged
    q = p * np.sqrt(1 / np.array([0.85, 0.93, 0.955, 0.99]) ** 2 - 1)
    out = correct_power_factors(-(p + 1j * q))    # load-negative convention
    new_pf = np.abs(out.real) / np.abs(out)
    assert np.allclose(new_pf, [0.97, 0.98, 0.99, 0.99], atol=1e-9)
    assert (out.imag <= 0).all()                  # sign preserved


def test_duplicate_max_scenario_clamps_window(caplog):
    """DV-P3-11: the legacy 1:42 window crashed on SNetComplete's 36
    columns — the port clamps with a warning."""
    cfg = WeldConfig()
    rng = np.random.default_rng(0)
    snet = -(rng.random((10, 36)) + 1j * rng.random((10, 36)))
    peak_col = 7
    snet[:, peak_col] *= 5
    import logging
    with caplog.at_level(logging.WARNING):
        out = duplicate_max_scenario(snet, cfg)
    assert out.shape == (10, 37)
    assert np.array_equal(out[:, 0], snet[:, peak_col])
    assert "clamped" in caplog.text


def test_reactive_winnowing_on_toy():
    from gridweld.model import WeldModel
    from gridweld.resize.reactive import place_reactive_support
    rng = np.random.default_rng(3)
    n = 12
    # ring + chords, gen at 0 (slack) and 1
    f = np.arange(n)
    t = (np.arange(n) + 1) % n
    branches = pd.DataFrame({
        "from_bus": np.concatenate([f, [0, 3]]),
        "to_bus": np.concatenate([t, [6, 9]]),
        "tap": 0.0, "r_pu": 0.01, "x_pu": 0.08, "b_pu": 0.02})
    model = WeldModel(buses=pd.DataFrame({"bus_id": range(n)}),
                      branches=branches,
                      generators=pd.DataFrame({"gen_id": [0, 1],
                                               "bus": [0, 1]}))
    load = -(20 + 5 * rng.random(n))
    load[0] = -load[1:].sum() * 1.02              # slack picks up losses
    load[1] = 60.0
    snet = (load + 1j * 0.3 * load).astype(complex)[:, None] \
        * np.ones((1, 3))
    cfg = WeldConfig()
    out = place_reactive_support(model, snet, 0, cfg)
    assert out["converged"]
    assert out["sync_cond"].dtype == bool
    n_typed = 1 + len(out["pv"]) + len(out["pq"])
    assert n_typed == n
    assert 0 not in out["pv"] and 0 not in out["pq"]


def test_ac_resize_terminates_without_overloads():
    """run_ac_resize must exit on pass 1 (before touching the catalogs —
    passed as None) when every branch has ample rating."""
    from gridweld.resize.ac_loop import run_ac_resize
    cfg = WeldConfig()
    rng = np.random.default_rng(0)
    n = 6
    br = pd.DataFrame({
        "From": np.arange(1, n), "To": np.arange(2, n + 1),
        "Tap": 0.0, "R": 0.01, "X": 0.08, "B": 0.02,
        "RateA": 1e4, "CondIndex": 10.0, "MaxCondIndex": 20.0,
        "MinCondIndex": 1.0, "Circuits": 1.0, "Resize": 0.0,
        "Overloaded": 0.0, "Undersized": 0.0})
    load = np.full(n, -10.0)
    load[0] = 10.0 * (n - 1) * 1.05
    snet = load.astype(complex)[:, None] * np.ones((1, 4))
    out, log = run_ac_resize(
        br, snet, n, 0, np.array([], int), np.arange(1, n), np.ones(n),
        None, None, None, None, None, cfg, rng)
    assert len(log) == 1
    assert log["eligible_overloads"].iloc[0] == 0


def _toy_chain(n: int, rate: float = 1e4) -> pd.DataFrame:
    """MATLAB-named n-bus chain branch frame (n−1 lines, no transformers)
    with enough columns for the full ac_loop resize path."""
    m = n - 1
    return pd.DataFrame({
        "From": np.arange(1, n), "To": np.arange(2, n + 1),
        "Tap": 0.0, "R": 0.01, "X": 0.08, "B": 0.02,
        "RateA": rate, "CondIndex": 10.0, "MaxCondIndex": 20.0,
        "MinCondIndex": 1.0, "Circuits": 1.0, "Resize": 0.0,
        "Overloaded": 0.0, "Undersized": 0.0,
        "Length": 10.0, "GMD": 30.0, "kVFrom": 115.0, "kVTo": 115.0,
        "FID": np.arange(m, dtype=float)})


def test_ac_envelope_dc_substitution():
    """DV-K (D-7 ruling): a scenario the AC solver cannot converge
    contributes its DC |MW| flows to the envelope; with the faithful
    zero-fill flag it vanishes entirely."""
    from gridweld.resize.ac_loop import _ac_envelope
    cfg = WeldConfig()
    n = 6
    br = _toy_chain(n)
    load = np.full(n, -10.0)
    load[0] = 10.0 * (n - 1) * 1.05
    absurd = load * 1e3            # far beyond AC feasibility; DC solves
    scen = np.column_stack([load, absurd]).astype(complex)
    env, frac, n_dc = _ac_envelope(br, n, scen, 0, np.array([], int),
                                   np.arange(1, n), np.ones(n), cfg)
    assert frac == pytest.approx(0.5)
    assert n_dc == 1
    # the chain's DC flows under the absurd scenario are (n−1−i)·1e4 MW
    assert env.min() > 5e3
    assert env.max() > 4e4
    cfg.resize.ac_dc_substitute = False
    env0, frac0, n0 = _ac_envelope(br, n, scen, 0, np.array([], int),
                                   np.arange(1, n), np.ones(n), cfg)
    assert frac0 == pytest.approx(0.5) and n0 == 0
    assert env0.max() < 100.0      # only the benign scenario contributes


def test_ac_resize_collapse_returns_input(caplog):
    """D-7 exit ruling: a 0-converged pass exits as 'collapse' BEFORE the
    termination valve (here eligible ≤ term would otherwise read as
    convergence) and returns the best-pass — i.e. the input — state."""
    import logging

    from gridweld.resize.ac_loop import run_ac_resize
    cfg = WeldConfig()
    rng = np.random.default_rng(0)
    n = 6
    br = _toy_chain(n)
    load = np.full(n, -1e4)
    load[0] = 1e4 * (n - 1) * 1.05
    snet = load.astype(complex)[:, None] * np.ones((1, 4))
    with caplog.at_level(logging.WARNING):
        out, lg = run_ac_resize(
            br, snet, n, 0, np.array([], int), np.arange(1, n), np.ones(n),
            None, None, None, None, None, cfg, rng)
    assert lg.attrs["exit"] == "collapse"
    assert lg.attrs["returned_pass"] == 1
    assert len(lg) == 1
    assert lg["converged_fraction"].iloc[0] == 0.0
    assert lg["dc_substituted"].iloc[0] == 3
    assert np.array_equal(out["RateA"].to_numpy(), br["RateA"].to_numpy())
    assert "NOT credible" in caplog.text


def test_ac_resize_zero_fill_collapse_is_not_success():
    """Faithful zero-fill mode (ac_dc_substitute=False) with no converged
    scenario must still exit as 'collapse' — never the legacy silent
    success where the all-zero envelope reads as zero overloads."""
    from gridweld.resize.ac_loop import run_ac_resize
    cfg = WeldConfig()
    cfg.resize.ac_dc_substitute = False
    rng = np.random.default_rng(0)
    n = 6
    br = _toy_chain(n)
    load = np.full(n, -1e4)
    load[0] = 1e4 * (n - 1) * 1.05
    snet = load.astype(complex)[:, None] * np.ones((1, 4))
    out, lg = run_ac_resize(
        br, snet, n, 0, np.array([], int), np.arange(1, n), np.ones(n),
        None, None, None, None, None, cfg, rng)
    assert lg.attrs["exit"] == "collapse"
    assert lg["eligible_overloads"].iloc[0] == 0   # zero envelope
    assert lg["dc_substituted"].iloc[0] == 0


def test_ac_resize_best_pass_snapshot(monkeypatch):
    """D-7 exit ruling: on collapse after productive passes the returned
    frame is the BEST-converged pass's ENTRY state. The envelope is
    scripted and the resize machinery stubbed to bump RateA by +1 per
    pass, so each pass's entry state is identifiable by its RateA."""
    import gridweld.resize.ac_loop as al
    fracs = iter([0.5, 0.8, 0.4, 0.0])

    def fake_env(br, n_buses, scen, slack, pv, pq, vset, cfg):
        return np.full(len(br), 1e9), next(fracs), 0

    def fake_resize(view, env, **kw):
        view["capacity_mva"] = view["capacity_mva"] + 1.0
        return {"upsized": len(view), "downsized": 0, "downsized_keep": 0}

    def fake_rxb(df, cat, gmd, cfg, matlab_compat=True):
        return pd.DataFrame({
            "r_pu": df["R"].to_numpy(), "x_pu": df["X"].to_numpy(),
            "b_pu": df["B"].to_numpy(),
            "capacity_mva": df["capacity_mva"].to_numpy()}, index=df.index)

    monkeypatch.setattr(al, "_ac_envelope", fake_env)
    monkeypatch.setattr("gridcrucible.network.conductors.resize_branches",
                        fake_resize)
    monkeypatch.setattr("gridweld.resize.dc_loop.step_circuits",
                        lambda view, max_circuits=8: 0)
    monkeypatch.setattr("gridweld.params.conductors.assign_conductors",
                        lambda df, cat, change_voltages=False: df.copy())
    monkeypatch.setattr("gridweld.params.impedance.compute_line_impedances",
                        fake_rxb)
    monkeypatch.setattr("gridweld.params.transformers.transformer_impedances",
                        lambda df, xfdata, xfxr, s_base=100.0: df.copy())

    cfg = WeldConfig()
    rng = np.random.default_rng(0)
    n = 8                                     # 7 branches > term (5)
    br = _toy_chain(n, rate=100.0)
    snet = np.full(n, -1.0 + 0j)[:, None] * np.ones((1, 3))
    out, lg = al.run_ac_resize(
        br, snet, n, 0, np.array([], int), np.arange(1, n), np.ones(n),
        None, None, None, None, None, cfg, rng)
    assert lg.attrs["exit"] == "collapse"
    assert lg.attrs["returned_pass"] == 2
    assert lg.attrs["best_converged_fraction"] == pytest.approx(0.8)
    assert len(lg) == 4
    # pass-2 entry state: exactly one +1 resize bump applied
    assert np.allclose(out["RateA"].to_numpy(),
                       br["RateA"].to_numpy() + 1.0)


def test_network_statistics_dv_p3_7():
    br = pd.DataFrame({
        "Length": [10.0, 20.0, 5.0, 8.0],
        "kVFrom": [115.0, 230.0, 500.0, 765.0],
        "RateA": [100.0, 200.0, 400.0, 800.0],
        "Tap": [0.0, 0.0, 0.0, 1.0]})
    bus = pd.DataFrame({"kV": [115.0, 230.0, 500.0]})
    import scipy.sparse as sp
    a = sp.csc_matrix(np.array([[1, 0, 0, 0], [-1, 1, 1, 0],
                                [0, -1, -1, -1]]))
    compat = network_statistics(br, bus, a, matlab_compat=True)
    fixed = network_statistics(br, bus, a, matlab_compat=False)
    hv = (20 * 200 + 5 * 400 + 8 * 800) / 1000.0
    # compat: 765 row clobbered with the HV total
    assert compat["branch"].iloc[-1]["gva_miles"] == pytest.approx(hv)
    # fixed: 765 keeps its own value; explicit hv_total row appended
    assert fixed["branch"].iloc[-2]["gva_miles"] == pytest.approx(6.4)
    assert fixed["branch"].iloc[-1]["gva_miles"] == pytest.approx(hv)
    assert compat["length"].iloc[-1]["total"] == pytest.approx(33.0)
    assert compat["node_degree"].tolist() == [1, 3, 3]
