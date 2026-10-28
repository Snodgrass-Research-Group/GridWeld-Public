"""P3a/P3b validation: the Form 1 seeding chain vs s2_branch_post_form1
(TowerIndex injected — the randi(3) draw is the only stochastic input),
SizeLines + RXB vs s2_branch_post_rxb, and TransformerImpedances vs the s6
fixture including the load-bearing stale-XHL carry (DV-P3-9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.params.conductors import (assign_conductors, crucible_catalog,
                                        init_branch_table, load_cats_catalog,
                                        matlab_catalog, seed_from_form1,
                                        size_lines)
from gridweld.params.impedance import compute_line_impedances, load_gmd_tables
from gridweld.params.transformers import transformer_impedances

FIX = REPO_ROOT / "validation" / "fixtures" / "matlab"
LEGACY = REPO_ROOT / "legacy"
DATA = REPO_ROOT / "src" / "gridweld" / "data"
needs_matlab = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                  reason="matlab fixtures not extracted")

pytestmark = needs_matlab


@pytest.fixture(scope="module")
def cfg() -> WeldConfig:
    return WeldConfig()


@pytest.fixture(scope="module")
def fix_form1() -> pd.DataFrame:
    return pd.read_csv(FIX / "s2_branch_post_form1.csv", low_memory=False)


@pytest.fixture(scope="module")
def seeded(cfg, fix_form1) -> pd.DataFrame:
    """The P3a chain: raw CAGISData.csv → init (TowerIndex injected from the
    fixture — the sole rng draw) → Form 1 seeding."""
    raw = pd.read_csv(
        LEGACY / "Jonathan_Code" / "Input Data" / "CAGISData.csv",
        low_memory=False, encoding_errors="replace")
    rng = np.random.default_rng(cfg.system.rng_seed)
    br = init_branch_table(raw, cfg, rng,
                           tower_index=fix_form1["TowerIndex"].to_numpy())
    form1 = pd.read_parquet(DATA / "form1.parquet")
    tower = pd.read_parquet(DATA / "tower.parquet")
    return seed_from_form1(br, form1, tower, matlab_catalog(), cfg)


def test_form1_seeding_matches_fixture(seeded, fix_form1):
    mine, fix = seeded, fix_form1
    assert len(mine) == len(fix) == 10439
    assert np.array_equal(mine["FID"].to_numpy(np.int64),
                          fix["FID"].to_numpy(np.int64))
    for col in ("kVFrom", "kVTo", "Circuits", "Size", "CondPerPhase",
                "RateA", "GMD"):
        assert np.allclose(mine[col].to_numpy(float),
                           fix[col].to_numpy(float), equal_nan=True), col
    for col in ("Length", "PathLength"):
        assert np.allclose(mine[col], fix[col], rtol=1e-9), col
    for col in ("Material", "Structure", "Type"):
        # strip: readtable trims the raw CSV's ' ' cells to '' (cosmetic)
        a = mine[col].fillna("").astype(str).str.strip().str.casefold()
        b = fix[col].fillna("").astype(str).str.strip().str.casefold()
        assert (a == b).all(), col
    assert (mine["Resize"] == 1).all()


def test_size_lines_and_rxb_match_fixture(seeded, cfg):
    fix = pd.read_csv(FIX / "s2_branch_post_rxb.csv", low_memory=False)
    sized = size_lines(seeded, matlab_catalog())
    for col in ("CondIndex", "MaxCondIndex", "MinCondIndex"):
        assert np.array_equal(sized[col].to_numpy(np.int64),
                              fix[col].to_numpy(np.int64)), col
    for col in ("Size", "CondPerPhase"):
        assert np.allclose(sized[col], fix[col]), col

    rxb_in = sized.rename(columns={
        "Length": "length_mi", "GMD": "gmd_ft", "CondIndex": "conductor_idx",
        "kVFrom": "voltage_kv_from", "Circuits": "n_circuits",
        "RateA": "capacity_mva", "FID": "fid"})
    out = compute_line_impedances(rxb_in, load_cats_catalog(),
                                  load_gmd_tables(), cfg, matlab_compat=True)
    for py, ml in (("r_pu", "R"), ("x_pu", "X"), ("b_pu", "B")):
        assert np.allclose(out[py], fix[ml], rtol=1e-9), (py, ml)
    # PSERCMain re-caps RateA after RXB (idempotent with the in-RXB cap)
    assert np.allclose(out["capacity_mva"], fix["RateA"], rtol=1e-9)


def test_assign_conductors_roundtrip_noop(cfg):
    """post-RXB rows have Resize == 0 → assign_conductors must be a no-op."""
    fix = pd.read_csv(FIX / "s2_branch_post_rxb.csv", low_memory=False)
    out = assign_conductors(fix.copy(), crucible_catalog())
    for col in ("CondIndex", "RateA", "Size", "CondPerPhase", "kVFrom"):
        assert np.allclose(out[col].to_numpy(float),
                           fix[col].to_numpy(float)), col


def test_assign_conductors_resnaps_flagged_rows():
    """Rigged single row: rating 3x the current conductor → per-circuit snap
    inside the kV block, capacity = circuits × exact table MVA."""
    cat = crucible_catalog()
    blk = cat[cat["kv"] == 115]
    row = pd.DataFrame({
        "RateA": [3 * blk["mva"].iloc[3]], "Circuits": [3.0],
        "CondIndex": [int(blk.index[0])], "kVFrom": [115.0], "kVTo": [115.0],
        "Size": [0.0], "CondPerPhase": [0.0], "Resize": [1.0],
        "MaxCondIndex": [int(blk.index[-1])], "MinCondIndex": [int(blk.index[0])],
        "Overloaded": [0.0], "Undersized": [0.0],
    })
    out = assign_conductors(row, cat)
    ci = int(out.at[0, "CondIndex"])
    assert cat.at[ci, "kv"] == 115
    assert out.at[0, "RateA"] == pytest.approx(3 * cat.at[ci, "mva"])
    assert out.at[0, "Resize"] == 0
    assert cat.at[ci, "mva"] == pytest.approx(blk["mva"].iloc[3])


# ---------------------------------------------------------------------------
# TransformerImpedances vs s6 (incl. stale-XHL carry)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s6_xf() -> pd.DataFrame:
    s6 = pd.read_csv(FIX / "s6_branch_post_sizetransformers.csv",
                     low_memory=False)
    return s6[s6["Tap"] != 0].reset_index(drop=True)


def test_transformer_impedances_match_s6(s6_xf):
    """Recompute X/R from the s6 rows' OWN RateA with the full stale-carry
    chain. Rows resized in the FINAL (partial-flag) TransformerImpedances
    call may have seen a different carry value — mismatches must be same-kV
    rows only and few (doc/DEVIATIONS DV-P3-9)."""
    xfdata = pd.read_parquet(DATA / "xfdata.parquet")
    xfxr = pd.read_parquet(DATA / "xfxr.parquet")
    inp = s6_xf.copy()
    inp["Resize"] = 1.0
    out = transformer_impedances(inp, xfdata, xfxr, s_base=100.0)

    assert np.array_equal(out["CondIndex"].to_numpy(float),
                          np.ceil(s6_xf["RateA"] / s6_xf["Circuits"]))
    x_ok = np.isclose(out["X"], s6_xf["X"], rtol=1e-9)
    r_ok = np.isclose(out["R"], s6_xf["R"], rtol=1e-9)
    same_kv = (s6_xf["kVFrom"] == s6_xf["kVTo"]).to_numpy()
    assert x_ok.mean() > 0.95 and r_ok.mean() > 0.95
    assert same_kv[~x_ok].all() and same_kv[~r_ok].all()
    # non-same-kV rows must be EXACT
    assert x_ok[~same_kv].all() and r_ok[~same_kv].all()


def test_transformer_impedances_boundaries():
    """RateA ∈ {49, 50, 2000, 2001} boundary semantics (doc 03)."""
    xfdata = pd.read_parquet(DATA / "xfdata.parquet")
    xfxr = pd.read_parquet(DATA / "xfxr.parquet")
    br = pd.DataFrame({
        "Tap": [1.0] * 4, "RateA": [49.0, 50.0, 2000.0, 2001.0],
        "kVFrom": [230.0] * 4, "kVTo": [500.0] * 4,
        "Circuits": [1.0] * 4, "Resize": [1.0] * 4,
        "MinCondIndex": [20.0] * 4, "Undersized": [0.0] * 4,
        "X": [np.nan] * 4, "R": [np.nan] * 4, "CondIndex": [0.0] * 4,
    })
    out = transformer_impedances(br, xfdata, xfxr)
    xhl = 14.6                                   # XFData 230/500
    for i, (rate, circ) in enumerate([(49, 1), (50, 1), (2000, 1), (2001, 2)]):
        assert out.at[i, "Circuits"] == circ
        if rate == 49:
            xr = 25.0
        else:
            xr = float(np.interp(rate / circ, xfxr["MVA"], xfxr["XR"]))
        z = (100.0 / rate) * (xhl / 100.0)
        assert out.at[i, "X"] == pytest.approx(
            np.sqrt(z * z / (1 + 1 / xr ** 2)), rel=1e-12)
        assert out.at[i, "R"] == pytest.approx(out.at[i, "X"] / xr, rel=1e-12)
    assert (out["CondIndex"] == np.ceil(out["RateA"] / out["Circuits"])).all()


def test_transformer_min_clamp_flags_undersized():
    xfdata = pd.read_parquet(DATA / "xfdata.parquet")
    xfxr = pd.read_parquet(DATA / "xfxr.parquet")
    br = pd.DataFrame({
        "Tap": [1.0], "RateA": [7.0], "kVFrom": [138.0], "kVTo": [230.0],
        "Circuits": [1.0], "Resize": [1.0], "MinCondIndex": [20.0],
        "Undersized": [0.0], "X": [np.nan], "R": [np.nan], "CondIndex": [0.0],
    })
    out = transformer_impedances(br, xfdata, xfxr)
    assert out.at[0, "RateA"] == 20.0
    assert out.at[0, "Undersized"] == 1.0


def test_transformer_same_kv_strict_raises():
    xfdata = pd.read_parquet(DATA / "xfdata.parquet")
    xfxr = pd.read_parquet(DATA / "xfxr.parquet")
    br = pd.DataFrame({
        "Tap": [1.0], "RateA": [100.0], "kVFrom": [138.0], "kVTo": [138.0],
        "Circuits": [1.0], "Resize": [1.0], "MinCondIndex": [20.0],
        "Undersized": [0.0], "X": [np.nan], "R": [np.nan], "CondIndex": [0.0],
    })
    with pytest.raises(ValueError, match="same-kV"):
        transformer_impedances(br, xfdata, xfxr, matlab_compat=False)
    out = transformer_impedances(br, xfdata, xfxr)   # compat: skipped row
    assert np.isnan(out.at[0, "X"])                  # no prior carry → skip
