"""P2a validation: MATPOWER parser/writer round-trip, Julia fixture
cross-checks, the ReadMatpower merge vs the deterministic S3 fixtures, and
the .mat/.csv → parquet converters vs the dumped MATLAB tables.

Fixture provenance (validation/fixtures/): julia/ = dump_reference.jl run
2026-07-16 (rng 42); matlab/ = dump_reference.m run 2026-07-15 (rng 42).
The 8822-bus external case the MATLAB dump consumed is reconstructed from
work/SynGrid.mat → mpc.order.ext (MATPOWER ext2int stores the full external
case there) — the standalone .m never made it into the repo.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.io.matlab_data import (convert_conductor_mats, convert_form1,
                                     convert_gmd_tables, convert_tower_data,
                                     convert_xf_tables)
from gridweld.io.matpower import (case_to_model, merge_topology, read_matpower_case, write_matpower_case)

FIX = REPO_ROOT / "validation" / "fixtures"
LEGACY = REPO_ROOT / "legacy"
needs_julia = pytest.mark.skipif(not (FIX / "julia" / "manifest.txt").exists(),
                                 reason="julia fixtures not extracted")
needs_matlab = pytest.mark.skipif(not (FIX / "matlab" / "manifest.txt").exists(),
                                  reason="matlab fixtures not extracted")
# The original CATS tree (legacy/ on the legacy-code branch) is not on main.
needs_legacy = pytest.mark.skipif(not (LEGACY / "Jonathan_Code").is_dir(),
                                  reason="legacy CATS tree not present")


@pytest.fixture(scope="module")
def legacy_case():
    path = LEGACY / "cal_largest_grid.m"
    if not path.exists():
        pytest.skip("legacy cal_largest_grid.m not present")
    return read_matpower_case(path)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_legacy_case_shapes(legacy_case):
    c = legacy_case
    assert c.name == "cal_largest_grid"
    assert c.base_mva == 1.0
    assert len(c.bus) == 8870 and len(c.branch) == 10823
    assert len(c.gen) == 2149 and len(c.gencost) == 2149
    assert set(c.cells) == {"bus_data", "gen_data", "branch_data", "load_data"}
    assert list(c.cells["bus_data"].columns) == ["Lon", "Import", "Lat", "Type", "FID"]


def test_parse_legacy_case_spot_values(legacy_case):
    c = legacy_case
    assert c.bus[0, 0] == 1 and c.bus[0, 2] == pytest.approx(69.23)
    assert c.gen[0, 0] == 998 and c.gen[0, 8] == pytest.approx(12.1)
    assert np.isnan(c.branch[0, 4])            # b = NaN at S2 (Q7)
    assert c.gencost[0, 4] == pytest.approx(0.059172355)
    row0 = c.cells["bus_data"].iloc[0]
    assert row0["FID"] == 0 and row0["Type"] == "Substation"
    assert row0["Import"] == " "


def test_roundtrip_write_read(legacy_case, tmp_path):
    model = case_to_model(legacy_case)
    out = write_matpower_case(model, tmp_path / "rt.m")
    back = read_matpower_case(out)
    for name, arr in legacy_case.arrays.items():
        got = back.arrays[name]
        assert got.shape[0] == arr.shape[0], name
        w = min(got.shape[1], arr.shape[1])
        assert np.allclose(got[:, :w], arr[:, :w], rtol=0, atol=0,
                           equal_nan=True), name
    for name, df in legacy_case.cells.items():
        got = back.cells[name]
        for col in df.columns:
            a, b = df[col], got[col]
            if pd.api.types.is_numeric_dtype(a):
                assert np.allclose(a.astype(float), b.astype(float),
                                   equal_nan=True), (name, col)
            else:
                assert (a.fillna("nan").astype(str)
                        == b.fillna("nan").astype(str)).all(), (name, col)


def test_case_to_model_ids_are_internal(legacy_case):
    model = case_to_model(legacy_case)
    assert model.validate() == []
    assert model.buses["bus_id"].tolist() == list(range(8870))
    # external ids only at the boundary: branch endpoints are 0-based rows
    assert model.branches["from_bus"].max() < 8870
    assert model.meta["per_unit"] is True and model.meta["base_mva"] == 1.0


# ---------------------------------------------------------------------------
# Julia fixture cross-checks (same run: dump .m ⇄ dump CSVs)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia_case():
    return read_matpower_case(FIX / "julia" / "cal_largest_grid_dump.m")


@needs_julia
def test_julia_dump_bus_matches_csv(julia_case):
    csv = pd.read_csv(FIX / "julia" / "p5_final_largest_bus.csv")
    c = julia_case
    assert len(c.bus) == len(csv) == 8870
    by_index = csv.set_index("index").sort_index()
    order = np.argsort(c.bus[:, 0])
    assert np.array_equal(c.bus[order, 0].astype(int), by_index.index.to_numpy())
    assert np.allclose(c.bus[order, 9], by_index["base_kv"].to_numpy())
    assert np.array_equal(c.bus[order, 1].astype(int),
                          by_index["bus_type"].to_numpy())


@needs_julia
def test_julia_dump_branch_matches_csv(julia_case):
    csv = pd.read_csv(FIX / "julia" / "p5_final_largest_branch.csv")
    c = julia_case
    assert len(c.branch) == len(csv) == 10823
    # branch matrix rows are sorted by component index on export
    by_index = csv.sort_values("index")
    assert np.array_equal(c.branch[:, 0].astype(int),
                          by_index["f_bus"].to_numpy())
    assert np.array_equal(c.branch[:, 1].astype(int),
                          by_index["t_bus"].to_numpy())
    tap = by_index["tap"].fillna(0.0).to_numpy()
    assert np.allclose(c.branch[:, 8], tap)
    xf = by_index["transformer"].to_numpy(dtype=bool)
    assert np.array_equal(c.branch[:, 8] != 0, xf)


@needs_julia
def test_julia_dump_gen_and_cost_match_csv(julia_case):
    csv = pd.read_csv(FIX / "julia" / "p5_final_largest_gen.csv").sort_values("index")
    c = julia_case
    assert np.array_equal(c.gen[:, 0].astype(int), csv["gen_bus"].to_numpy())
    assert np.allclose(c.gen[:, 8], csv["pmax"].to_numpy())
    cost = csv["cost"].map(
        lambda s: [float(x) for x in s.strip("[]").split(",")]).tolist()
    assert np.allclose(c.gencost[:, 4:7], np.asarray(cost))


@needs_julia
def test_julia_dump_loads_fold_into_bus_pd(julia_case):
    """Loads have no MATPOWER table — per-bus sums must equal bus Pd/Qd."""
    csv = pd.read_csv(FIX / "julia" / "p5_final_largest_load.csv")
    c = julia_case
    pd_by_bus = csv.groupby("load_bus")["pd"].sum()
    qd_by_bus = csv.groupby("load_bus")["qd"].sum()
    bus_pd = pd.Series(c.bus[:, 2], index=c.bus[:, 0].astype(int))
    bus_qd = pd.Series(c.bus[:, 3], index=c.bus[:, 0].astype(int))
    assert np.allclose(bus_pd.loc[pd_by_bus.index], pd_by_bus, rtol=1e-9)
    assert np.allclose(bus_qd.loc[qd_by_bus.index], qd_by_bus, rtol=1e-9)
    no_load = bus_pd.index.difference(pd_by_bus.index)
    assert (bus_pd.loc[no_load] == 0).all()


# ---------------------------------------------------------------------------
# ReadMatpower merge vs deterministic S3 fixtures
# ---------------------------------------------------------------------------

#: s2/s3 MATLAB branch-table header → model-style names (D6 / gridcrucible).
BRANCH_RENAME = {
    "FID": "fid", "kVFrom": "voltage_kv_from", "kVTo": "voltage_kv_to",
    "Length": "length_mi", "PathLength": "path_length_mi",
    "Circuits": "n_circuits", "Size": "kcmil", "CondPerPhase": "n_cond_per_phase",
    "RateA": "capacity_mva", "GMD": "gmd_ft", "R": "r_pu", "X": "x_pu",
    "B": "b_pu", "Overloaded": "overloaded", "Undersized": "undersized",
    "TowerIndex": "tower_index", "Resize": "resize", "CondIndex": "conductor_idx",
    "MaxCondIndex": "max_conductor_idx", "MinCondIndex": "min_conductor_idx",
    "Structure": "structure", "Material": "material",
    "Lat1": "lat1", "Lon1": "lon1", "Lat2": "lat2", "Lon2": "lon2",
    "From": "from_bus", "To": "to_bus", "Tap": "tap",
}


@pytest.fixture(scope="module")
def mpcext_8822():
    """Reconstruct the 8822-bus external case from SynGrid.mat (mpc.order.ext
    + the untouched cell tables). Cell column names are positional — the
    same fixed positions ReadMatpower.m's constants assume."""
    import scipy.io as sio
    from gridweld.io.matpower import MatpowerCase
    sg = sio.loadmat(FIX / "matlab" / "work" / "SynGrid.mat", simplify_cells=True)
    mpc = sg["mpc"]
    ext = mpc["order"]["ext"]
    case = MatpowerCase(name="cal_largest_grid_8822", version="2", base_mva=1.0)
    case.arrays = {"bus": np.asarray(ext["bus"], dtype=float),
                   "gen": np.asarray(ext["gen"], dtype=float),
                   "branch": np.asarray(ext["branch"], dtype=float),
                   "gencost": np.asarray(ext["gencost"], dtype=float)}
    case.cells["bus_data"] = pd.DataFrame(
        mpc["bus_data"], columns=["Lon", "Import", "Lat", "Type", "FID"])
    case.cells["gen_data"] = pd.DataFrame(
        mpc["gen_data"],
        columns=["PlantCode", "FuelType", "Lon", "GenID", "Lat",
                 "gen_bus_distance"])
    case.cells["branch_data"] = pd.DataFrame(
        mpc["branch_data"],
        columns=["t_bus_distance", "Lon2", "Lon1", "kV", "FID",
                 "f_bus_distance", "Lat1", "Lat2"])
    # sanity: the positional naming matches ReadMatpower's constants
    assert case.cells["bus_data"]["Type"].iloc[0] in ("Substation", "AddedNode")
    assert case.cells["gen_data"]["FuelType"].iloc[0] == "Conventional Hydroelectric"
    assert float(case.cells["branch_data"]["FID"].iloc[0]) == 0.0
    return case


@needs_matlab
def test_in_repo_8822_file_matches_syngrid_recovery(mpcext_8822):
    """B3 follow-up (2026-07-17): pin_matpower_version.m showed loadcase
    resolved legacy/Jonathan_Code/Input Data/cal_largest_grid.m — the s3
    input case was in the repo all along. Assert it equals the SynGrid.mat
    order.ext recovery, excluding exactly the mutations ReadMatpower applied
    BEFORE ext2int (bus-type repair; branch default impedances; gen zero-pad
    to 21 columns)."""
    raw = read_matpower_case(
        LEGACY / "Jonathan_Code" / "Input Data" / "cal_largest_grid.m")
    ext = mpcext_8822
    assert len(raw.bus) == len(ext.bus) == 8822
    bus_cols = [c for c in range(13) if c != 1]              # skip BUS_TYPE
    assert np.allclose(raw.bus[:, bus_cols], ext.bus[:, bus_cols],
                       equal_nan=True)
    br_cols = [0, 1, 5, 6, 7, 8, 9, 10]                      # skip R/X/B
    assert np.allclose(raw.branch[:, br_cols], ext.branch[:, br_cols],
                       equal_nan=True)
    w = raw.gen.shape[1]
    assert np.allclose(raw.gen[:, :w], ext.gen[:, :w], equal_nan=True)
    assert np.allclose(ext.gen[:, w:], 0.0)                  # the APF zero-pad
    assert np.allclose(raw.gencost, ext.arrays["gencost"], equal_nan=True)
    for name in ("bus_data", "gen_data", "branch_data"):
        a, b = raw.cells[name], ext.cells[name]
        assert len(a) == len(b)
        for col in b.columns:
            av = pd.to_numeric(a[col], errors="coerce")
            bv = pd.to_numeric(b[col], errors="coerce")
            if name == "branch_data" and col == "FID":
                # ReadMatpower replaced the 680 NaN transformer FIDs with
                # fresh ids BEFORE SynGrid.mat was saved — the raw file
                # still has the NaNs (the only cell-table mutation).
                # (680 XF rows; 10174 GIS-FID rows; 265 GIS lines unmatched
                # → dropped: 10439 − 265 + 680 = 10854 = the s3 row count.)
                fresh = av.isna()
                assert fresh.sum() == 680
                assert (bv[fresh] > 10438).all() and bv[fresh].notna().all()
                num = ~fresh
            else:
                num = av.notna() & bv.notna()
            assert np.allclose(av[num], bv[num]), (name, col)
            if name == "branch_data" and col == "FID":
                continue                     # fresh-id rows handled above
            assert (a[col].astype(str)[~num] == b[col].astype(str)[~num]).all(), \
                (name, col)


@needs_matlab
def test_merge_topology_matches_s3_fixtures(mpcext_8822):
    cfg = WeldConfig()
    gis = pd.read_csv(FIX / "matlab" / "s2_branch_post_rxb.csv",
                      low_memory=False).rename(columns=BRANCH_RENAME)
    bus, branch, gen, params = merge_topology(gis, mpcext_8822, cfg,
                                              matlab_compat=True)

    fb = pd.read_csv(FIX / "matlab" / "s3_bus_post_readmatpower.csv")
    fg = pd.read_csv(FIX / "matlab" / "s3_gen_post_readmatpower.csv")
    fbr = pd.read_csv(FIX / "matlab" / "s3_branch_post_readmatpower.csv",
                      low_memory=False).rename(columns=BRANCH_RENAME)

    # --- branch: FID join + appended transformers ---
    assert len(branch) == len(fbr) == 10854
    for col in ("from_bus", "to_bus"):
        assert np.array_equal(branch[col].to_numpy(dtype=np.int64),
                              fbr[col].to_numpy(dtype=np.int64)), col
    assert np.allclose(branch["tap"], fbr["tap"])
    for col in ("fid", "voltage_kv_from", "voltage_kv_to", "length_mi",
                "n_cond_per_phase", "n_circuits", "gmd_ft", "capacity_mva",
                "r_pu", "x_pu", "b_pu", "resize", "conductor_idx",
                "max_conductor_idx", "min_conductor_idx"):
        assert np.allclose(branch[col].to_numpy(dtype=float),
                           fbr[col].to_numpy(dtype=float), equal_nan=True), col
    xf = branch["tap"].to_numpy() > 0
    assert (branch.loc[xf, "structure"] == "Transformer").all()
    assert (fbr.loc[xf, "structure"] == "Transformer").all()

    # --- bus table (matlab_compat reproduces DV-P3-1 MaxMW values) ---
    assert len(bus) == len(fb) == 8822
    assert np.array_equal(bus["bus_num"], fb["Num"])
    assert np.array_equal(bus["external_num"], fb["NumExt"])
    assert np.allclose(bus["lat"], fb["Lat"]) and np.allclose(bus["lon"], fb["Lon"])
    assert np.array_equal(bus["fid"].astype(int), fb["FID"])
    assert (bus["node_type"] == fb["Category"]).all()
    assert np.array_equal(bus["bus_type"], fb["Type"])
    assert np.allclose(bus["pd_mw"], fb["MW"]) and np.allclose(bus["qd_mvar"], fb["MVar"])
    assert np.allclose(bus["base_kv"], fb["kVActual"])
    assert np.allclose(bus["kv_class"], fb["kV"])
    assert np.allclose(bus["max_mw"], fb["MaxMW"])

    # --- gen table ---
    assert len(gen) == len(fg) == 2298
    assert np.array_equal(gen["bus"], fg["BusNum"])
    assert np.array_equal(gen["unit_id"], fg["ID"])
    for a, b in (("pmin_mw", "MinMW"), ("pmax_mw", "MaxMW"),
                 ("qmin_mvar", "MinMVar"), ("qmax_mvar", "MaxMVar"),
                 ("vg_pu", "Vg"), ("c0", "A"), ("c1", "B"), ("c2", "C")):
        assert np.allclose(gen[a], fg[b]), (a, b)
    assert (gen["fuel_type"] == fg["FuelType"]).all()
    assert np.array_equal(gen["plant_code"].fillna(0).astype(int),
                          fg["PlantCode"].fillna(0).astype(int))

    assert params["total_lines"] == 10854
    assert params["buses"] == 8822


@needs_matlab
def test_merge_topology_fixed_mode_keeps_min_mw(mpcext_8822):
    cfg = WeldConfig()
    gis = pd.read_csv(FIX / "matlab" / "s2_branch_post_rxb.csv",
                      low_memory=False).rename(columns=BRANCH_RENAME)
    bus, _, gen, _ = merge_topology(gis, mpcext_8822, cfg, matlab_compat=False)
    assert "min_mw" in bus.columns
    gen_max = gen.groupby("bus")["pmax_mw"].sum()
    got = bus.set_index("bus_num").loc[gen_max.index, "max_mw"]
    assert np.allclose(got, gen_max)          # no Pmin overwrite in fixed mode


# ---------------------------------------------------------------------------
# matlab_data converters
# ---------------------------------------------------------------------------

@needs_matlab
def test_conductor_pool_matches_matlab_fixture(tmp_path):
    cfg = WeldConfig()
    pooled = convert_conductor_mats(
        LEGACY / "Jonathan_Code" / "Data Files" / "Conductor Data Tables",
        tmp_path / "conductors.parquet", list(cfg.system.voltage_levels))
    fix = pd.read_csv(FIX / "matlab" / "s2_conductor_data.csv")
    assert len(pooled) == len(fix) == 101
    for py, ml in zip(["mva", "kv", "kcmil", "cond_per_phase", "ra_ohm_per_mi",
                       "xa_ohm_per_mi", "xap_mohm_mi", "cond_index"],
                      ["MVA", "kV", "Size", "CondPhase", "Ra", "Xa", "XaP",
                       "CondIndex"]):
        assert np.allclose(pooled[py], fix[ml]), (py, ml)
    assert (tmp_path / "conductors.parquet").exists()


@needs_legacy
def test_tower_and_form1_convert(tmp_path):
    tower = convert_tower_data(
        LEGACY / "Jonathan_Code" / "Input Data" / "TowerData.csv",
        tmp_path / "tower.parquet")
    assert len(tower) == 81
    form1 = convert_form1(LEGACY / "Jonathan_Code" / "Input Data" / "Form1.csv",
                          tmp_path / "form1.parquet")
    assert len(form1) == 292
    assert "State" in form1.columns and "MaterialType" in form1.columns


@needs_matlab
def test_xf_tables_convert(tmp_path):
    xfdata, xfxr = convert_xf_tables(FIX / "matlab" / "xfdata.csv",
                                     FIX / "matlab" / "xfxr.csv", tmp_path)
    assert {"Pri", "Sec", "XHL"} <= set(xfdata.columns)
    assert xfxr["MVA"].is_monotonic_increasing


@needs_legacy
def test_gmd_tables_convert(tmp_path):
    out = tmp_path / "gmd.npz"
    info = convert_gmd_tables(
        LEGACY / "Jonathan_Code" / "Data Files" /
        "GMD Statistics and lookup tables" / "GMDDataCAISO.m", out)
    z = np.load(out)
    assert set(z.files) == {"gmd_table", "xd", "xd_prime"}
    assert len(z["xd"]) == len(z["xd_prime"]) == info["xd_len"]
    assert np.isfinite(z["xd"]).all()
    # Xd(d) = 0.2794·log10(d): first entry (d=1 ft) is exactly 0
    assert z["xd"][0] == pytest.approx(0.0, abs=1e-12)
