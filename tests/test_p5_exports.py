"""P5 export validation: PowerWorld writers vs the legacy PowerWorld Data
CSVs (schema exact everywhere; values for every state-independent column —
the legacy CSVs were written at a later run state than the dcopf-fixture
frames, so Vm/Va/R/X/RateA-derived columns are not comparable), the
WriteLoads format, the multi-unit/sniff/sign quirks on toys, the GeoJSON
round-trip vs the shipped Final_Grid/{tl,bus,gen}.json (DV-07b/07f
exclusions), and the S2 QA reports."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.io.powerworld import (branch_import, bus_import, gen_import,
                                    peak_load_import, substation_import,
                                    switched_shunts_import,
                                    write_load_timeseries,
                                    write_powerworld_imports)

FIX = REPO_ROOT / "validation" / "fixtures" / "dcopf"
PW = (REPO_ROOT / "legacy" / "Jonathan_Code" / "Aditya_Code"
      / "PowerWorld Data")
FINAL = REPO_ROOT / "legacy" / "Final_Grid"
needs_dcopf = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                 reason="dcopf fixtures not extracted")
needs_pw = pytest.mark.skipif(not (PW / "BusImport.csv").exists(),
                              reason="legacy PowerWorld CSVs not present")
needs_gis = pytest.mark.skipif(not (FINAL / "tl.json").exists(),
                               reason="legacy Final_Grid JSONs not present")


@pytest.fixture(scope="module")
def frames():
    if not (FIX / "in_bus.csv").exists():
        pytest.skip("dcopf fixtures not extracted")
    bus = pd.read_csv(FIX / "in_bus.csv", low_memory=False)
    br = pd.read_csv(FIX / "in_branch.csv", low_memory=False)
    gen = pd.read_csv(FIX / "in_gen.csv", low_memory=False)
    return bus, br, gen


# ---------------------------------------------------------------------------
# WritePowerWorld vs legacy CSVs
# ---------------------------------------------------------------------------


@needs_dcopf
@needs_pw
def test_powerworld_schemas_and_state_free_values(frames, tmp_path):
    bus, br, gen = frames
    cfg = WeldConfig()
    snet = (pd.read_csv(FIX / "in_snetfull_real.csv", header=None).to_numpy()
            + 1j * pd.read_csv(FIX / "in_snetfull_imag.csv",
                               header=None).to_numpy())
    cfg.export.powerworld.network_tag = ""      # legacy catch-branch names
    paths = write_powerworld_imports(bus, br, gen, snet, tmp_path, cfg,
                                     ac_stage=True, units="mw")
    assert set(paths) == {"Bus", "Substation", "Branch", "PeakLoad", "Gen",
                          "SwitchedShunts"}

    for kind in paths:
        mine = pd.read_csv(paths[kind], low_memory=False)
        legacy = pd.read_csv(PW / f"{kind}Import.csv", low_memory=False)
        assert list(mine.columns) == list(legacy.columns), kind

    mine = pd.read_csv(paths["Bus"], low_memory=False)
    legacy = pd.read_csv(PW / "BusImport.csv", low_memory=False)
    for col in ("Number", "NomkV", "AreaNumber", "ZoneNumber", "NomB",
                "NomG", "Latitude", "Longitude"):
        assert np.allclose(mine[col], legacy[col]), col
    assert (mine["Name"].astype(str) == legacy["Name"].astype(str)).all()

    mine = pd.read_csv(paths["Branch"], low_memory=False)
    legacy = pd.read_csv(PW / "BranchImport.csv", low_memory=False)
    for col in ("BusNumFrom", "BusNumTo", "Circuit"):
        assert np.array_equal(mine[col], legacy[col]), col
    for col in ("BranchDeviceType", "IsXF"):
        assert (mine[col] == legacy[col]).all(), col
    assert np.allclose(mine["LineLength"], legacy["LineLength"],
                       equal_nan=True)
    # sanity of the derived limits on OUR state (legacy state differs)
    assert np.allclose(mine["LimitMVAB"], 1.1 * br["RateA"])
    assert np.allclose(mine["LimitMVAC"], 1.15 * br["RateA"])

    mine = pd.read_csv(paths["Gen"], low_memory=False)
    legacy = pd.read_csv(PW / "GenImport.csv", low_memory=False)
    for col in ("BusNum", "ID"):
        assert np.array_equal(mine[col], legacy[col]), col
    for col in ("MWMin", "MWMax", "MvarMin", "MvarMax", "FixedCostInd",
                "CubicB", "CubicC", "VoltSet", "PartFact"):
        assert np.allclose(mine[col], legacy[col]), col
    for col in ("FuelType", "CostModel", "AGC", "AVR"):
        assert (mine[col].astype(str) == legacy[col].astype(str)).all(), col


def test_gen_import_multi_unit_split():
    """DV-07a APPROVED (2026-07-19): bus-level setpoints split MWMax-
    proportionally across the bus's units; 'Open' hits EVERY unit at a
    decommitted bus (legacy: first unit only in both cases). Per-bus
    totals equal the legacy bus-level values."""
    cfg = WeldConfig()
    bus = pd.DataFrame({
        "Num": [1, 2], "GenMWSet": [50.0, 0.0],
        "PNet": [47.5, -3.0], "QNet": [1.5, 0.0]})
    gen = pd.DataFrame({
        "BusNum": [1, 1, 2, 2], "ID": [1, 2, 1, 2],
        "FuelType": ["Gas"] * 4, "MinMW": [0.0] * 4,
        "MaxMW": [10.0, 20.0, 30.0, 40.0], "MinMVar": [0.0] * 4,
        "MaxMVar": [0.0] * 4, "A": [1.0] * 4, "B": [2.0] * 4,
        "C": [0.0] * 4})
    out = gen_import(bus, gen, cfg)
    assert list(out["Status"]) == ["Closed", "Closed", "Open", "Open"]
    assert np.allclose(out["MWSetPoint"], [47.5 / 3, 95.0 / 3, 0.0, 0.0])
    assert np.isclose(out["MWSetPoint"][:2].sum(), 47.5)   # bus total kept
    assert np.allclose(out["MvarSetpoint"], [0.5, 1.0, 0.0, 0.0])
    assert np.allclose(out["PartFact"], np.array([10, 20, 30, 40]) / 100.0)


def test_peak_load_sign_and_sniff():
    """DV-07e RULED (2026-07-19): loads are the NEGATIVE real part of the
    injection-convention SNet — the default; 'positive' replicates the
    Aditya-fork bug (selected generation). DV-07c: the fork's <20000 MW
    auto-pu rule scales by SBase, with a warning."""
    cfg = WeldConfig()
    bus = pd.DataFrame({"Num": [1, 2, 3], "MW": [0.0] * 3,
                        "MVar": [0.0] * 3})
    snet = np.array([[100.0 + 10.0j], [-50.0 - 5.0j], [0.0 + 0.0j]])
    out = peak_load_import(bus, snet, cfg, units="mw")
    assert list(out["BusNum"]) == [2]              # negative = load (ruled)
    assert out.at[0, "SMW"] == 50.0 and out.at[0, "SMvar"] == 5.0
    cfg.export.powerworld.load_sign = "positive"   # fork replication
    out = peak_load_import(bus, snet, cfg, units="mw")
    assert list(out["BusNum"]) == [1]
    cfg.export.powerworld.load_sign = "negative"
    out = peak_load_import(bus, snet * 0.01, cfg, units="auto")   # pu-ish
    assert np.isclose(out.at[0, "SMW"], 50.0)      # sniff scaled ×100


def test_switched_shunts_blocks():
    cfg = WeldConfig()
    bus = pd.DataFrame({"Num": [1, 2, 3, 4],
                        "SyncCond": [1.0, 1.0, 1.0, 0.0],
                        "QNet": [50.0, -30.0, 0.0, 99.0]})
    out = switched_shunts_import(bus, cfg)
    assert list(out["BusNum"]) == [1, 2, 3]        # SyncCond rows only
    assert np.allclose(out["BlockMvarPerStep1"], [-1.0, -math.ceil(39 / 10), 0])
    assert np.allclose(out["BlockMvarPerStep2"], [math.ceil(65 / 10), 1.0, 0])
    assert np.allclose(out["MvarNom"], [50.0, -30.0, 0.0])
    assert (out["BlockNumberStep1"] == 10).all()


def test_write_load_timeseries_roundtrip(tmp_path):
    snet = np.array([[10.0 + 1j, -20.0 - 2j], [5.0 + 0.5j, 15.0 + 1.5j]])
    mw, mvar = write_load_timeseries(snet, tmp_path, s_base=100.0,
                                     units="mw")
    got = pd.read_csv(mw, header=None).to_numpy()
    assert got.shape == (2, 2)                     # rows=scenarios
    assert np.allclose(got, np.abs(snet.real).T)
    # auto sniff: tiny pu totals vs big Bus.MW → ×SBase
    mw, _ = write_load_timeseries(snet / 100, tmp_path, s_base=100.0,
                                  units="auto",
                                  bus_mw=np.array([1000.0, 2000.0]))
    got = pd.read_csv(mw, header=None).to_numpy()
    assert np.allclose(got, np.abs(snet.real).T)
    with pytest.raises(ValueError):
        write_load_timeseries(snet, tmp_path, s_base=100.0, units="auto")


def test_bus_import_optional_voltage(frames):
    bus, _, _ = frames
    out = bus_import(bus)
    assert "Vpu" in out.columns
    # DV-07g: default converts the S5 radians to PowerWorld degrees
    assert np.allclose(out["Vangle"],
                       np.degrees(bus["Va"].to_numpy(float)))
    out_deg = bus_import(bus, va_unit="deg")
    assert np.allclose(out_deg["Vangle"], bus["Va"].to_numpy(float))
    out2 = bus_import(bus.drop(columns=["Vm", "Va"]))
    assert "Vpu" not in out2.columns               # the try/catch branch
    assert list(substation_import(bus).columns) == ["Number", "Name",
                                                    "Latitude", "Longitude"]


@needs_dcopf
def test_branch_import_device_type(frames):
    _, br, _ = frames
    cfg = WeldConfig()
    out = branch_import(br, cfg)
    assert (out["IsXF"] == "YES").sum() == 663     # the CATS transformers
    assert ((out["BranchDeviceType"] == "Transformer")
            == (out["IsXF"] == "YES")).all()


# ---------------------------------------------------------------------------
# GeoJSON round-trip vs shipped Final_Grid JSONs
# ---------------------------------------------------------------------------


def _eqv(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _props_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in b:
        va, vb = a[k], b[k]
        if isinstance(vb, list):
            if (not isinstance(va, list) or len(va) != len(vb)
                    or any(not _eqv(x, y) for x, y in zip(va, vb))):
                return False
        elif not _eqv(va, vb):
            return False
    return True


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    from gridweld.io.geojson import roundtrip_gis
    out_dir = tmp_path_factory.mktemp("gis_roundtrip")
    cfg = WeldConfig()
    # legacy replace-props mode: the shipped bus.json embodies it; the
    # production default preserves originals (DV-07d approved 2026-07-19)
    cfg.export.gis.preserve_added_node_props = False
    return roundtrip_gis(REPO_ROOT / "legacy" / "cal_largest_grid.m",
                         FINAL / "CATS.m", out_dir, cfg)


@needs_gis
def test_roundtrip_tl_matches_shipped(roundtrip):
    mine = json.load(open(roundtrip["tl"], encoding="utf-8"))
    ship = json.load(open(FINAL / "tl.json", encoding="utf-8"))
    assert len(mine["features"]) == len(ship["features"]) == 11100
    m = {f["properties"]["CATS_ID"]: f for f in mine["features"]
         if "CATS_ID" in f.get("properties", {})}
    s = {f["properties"]["CATS_ID"]: f for f in ship["features"]
         if "CATS_ID" in f.get("properties", {})}
    assert set(m) == set(s) and len(s) == 10823
    bad = [k for k in s
           if not _props_equal(m[k]["properties"], s[k]["properties"])]
    assert not bad, f"{len(bad)} features differ, first {bad[:3]}"
    # geometries: identical for real lines; transformers EXCLUDED (DV-07b —
    # the shipped coordinates are the [[Lon1,Lon2],[Lon2,Lat2]] bug)
    geom_bad = [k for k in s if not s[k]["properties"]["transformer"]
                and m[k]["geometry"] != s[k]["geometry"]]
    assert not geom_bad


@needs_gis
def test_roundtrip_bus_matches_shipped(roundtrip):
    mine = json.load(open(roundtrip["bus"], encoding="utf-8"))
    ship = json.load(open(FINAL / "bus.json", encoding="utf-8"))
    assert len(mine["features"]) == len(ship["features"])
    ms = {f["properties"]["CATS_ID"]: f for f in mine["features"]
          if f["properties"].get("Type") == "Substation"}
    ss = {f["properties"]["CATS_ID"]: f for f in ship["features"]
          if f["properties"].get("Type") == "Substation"}
    assert set(ms) == set(ss) and len(ss) == 3171
    for k in ss:
        assert ms[k]["properties"] == ss[k]["properties"]
        assert all(_eqv(a, b) for a, b in
                   zip(ms[k]["geometry"]["coordinates"],
                       ss[k]["geometry"]["coordinates"]))
    # AddedNode SLOTS identical; slot survivors at shared FIDs are hash-
    # order-arbitrary in the shipped file (DV-07f) — only slots compared
    man = [i for i, f in enumerate(mine["features"])
           if f["properties"].get("Type") == "AddedNode"]
    san = [i for i, f in enumerate(ship["features"])
           if f["properties"].get("Type") == "AddedNode"]
    assert man == san
    for i in man:
        assert set(mine["features"][i]["properties"]) == {
            "CATS_ID", "Type", "kV", "Comments"}


@needs_gis
def test_roundtrip_gen_matches_shipped(roundtrip):
    mine = json.load(open(roundtrip["gen"], encoding="utf-8"))
    ship = json.load(open(FINAL / "gen.json", encoding="utf-8"))
    mg = {f["id"]: f for f in mine["features"]}
    sg = {f["id"]: f for f in ship["features"]}
    assert set(mg) == set(sg) and len(sg) == 2149
    bad = [k for k in sg
           if not _props_equal(mg[k]["properties"], sg[k]["properties"])]
    assert not bad, f"{len(bad)} gens differ, first {bad[:3]}"


def test_added_node_props_preserved_by_default():
    """DV-07d approved: the default AUGMENTS added-node properties instead
    of replacing them (legacy replace stays behind the flag)."""
    import numpy as np_
    from gridweld.io.geojson import _reformat_buses
    from gridweld.io.matpower import MatpowerCase
    import pandas as pd_

    case = MatpowerCase(name="t", version="2", base_mva=1.0)
    case.arrays = {"bus": np_.array([[7, 1, 0, 0, 0, 0, 1, 1, 0, 115, 1,
                                      1.1, 0.9]])}
    case.cells["bus_data"] = pd_.DataFrame(
        {"Lon": [-120.0], "Lat": [37.0], "FID": [0.0],
         "Type": ["AddedNode"]})
    layer = {"features": [{"id": 0, "type": "Feature",
                           "geometry": {"type": "Point",
                                        "coordinates": [-120.0, 37.0]},
                           "properties": {"orig": "kept", "FID": 0}}]}
    cfg = WeldConfig()
    out = _reformat_buses(layer, case, cfg)
    props = out["features"][0]["properties"]
    assert props["orig"] == "kept" and props["CATS_ID"] == "7"
    cfg.export.gis.preserve_added_node_props = False       # legacy mode
    layer["features"][0]["properties"] = {"orig": "kept", "FID": 0}
    out = _reformat_buses(layer, case, cfg)
    assert "orig" not in out["features"][0]["properties"]


# ---------------------------------------------------------------------------
# QA reports
# ---------------------------------------------------------------------------


def test_qa_reports(tmp_path):
    from gridweld.model import WeldModel
    from gridweld.viz.plots import write_qa_reports

    model = WeldModel(
        buses=pd.DataFrame({"bus_id": range(4)}),
        branches=pd.DataFrame({
            "from_bus": [0, 1, 2, 0, 1], "to_bus": [1, 1, 3, 1, 0],
            "lat1": [1.0, 2.0, 3.0, 1.0, 1.0],
            "lat2": [1.5, 2.0, 3.5, 1.5, 1.5],
            "lon1": [1.0, 2.0, 3.0, 1.0, 1.0],
            "lon2": [1.5, 2.0, 3.5, 1.5, 1.5],
            "is_transformer": [False, False, True, False, False],
            "tap": [0.0, 0.0, 1.0, 0.0, 0.0],
            "f_bus_distance_km": [0.1, 20.0, np.nan, 0.2, 0.1],
            "t_bus_distance_km": [0.1, 0.1, np.nan, 0.2, 0.1]}),
        generators=pd.DataFrame({
            "gen_id": [0, 1], "bus": [0, 1],
            "gen_bus_distance_km": [1.0, 30.0]}),
        loads=pd.DataFrame({
            "load_id": [0, 1, 2], "bus": [1, 1, 3],
            "pd_mw": [5.0, 7.0, 2.0], "qd_mvar": [1.0, 1.0, 0.5]}))
    paths = write_qa_reports(model, tmp_path, WeldConfig())
    self_loops = pd.read_csv(paths["self_loops"])
    assert len(self_loops) == 1                    # branch 1→1
    tiny = pd.read_csv(paths["tiny_lines"])
    assert len(tiny) == 1                          # identical endpoints
    assert len(pd.read_csv(paths["transformers"])) == 1
    assert len(pd.read_csv(paths["long_distance_generators"])) == 1
    assert len(pd.read_csv(paths["long_distance_transmission"])) == 1
    dup = pd.read_csv(paths["transmission_same_endpoints"])
    assert len(dup) == 3                           # 0↔1 three times, non-XF
    loads = pd.read_csv(paths["grid_loads"])
    assert loads.loc[loads["bus"] == 1, "pd_mw"].iloc[0] == 12.0
    assert paths["bus_loads_html"].exists()
