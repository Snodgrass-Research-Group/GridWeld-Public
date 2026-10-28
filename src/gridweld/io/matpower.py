"""MATPOWER case ⇄ WeldModel bridge (S2 output → S3 input; final export).

Origin files (algorithm docs 03 §ReadMatpower/Matpower2Table/Table2Matpower,
02 §I/O contract, 07):
* ``legacy/cal_largest_grid.m`` — PowerModels ``export_file`` output: MATPOWER
  case v2 plus nonstandard ``%column_names%`` cell tables (``bus_data``,
  ``gen_data``, ``branch_data``, ``load_data``) carrying GIS provenance.
* ``legacy/Jonathan_Code/Matlab Functions/ReadMatpower.m`` — merges the GIS
  branch table with the MATPOWER case topology (FID join) and builds the
  Bus/Branch/Gen working tables. Ported as :func:`merge_topology`.
* ``Matpower2Table.m`` / ``Table2Matpower.m`` — plain extraction / rebuild;
  subsumed by :func:`case_to_model` / :func:`model_to_case` (DV-P3-8: the
  ``savecase`` side effect and the undefined-``Sub`` branch are not ported;
  writing is explicit via :func:`write_matpower_case`).

Reuse (D5, REUSE-WITH-EXTENSION): numeric-block parsing follows
``gridcrucible.io.matlab_import._parse_matpower_case``; extended for the
``*_data`` cell blocks and the FID join + bus-type repair. No MATPOWER
toolbox at runtime: ``find_islands`` → scipy ``connected_components``;
``ext2int`` → the D6 internal-id mapping.

Deviations: DV-P2a-1 (fixed cell-table column order on export), DV-P3-1
(``matlab_compat`` reproduces the Bus.MaxMW/Pmin overwrite), DV-P3-8.

Units: bus Pd/Qd in MW/MVAr (baseMVA=1 & per_unit=true at S2 — numerically
identical to pu; never rescaled here); impedances per-unit; kV in kilovolts;
cell-table distances in km (haversine, R = 6371 km).

ext2int semantics (verified empirically against
``validation/fixtures/matlab/work/SynGrid.mat`` → ``mpc.order.ext``):
buses renumber consecutively in bus-table ROW order; generator and branch
row ORDER is preserved (endpoint ids remapped only); isolated (type-4)
buses and their gens/branches are dropped.

Phase: P2a.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..model import WeldModel

# MATPOWER column indices (0-based; define_constants equivalents).
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = range(11)
MODEL, STARTUP, SHUTDOWN, NCOST, COST = 0, 1, 2, 3, 4   # gencost; COST = c2 (descending)

#: Fixed cell-table column order on export (DV-P2a-1). Readers key on the
#: ``%column_names%`` header, never on position.
_CELL_EXPORT_ORDER = {
    "bus_data": ["Lon", "Import", "Lat", "Type", "FID"],
    "gen_data": ["PlantCode", "FuelType", "Lon", "GenID", "Lat", "gen_bus_distance"],
    "branch_data": ["t_bus_distance", "Lon2", "Lon1", "kV", "FID",
                    "f_bus_distance", "Lat1", "Lat2"],
    "load_data": ["Lon", "Lat", "load_bus_distance"],
}


@dataclass
class MatpowerCase:
    """Parsed MATPOWER case: numeric blocks + ``%column_names%`` cell tables.

    ``arrays`` holds every numeric ``mpc.<name> = [...]`` block (bus, gen,
    branch, gencost, ...); ``cells`` every ``mpc.<name> = {...}`` block as a
    DataFrame named by its ``%column_names%`` header.
    """

    name: str = "mpc"
    version: str = "2"
    base_mva: float = 100.0
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    cells: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def bus(self) -> np.ndarray:
        return self.arrays["bus"]

    @property
    def gen(self) -> np.ndarray:
        return self.arrays["gen"]

    @property
    def branch(self) -> np.ndarray:
        return self.arrays["branch"]

    @property
    def gencost(self) -> np.ndarray | None:
        return self.arrays.get("gencost")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUM_BLOCK = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.S)
_CELL_BLOCK = re.compile(
    r"(?:%column_names%[ \t]*([^\r\n]*)\r?\n)?mpc\.(\w+)\s*=\s*\{(.*?)\};", re.S)
_SCALAR = re.compile(r"mpc\.(\w+)\s*=\s*([0-9.eE+-]+)\s*;")
_STRING = re.compile(r"mpc\.(\w+)\s*=\s*'([^']*)'\s*;")
_FUNC = re.compile(r"function\s+mpc\s*=\s*(\w+)")


def _parse_matrix(body: str) -> np.ndarray:
    rows = []
    for line in body.splitlines():
        line = line.split("%")[0].strip().rstrip(";")
        if not line:
            continue
        rows.append([float(tok) for tok in line.split()])
    if not rows:
        return np.empty((0, 0))
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _parse_cell_token(tok: str) -> Any:
    tok = tok.strip()
    if tok.startswith("'") and tok.endswith("'"):
        return tok[1:-1]
    if tok in ("NaN", "nan"):
        return np.nan
    try:
        return float(tok)
    except ValueError:
        return tok


def _parse_cell(body: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        rows.append([_parse_cell_token(t) for t in line.split("\t") if t.strip() != ""])
    df = pd.DataFrame(rows)
    if columns and len(columns) == df.shape[1]:
        df.columns = columns
    return df


def read_matpower_case(path: str | Path) -> MatpowerCase:
    """Parse a MATPOWER case script, including the PowerModels cell tables."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    case = MatpowerCase()
    m = _FUNC.search(text)
    if m:
        case.name = m.group(1)
    for name, val in _STRING.findall(text):
        if name == "version":
            case.version = val
    for name, val in _SCALAR.findall(text):
        if name == "baseMVA":
            case.base_mva = float(val)
    for m in _NUM_BLOCK.finditer(text):
        try:
            arr = _parse_matrix(m.group(2))
        except ValueError:
            continue     # not a plain numeric block
        if arr.size:
            case.arrays[m.group(1)] = arr
    for m in _CELL_BLOCK.finditer(text):
        header, name, body = m.group(1), m.group(2), m.group(3)
        case.cells[name] = _parse_cell(body, header.split() if header else [])
    for req in ("bus", "branch", "gen"):
        if req not in case.arrays:
            raise ValueError(f"{path}: no numeric mpc.{req} block found")
    return case


# ---------------------------------------------------------------------------
# Case → WeldModel
# ---------------------------------------------------------------------------

def case_to_model(case: MatpowerCase) -> WeldModel:
    """MATPOWER case → WeldModel (0-based internal ids, D6).

    Loads are FOLDED into bus pd/qd in the case (MATPOWER has no load
    table); ``load_data`` carries only Lon/Lat/distance, so ``model.loads``
    stays empty and the cell table rides in ``meta['load_data']`` for
    round-trip fidelity.
    """
    bus = case.bus
    n = len(bus)
    ext2int = {int(b): i for i, b in enumerate(bus[:, BUS_I])}

    buses = pd.DataFrame({
        "bus_id": np.arange(n, dtype=np.int64),
        "external_num": bus[:, BUS_I].astype(np.int64),
        "base_kv": bus[:, BASE_KV],
        "bus_type": bus[:, BUS_TYPE].astype(np.int64),
        "pd_mw": bus[:, PD],
        "qd_mvar": bus[:, QD],
        "vm_pu": bus[:, VM],
        "va_deg": bus[:, VA],
    })
    bus_data = case.cells.get("bus_data")
    if bus_data is not None:
        if len(bus_data) != n:
            raise ValueError(f"bus_data rows {len(bus_data)} != buses {n}")
        buses["fid"] = pd.to_numeric(bus_data["FID"], errors="coerce")
        buses["lat"] = bus_data["Lat"].astype(float)
        buses["lon"] = bus_data["Lon"].astype(float)
        buses["node_type"] = bus_data["Type"].astype(str)
        buses["import_flag"] = bus_data["Import"].astype(str).str.strip() == "IMPORT"

    gen = case.gen
    generators = pd.DataFrame({
        "gen_id": np.arange(len(gen), dtype=np.int64),
        "bus": np.array([ext2int[int(b)] for b in gen[:, GEN_BUS]], dtype=np.int64),
        "pmax_mw": gen[:, PMAX],
        "pmin_mw": gen[:, PMIN] if gen.shape[1] > PMIN else 0.0,
        "qmax_mvar": gen[:, QMAX],
        "qmin_mvar": gen[:, QMIN],
        "vg_pu": gen[:, VG],
        "status": gen[:, GEN_STATUS].astype(np.int64),
    })
    gen_data = case.cells.get("gen_data")
    if gen_data is not None:
        generators["plant_code"] = pd.to_numeric(gen_data["PlantCode"], errors="coerce")
        generators["fuel_type"] = gen_data["FuelType"].astype(str)
        generators["eia_gen_id"] = gen_data["GenID"].astype(str)
        generators["lat"] = gen_data["Lat"].astype(float)
        generators["lon"] = gen_data["Lon"].astype(float)
        generators["gen_bus_distance_km"] = pd.to_numeric(
            gen_data["gen_bus_distance"], errors="coerce")
    gencost = case.gencost
    if gencost is not None and len(gencost) == len(gen):
        generators["cost_model"] = gencost[:, MODEL].astype(np.int64)
        generators["startup_usd"] = gencost[:, STARTUP]
        generators["shutdown_usd"] = gencost[:, SHUTDOWN]
        generators["ncost"] = gencost[:, NCOST].astype(np.int64)
        generators["c2"] = gencost[:, COST]
        generators["c1"] = gencost[:, COST + 1]
        generators["c0"] = gencost[:, COST + 2]

    br = case.branch
    branches = pd.DataFrame({
        "from_bus": np.array([ext2int[int(b)] for b in br[:, F_BUS]], dtype=np.int64),
        "to_bus": np.array([ext2int[int(b)] for b in br[:, T_BUS]], dtype=np.int64),
        "r_pu": br[:, BR_R],
        "x_pu": br[:, BR_X],
        "b_pu": br[:, BR_B],
        "capacity_mva": br[:, RATE_A],
        "tap": br[:, TAP],
        "shift": br[:, SHIFT],
        "status": br[:, BR_STATUS].astype(np.int64),
    })
    branches["is_transformer"] = branches["tap"].to_numpy() != 0
    branch_data = case.cells.get("branch_data")
    if branch_data is not None:
        branches["fid"] = pd.to_numeric(branch_data["FID"], errors="coerce")
        for src, dst in [("Lat1", "lat1"), ("Lon1", "lon1"),
                         ("Lat2", "lat2"), ("Lon2", "lon2")]:
            branches[dst] = pd.to_numeric(branch_data[src], errors="coerce")
        kv = pd.to_numeric(branch_data["kV"], errors="coerce")
        branches["voltage_kv_from"] = kv
        branches["voltage_kv_to"] = kv
        branches["f_bus_distance_km"] = pd.to_numeric(
            branch_data["f_bus_distance"], errors="coerce")
        branches["t_bus_distance_km"] = pd.to_numeric(
            branch_data["t_bus_distance"], errors="coerce")

    meta: dict[str, Any] = {
        "name": case.name, "version": case.version, "base_mva": case.base_mva,
        "per_unit": True,
    }
    if "load_data" in case.cells:
        meta["load_data"] = case.cells["load_data"]
    return WeldModel(buses=buses, branches=branches, generators=generators,
                     meta=meta)


# ---------------------------------------------------------------------------
# WeldModel → case + writer
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64)
    return np.full(len(df), default, dtype=np.float64)


def model_to_case(model: WeldModel) -> MatpowerCase:
    """WeldModel → MATPOWER case arrays. External bus numbers come from
    ``buses.external_num`` (falling back to ``bus_id + 1``); internal
    0-based ids appear nowhere in the output (D6 — the ``CreateBranch.m``
    int→ext shim is exactly this mapping, doc 04 §11)."""
    buses, branches, gens = model.buses, model.branches, model.generators
    n = len(buses)
    if "external_num" in buses.columns and buses["external_num"].notna().all():
        ext = buses["external_num"].to_numpy(dtype=np.int64)
    else:
        ext = buses["bus_id"].to_numpy(dtype=np.int64) + 1

    bus = np.zeros((n, 13))
    bus[:, BUS_I] = ext
    bus[:, BUS_TYPE] = _col(buses, "bus_type", 1)
    bus[:, PD] = _col(buses, "pd_mw", 0.0)
    bus[:, QD] = _col(buses, "qd_mvar", 0.0)
    bus[:, VM] = _col(buses, "vm_pu", 0.0)
    bus[:, VA] = _col(buses, "va_deg", 0.0)
    bus[:, BASE_KV] = _col(buses, "base_kv", 0.0)

    g = np.zeros((len(gens), 10))
    g[:, GEN_BUS] = ext[gens["bus"].to_numpy(dtype=np.int64)]
    g[:, QMAX] = _col(gens, "qmax_mvar", 0.0)
    g[:, QMIN] = _col(gens, "qmin_mvar", 0.0)
    g[:, VG] = _col(gens, "vg_pu", 0.0)
    g[:, GEN_STATUS] = _col(gens, "status", 1)
    g[:, PMAX] = _col(gens, "pmax_mw", 0.0)
    g[:, PMIN] = _col(gens, "pmin_mw", 0.0)

    b = np.zeros((len(branches), 13))
    b[:, F_BUS] = ext[branches["from_bus"].to_numpy(dtype=np.int64)]
    b[:, T_BUS] = ext[branches["to_bus"].to_numpy(dtype=np.int64)]
    b[:, BR_R] = _col(branches, "r_pu", 0.0)
    b[:, BR_X] = _col(branches, "x_pu", 0.0)
    b[:, BR_B] = _col(branches, "b_pu", np.nan)
    b[:, RATE_A] = _col(branches, "capacity_mva", 0.0)
    b[:, TAP] = _col(branches, "tap", 0.0)
    b[:, SHIFT] = _col(branches, "shift", 0.0)
    b[:, BR_STATUS] = _col(branches, "status", 1)

    case = MatpowerCase(name=str(model.meta.get("name", "mpc")),
                        version=str(model.meta.get("version", "2")),
                        base_mva=float(model.meta.get("base_mva", 100.0)))
    case.arrays = {"bus": bus, "gen": g, "branch": b}
    if "c2" in gens.columns:
        gc = np.zeros((len(gens), 7))
        gc[:, MODEL] = _col(gens, "cost_model", 2)
        gc[:, STARTUP] = _col(gens, "startup_usd", 0.0)
        gc[:, SHUTDOWN] = _col(gens, "shutdown_usd", 0.0)
        gc[:, NCOST] = _col(gens, "ncost", 3)
        gc[:, COST] = gens["c2"].to_numpy(dtype=np.float64)
        gc[:, COST + 1] = gens["c1"].to_numpy(dtype=np.float64)
        gc[:, COST + 2] = gens["c0"].to_numpy(dtype=np.float64)
        case.arrays["gencost"] = gc

    # Provenance cell tables in fixed column order (DV-P2a-1).
    if "lat" in buses.columns:
        case.cells["bus_data"] = pd.DataFrame({
            "Lon": buses["lon"].to_numpy(),
            "Import": np.where(buses["import_flag"].to_numpy(dtype=bool)
                               if "import_flag" in buses.columns else False,
                               "IMPORT", " "),
            "Lat": buses["lat"].to_numpy(),
            "Type": buses.get("node_type", pd.Series([""] * n)).to_numpy(),
            "FID": buses.get("fid", pd.Series([np.nan] * n)).to_numpy(),
        })
    if "lat" in gens.columns:
        case.cells["gen_data"] = pd.DataFrame({
            "PlantCode": gens.get("plant_code", np.nan),
            "FuelType": gens.get("fuel_type", ""),
            "Lon": gens["lon"].to_numpy(),
            "GenID": gens.get("eia_gen_id", ""),
            "Lat": gens["lat"].to_numpy(),
            "gen_bus_distance": gens.get("gen_bus_distance_km", np.nan),
        })
    if "lat1" in branches.columns:
        case.cells["branch_data"] = pd.DataFrame({
            "t_bus_distance": branches.get("t_bus_distance_km", np.nan),
            "Lon2": branches["lon2"].to_numpy(),
            "Lon1": branches["lon1"].to_numpy(),
            "kV": branches.get("voltage_kv_from", np.nan),
            "FID": branches.get("fid", np.nan),
            "f_bus_distance": branches.get("f_bus_distance_km", np.nan),
            "Lat1": branches["lat1"].to_numpy(),
            "Lat2": branches["lat2"].to_numpy(),
        })
    if isinstance(model.meta.get("load_data"), pd.DataFrame):
        case.cells["load_data"] = model.meta["load_data"]
    return case


def _fmt(x: Any) -> str:
    if isinstance(x, str):
        return f"'{x}'"
    xf = float(x)
    if np.isnan(xf):
        return "NaN"
    return repr(xf)


def write_matpower_case(case: MatpowerCase | WeldModel, path: str | Path) -> Path:
    """Write a MATPOWER case v2 script (PowerModels layout). Pure writer —
    the legacy ``savecase``-inside-a-converter side effect is not ported
    (DV-P3-8)."""
    if isinstance(case, WeldModel):
        case = model_to_case(case)
    path = Path(path)
    lines = ["%% MATPOWER Case Format : Version 2",
             f"function mpc = {case.name}",
             f"mpc.version = '{case.version}';", "",
             f"mpc.baseMVA = {_fmt(case.base_mva)};"]
    for name in ("bus", "gen", "branch", "gencost"):
        arr = case.arrays.get(name)
        if arr is None:
            continue
        lines.append(f"mpc.{name} = [")
        for row in arr:
            lines.append("\t" + "\t".join(_fmt(v) for v in row))
        lines.append("];\n")
    for name, order in _CELL_EXPORT_ORDER.items():
        df = case.cells.get(name)
        if df is None:
            continue
        cols = ([c for c in order if c in df.columns]
                + [c for c in df.columns if c not in order])
        lines.append("%column_names% " + " ".join(cols) + " ")
        lines.append(f"mpc.{name} = {{")
        for row in df[cols].itertuples(index=False):
            lines.append("\t" + "\t".join(_fmt(v) for v in row))
        lines.append("};\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ReadMatpower port: topology merge
# ---------------------------------------------------------------------------

def snap_to_levels(values: np.ndarray, levels: list[int]) -> np.ndarray:
    """Nearest voltage class; ties → FIRST (lowest) level (MATLAB ``min`` /
    Julia ``findmin`` ≡ ``np.argmin`` first occurrence). Shared by the S2
    kV remap (doc 02) and the S3 bus/transformer snap (doc 03)."""
    lv = np.asarray(levels, dtype=float)
    idx = np.abs(np.asarray(values, dtype=float)[:, None] - lv[None, :]).argmin(axis=1)
    return lv[idx]


def merge_topology(branch_df: pd.DataFrame, case: MatpowerCase, cfg,
                   *, matlab_compat: bool = True
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """ReadMatpower.m port: merge the GIS branch table with the MATPOWER case.

    Parameters
    ----------
    branch_df:
        Parameterized GIS branch table (model-style column names for the
        MATLAB ``Branch`` schema: ``fid``, ``voltage_kv_from/to`` [kV],
        ``length_mi`` [mi], ``r_pu/x_pu/b_pu``, ``capacity_mva`` [MVA],
        ``n_circuits``, ``n_cond_per_phase``, ``gmd_ft`` [ft], ...).
        Not mutated; a merged copy is returned.
    case:
        Parsed S2 MATPOWER case (``cal_largest_grid``-shaped with
        ``bus_data``/``gen_data``/``branch_data`` cell tables).
    cfg:
        WeldConfig — uses ``system.voltage_levels``, ``system.region``,
        ``io.default_line_rxb``, ``io.default_xf_rxb``, ``io.dummy_cost``.
    matlab_compat:
        True (default) reproduces DV-P3-1 (bus ``max_mw`` overwritten by
        fleet Pmin where nonzero; ``min_mw`` not stored). False stores both.

    Returns
    -------
    ``(bus, branch, gen, params)`` — tables in model-style column names.
    ``bus_num``/``from_bus``/``to_bus``/``bus`` are **1-based internal**
    MATPOWER ids at this I/O boundary (callers building a WeldModel
    subtract 1); ``params`` mirrors MATLAB's params fields.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    levels = list(cfg.system.voltage_levels)
    bus = case.bus.copy()
    gen = case.gen.copy()
    branch = case.branch.copy()
    gencost = None if case.gencost is None else case.gencost.copy()

    # --- bus-type repair [ReadMatpower.m:14-33] ---
    if (bus[:, BUS_TYPE] >= 4).any():
        bus[:, BUS_TYPE] = 1
    if not (bus[:, BUS_TYPE] == 2).any():
        bus[np.isin(bus[:, BUS_I], np.unique(gen[:, GEN_BUS])), BUS_TYPE] = 2
    if not (bus[:, BUS_TYPE] == 3).any():
        max_gen = int(np.argmax(gen[:, PMAX]))
        bus[bus[:, BUS_I] == gen[max_gen, GEN_BUS], BUS_TYPE] = 3
    if gencost is None:
        gencost = np.zeros((len(gen), 7))
        gencost[:, MODEL] = 2
        gencost[:, NCOST] = 3
        gencost[:, COST:COST + 3] = cfg.io.dummy_cost

    # --- islands → drop non-largest [ReadMatpower.m:35-39] ---
    n = len(bus)
    ext2row = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    f_rows = np.array([ext2row[int(b)] for b in branch[:, F_BUS]])
    t_rows = np.array([ext2row[int(b)] for b in branch[:, T_BUS]])
    adj = sp.csr_matrix((np.ones(len(f_rows)), (f_rows, t_rows)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    keep_rows = labels == int(np.argmax(np.bincount(labels)))

    # default impedances on the EXTERNAL branch table [ReadMatpower.m:43-49]
    xf_mask = branch[:, TAP] > 0
    branch[:, BR_R], branch[:, BR_X], branch[:, BR_B] = cfg.io.default_line_rxb
    branch[xf_mask, BR_R], branch[xf_mask, BR_X], branch[xf_mask, BR_B] = \
        cfg.io.default_xf_rxb

    # ext2int: drop isolated buses (+ their gens/branches); renumber to row
    # order; gen/branch row order preserved (verified vs SynGrid.mat).
    kept_bus = bus[keep_rows]
    n2i = {int(b): i + 1 for i, b in enumerate(kept_bus[:, BUS_I])}   # 1-based
    gen_keep = np.array([int(b) in n2i for b in gen[:, GEN_BUS]])
    br_keep = np.array([int(f) in n2i and int(t) in n2i
                        for f, t in zip(branch[:, F_BUS], branch[:, T_BUS])])
    mpc_bus = kept_bus.copy()
    mpc_bus[:, BUS_I] = np.arange(1, len(kept_bus) + 1)
    mpc_gen = gen[gen_keep].copy()
    mpc_gen[:, GEN_BUS] = [n2i[int(b)] for b in mpc_gen[:, GEN_BUS]]
    mpc_branch = branch[br_keep].copy()
    mpc_branch[:, F_BUS] = [n2i[int(b)] for b in mpc_branch[:, F_BUS]]
    mpc_branch[:, T_BUS] = [n2i[int(b)] for b in mpc_branch[:, T_BUS]]
    mpc_gencost = gencost[gen_keep]
    bus_data = case.cells["bus_data"].loc[keep_rows].reset_index(drop=True)
    gen_data = case.cells["gen_data"].loc[gen_keep].reset_index(drop=True)
    branch_data = case.cells["branch_data"].loc[br_keep].reset_index(drop=True)

    # --- FID join [ReadMatpower.m:53-92] ---
    out = branch_df.copy()
    mpfid = pd.to_numeric(branch_data["FID"], errors="coerce").to_numpy(dtype=float)
    fresh = np.isnan(mpfid)
    if fresh.any():
        start = np.nanmax(np.concatenate([out["fid"].to_numpy(dtype=float), mpfid]))
        mpfid[fresh] = start + np.arange(1, fresh.sum() + 1)

    out["from_bus"] = 0
    out["to_bus"] = 0
    out["tap"] = 0.0
    fid_to_row = pd.Series(np.arange(len(mpfid)), index=mpfid.astype(np.int64))
    dup = fid_to_row.index.duplicated()
    if dup.any():
        raise ValueError("duplicate FIDs in MATPOWER branch_data: "
                         f"{fid_to_row.index[dup].tolist()[:10]}")
    gis_fids = out["fid"].to_numpy(dtype=np.int64)
    hit = fid_to_row.reindex(gis_fids)
    matched = hit.notna().to_numpy()
    rows = hit.to_numpy(dtype=float)[matched].astype(np.int64)
    out.loc[matched, "from_bus"] = mpc_branch[rows, F_BUS].astype(np.int64)
    out.loc[matched, "to_bus"] = mpc_branch[rows, T_BUS].astype(np.int64)
    out.loc[matched, "tap"] = mpc_branch[rows, TAP]

    # MATPOWER-only FIDs with TAP != 0 appended (setdiff → ascending FID)
    mp_only = np.setdiff1d(mpfid.astype(np.int64), gis_fids)          # sorted
    mp_rows = fid_to_row.loc[mp_only].to_numpy()
    add = mp_rows[mpc_branch[mp_rows, TAP] != 0]
    appended = pd.DataFrame({
        "fid": mpfid[add].astype(np.int64),
        "from_bus": mpc_branch[add, F_BUS].astype(np.int64),
        "to_bus": mpc_branch[add, T_BUS].astype(np.int64),
        "tap": mpc_branch[add, TAP],
        "voltage_kv_from": mpc_bus[mpc_branch[add, F_BUS].astype(np.int64) - 1,
                                   BASE_KV],
        "voltage_kv_to": mpc_bus[mpc_branch[add, T_BUS].astype(np.int64) - 1,
                                 BASE_KV],
        "length_mi": 0.0,
        "n_cond_per_phase": 1.0,
        "n_circuits": 1.0,
        "gmd_ft": 0.0,
    })
    out = pd.concat([out, appended], ignore_index=True)

    # --- transformer bus-tie defaults [ReadMatpower.m:94-113] ---
    xf = out["tap"].to_numpy() > 0
    out.loc[xf, "structure"] = "Transformer"
    out.loc[xf, "material"] = "Transformer"
    out.loc[xf, "capacity_mva"] = 0.0
    out.loc[xf, ["r_pu", "x_pu", "b_pu"]] = cfg.io.default_xf_rxb
    out.loc[xf, "n_cond_per_phase"] = 1.0
    out.loc[xf, "resize"] = 1.0
    out.loc[xf, "conductor_idx"] = 0.0
    out.loc[xf, "max_conductor_idx"] = 50.0      # inverted sentinels kept
    out.loc[xf, "min_conductor_idx"] = 2000.0    # as-is (quirk c, doc 03)
    out.loc[xf, "voltage_kv_from"] = snap_to_levels(
        out.loc[xf, "voltage_kv_from"].to_numpy(), levels)
    out.loc[xf, "voltage_kv_to"] = snap_to_levels(
        out.loc[xf, "voltage_kv_to"].to_numpy(), levels)

    # --- gen table [ReadMatpower.m:115-151] ---
    gen_bus_int = mpc_gen[:, GEN_BUS].astype(np.int64)
    gdf = pd.DataFrame({"bus": gen_bus_int})
    gdf["unit_id"] = gdf.groupby("bus").cumcount() + 1
    gdf["pmin_mw"] = mpc_gen[:, PMIN]
    gdf["pmax_mw"] = mpc_gen[:, PMAX]
    gdf["qmin_mvar"] = mpc_gen[:, QMIN]
    gdf["qmax_mvar"] = mpc_gen[:, QMAX]
    gdf["vg_pu"] = 1.0                            # forced [ReadMatpower.m:130]
    gdf["fuel_type"] = gen_data["FuelType"].astype(str).to_numpy()
    gdf["plant_code"] = pd.to_numeric(gen_data["PlantCode"],
                                      errors="coerce").to_numpy()
    if "GenID" in gen_data.columns:       # EIA unit id (assign_gen_limits, P5)
        gdf["eia_gen_id"] = gen_data["GenID"].astype(str).to_numpy()
    gdf["region"] = cfg.system.region
    gdf["c0"] = mpc_gencost[:, COST + 2]          # Gen.A [ReadMatpower.m:144]
    gdf["c1"] = mpc_gencost[:, COST + 1]          # Gen.B
    gdf["c2"] = mpc_gencost[:, COST]              # Gen.C

    # --- bus table [ReadMatpower.m:153-189] ---
    bdf = pd.DataFrame({"bus_num": mpc_bus[:, BUS_I].astype(np.int64)})
    if len(mpc_bus) == len(bus):
        bdf["external_num"] = bus[:, BUS_I].astype(np.int64)
    bdf["lat"] = bus_data["Lat"].astype(float).to_numpy()
    bdf["lon"] = bus_data["Lon"].astype(float).to_numpy()
    bdf["fid"] = pd.to_numeric(bus_data["FID"], errors="coerce").to_numpy()
    bdf["node_type"] = bus_data["Type"].astype(str).to_numpy()
    if "Import" in bus_data.columns:      # 'IMPORT'/' ' strings (P5 exports)
        bdf["import"] = bus_data["Import"].astype(str).to_numpy()
    bdf["bus_type"] = mpc_bus[:, BUS_TYPE].astype(np.int64)
    bdf["pd_mw"] = mpc_bus[:, PD]
    bdf["qd_mvar"] = mpc_bus[:, QD]
    bdf["base_kv"] = mpc_bus[:, BASE_KV]
    bdf["kv_class"] = snap_to_levels(mpc_bus[:, BASE_KV], levels)
    bdf["vm_pu"] = mpc_bus[:, VM]
    bdf["va_deg"] = mpc_bus[:, VA]

    gen_max = np.zeros(len(bdf))
    gen_min = np.zeros(len(bdf))
    np.add.at(gen_max, gen_bus_int - 1, mpc_gen[:, PMAX])
    np.add.at(gen_min, gen_bus_int - 1, mpc_gen[:, PMIN])
    max_mw = np.zeros(len(bdf))
    max_mw[gen_max != 0] = gen_max[gen_max != 0]
    if matlab_compat:
        # DV-P3-1: fleet Pmin overwrites max_mw where nonzero; min_mw dropped
        max_mw[gen_min != 0] = gen_min[gen_min != 0]
    else:
        bdf["min_mw"] = gen_min
    bdf["max_mw"] = max_mw
    bdf["region"] = cfg.system.region

    # --- drop unmatched GIS rows + params [ReadMatpower.m:192-201] ---
    out = out[(out["from_bus"] != 0) & (out["to_bus"] != 0)].reset_index(drop=True)
    params = {
        "buses": len(np.unique(np.concatenate(
            [out["from_bus"].to_numpy(), out["to_bus"].to_numpy()]))),
    }
    params["nodes"] = params["buses"]
    params["total_paths"] = params["total_lines"] = len(out)
    return bdf, out, gdf, params
