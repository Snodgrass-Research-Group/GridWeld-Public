"""GeoJSON round-trip: final electrical parameters back into the GIS layers.

Origin: ``prototyping_code/reformat_GIS.jl`` (Sofia Taylor, Feb 2023) —
algorithm doc 07. Writes CATS_ID (MATPOWER branch/bus/gen ids as STRINGS,
the PowerModels dict keys) + electrical parameters from the FINAL case
into the lines/added-nodes layers, appends transformer LineStrings and
substation Points, builds a fresh generator FeatureCollection →
``{tl,bus,gen}.json`` for ArcGIS reload.

Key linkage (asserted like the Julia original): branch/bus ``FID`` ↔
feature index ``FID`` (0-based list; Julia's ``FID+1``) ↔ ``CATS_ID``.
Electrical values come from the FINAL case positionally: branch row i of
``case`` ↔ row i of ``final_case`` (the Julia script's unverified TODO —
enforced here by a row-count assert).

Ported fixes / deviations:
* DV-07b — legacy transformer geometries were ``[[Lon1,Lon2],[Lon2,Lat2]]``
  (shipped tl.json transformer coords are wrong); this port emits
  ``[[Lon1,Lat1],[Lon2,Lat2]]``.
* DV-07d — the AddedNode property dict is REPLACED (legacy default);
  ``export.gis.preserve_added_node_props=True`` keeps the originals.
* DV-07f — deterministic ascending-id iteration replaces Julia Dict-hash
  order: appended-feature order/ids differ from the shipped files
  (validation keys on CATS_ID).
* The loaded-then-shadowed ``substations.geojson`` layer (Julia bug: the
  loop variable shadows it; the layer is never used) is NOT loaded.
* The unreachable ``println["WARNING"]`` square-bracket bug becomes a
  raised ``WeldError`` (a bus Type outside {AddedNode, Substation} would
  have crashed Julia too).

Gen properties replicate the PowerModels parse_file dict (pg/qg/limits/
model/startup/shutdown/ncost/cost list/index/source_id + the cell-table
extras). ⚠ Only valid at ``baseMVA == 1`` (the S2 export convention,
``system.build_base_mva`` — asserted): PowerModels' per-unit conversion is
then the identity, so no pu scaling is replicated.

Uses stdlib json (allow_nan keeps Julia's NaN literals round-trippable).
Units: Length_km from Length_Fee ft × 0.0003048; kV as-is; MW at baseMVA=1.
Phase: P5. Validated vs legacy/Final_Grid/{tl,bus,gen}.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import WeldError
from .matpower import MatpowerCase, read_matpower_case

log = logging.getLogger(__name__)

#: mpc.branch column positions used here (MATPOWER v2).
_F_BUS, _T_BUS, _BR_R, _BR_X, _BR_B, _RATE_A, _TAP = 0, 1, 2, 3, 4, 5, 8
#: mpc.gen / mpc.gencost positions.
_GEN_BUS, _PG, _QG, _QMAX, _QMIN, _VG, _MBASE, _GEN_STATUS, _PMAX, _PMIN = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
_MODEL, _STARTUP, _SHUTDOWN, _NCOST, _COST = 0, 1, 2, 3, 4


def _cell_value(cells: pd.DataFrame, col: str, i: int):
    v = cells[col].iloc[i]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _reformat_lines(tl_layer: dict, case: MatpowerCase,
                    final_case: MatpowerCase, tl_extra: pd.DataFrame,
                    cfg: WeldConfig) -> dict:
    """The branch loop [reformat_GIS.jl:54-117] in ascending branch id."""
    gis = cfg.export.gis
    feats = tl_layer["features"]
    br_cells = case.cells["branch_data"]
    fids = pd.to_numeric(br_cells["FID"], errors="coerce").to_numpy(float)
    fb = final_case.branch
    if len(fb) != len(case.branch):
        raise WeldError(
            f"final case has {len(fb)} branches, build case {len(case.branch)}"
            " — the positional branchid linkage the Julia script assumed "
            "does not hold")
    extra = tl_extra.drop_duplicates("FID", keep="first").set_index("FID")

    for i in range(len(case.branch)):
        branchid = str(i + 1)                       # PowerModels dict key
        electrical = {
            "kV": _cell_value(br_cells, "kV", i),
            "f_bus": int(fb[i, _F_BUS]),
            "t_bus": int(fb[i, _T_BUS]),
            "br_x": float(fb[i, _BR_X]),
            "br_r": float(fb[i, _BR_R]),
            "br_b": float(fb[i, _BR_B]) + 0.0,      # b_to + b_fr ≡ B split
            "transformer": bool(case.branch[i, _TAP] != 0),
            # PowerModels stores rates in per-unit: MVA / the FINAL case's
            # baseMVA (100 for CATS.m; r/x/b are already pu and untouched)
            "rate_a": float(fb[i, _RATE_A]) / final_case.base_mva,
        }
        if not np.isnan(fids[i]):                   # regular line
            fid = int(round(fids[i]))
            props = feats[fid]["properties"]
            if props["FID"] != fids[i]:
                raise WeldError(f"feature {fid}: stored FID {props['FID']} "
                                f"!= branch FID {fids[i]} (layer/case skew)")
            if fid not in extra.index:
                raise WeldError(f"FID {fid} missing from the tl-extra CSV "
                                "(Julia indexed .Material[1] — BoundsError)")
            props["CATS_ID"] = branchid
            if props.get("Length_Fee") == " ":
                props["Length_Fee"] = "0.0"
            props["Length_km"] = float(props["Length_Fee"]) * gis.feet_to_km
            props["Structure_Material"] = extra.at[fid, "Material"]
            props["Structure_Type"] = extra.at[fid, "Structure"]
            props.update(electrical)
            for field in gis.outdated_fields:
                props.pop(field, None)
        else:                                       # appended transformer
            lat1 = _cell_value(br_cells, "Lat1", i)
            lat2 = _cell_value(br_cells, "Lat2", i)
            lon1 = _cell_value(br_cells, "Lon1", i)
            lon2 = _cell_value(br_cells, "Lon2", i)
            feats.append({
                "id": len(feats),                   # DV-07f deterministic
                "type": "Feature",
                "geometry": {"type": "LineString",  # DV-07b corrected
                             "coordinates": [[lon1, lat1], [lon2, lat2]]},
                "properties": {
                    "Length_km": 0.0, "Comments": "",
                    "Structure_Material": "", "Structure_Type": "",
                    "Lon1": lon1, "Lon2": lon2, "Lat1": lat1, "Lat2": lat2,
                    "CATS_ID": branchid, "Circuit": "", "Type": "",
                    **electrical,
                },
            })
    return tl_layer


def _reformat_buses(bus_layer: dict, case: MatpowerCase,
                    cfg: WeldConfig) -> dict:
    """The bus loop [reformat_GIS.jl:120-148] in ascending bus id."""
    feats = bus_layer["features"]
    cells = case.cells["bus_data"]
    for i in range(len(case.bus)):
        # PowerModels keys the BUS dict by bus_i (non-contiguous original
        # numbers in a largest-component export) — branches/gens are keyed
        # by row index instead.
        busid = str(int(case.bus[i, 0]))
        node_type = str(_cell_value(cells, "Type", i))
        if node_type == "AddedNode":
            fid = int(round(float(_cell_value(cells, "FID", i))))
            new_props = {"CATS_ID": busid, "Type": "AddedNode",
                         "kV": "", "Comments": ""}
            if cfg.export.gis.preserve_added_node_props:   # DV-07d opt-in
                feats[fid]["properties"].update(new_props)
            else:                                          # legacy: replace
                feats[fid]["properties"] = new_props
        elif node_type == "Substation":
            feats.append({
                "id": len(feats),                          # DV-07f
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [_cell_value(cells, "Lon", i),
                                             _cell_value(cells, "Lat", i)]},
                "properties": {"CATS_ID": busid, "Type": "Substation",
                               "kV": "", "Comments": ""},
            })
        else:
            raise WeldError(                # legacy: println["WARNING"] bug
                f"bus {busid}: Type {node_type!r} is neither AddedNode nor "
                "Substation")
    return bus_layer


def _gen_features(case: MatpowerCase) -> dict:
    """The generator FeatureCollection [reformat_GIS.jl:151-165]:
    properties = the full PowerModels gen dict (valid at baseMVA == 1 —
    asserted by the caller); feature id = 1-based genid (the source's
    unresolved 'SHOULD THIS BE id-1???' kept as-is)."""
    cells = case.cells.get("gen_data")
    if cells is not None:
        # Julia types cell columns as a whole: a MIXED column (strings +
        # numbers, e.g. GenID '1P' vs the import gens' bare 0) coerces its
        # numeric entries to strings ('0'); all-numeric columns stay numeric.
        cells = cells.copy()
        for col in cells.columns:
            is_str = cells[col].map(lambda v: isinstance(v, str))
            if is_str.any() and not is_str.all():
                cells[col] = cells[col].map(
                    lambda v: v if isinstance(v, str)
                    else (v if isinstance(v, float) and np.isnan(v)
                          else str(int(v)) if float(v).is_integer()
                          else str(v)))
    gencost = case.gencost
    features = []
    for i in range(len(case.gen)):
        row = case.gen[i]
        props = {
            "gen_bus": int(row[_GEN_BUS]),
            "pg": float(row[_PG]), "qg": float(row[_QG]),
            "qmax": float(row[_QMAX]), "qmin": float(row[_QMIN]),
            "vg": float(row[_VG]), "mbase": float(row[_MBASE]),
            "gen_status": int(row[_GEN_STATUS]),
            "pmax": float(row[_PMAX]), "pmin": float(row[_PMIN]),
            "index": i + 1,
            "source_id": ["gen", i + 1],
        }
        if gencost is not None:
            ncost = int(gencost[i, _NCOST])
            cost = [float(c) for c in gencost[i, _COST:_COST + ncost]]
            while cost and cost[0] == 0.0:      # PowerModels strips leading
                cost.pop(0)                     # zero coefficients (ncost
            props.update({                      # follows; all-zero → 0)
                "model": int(gencost[i, _MODEL]),
                "startup": float(gencost[i, _STARTUP]),
                "shutdown": float(gencost[i, _SHUTDOWN]),
                "ncost": len(cost),
                "cost": cost,
            })
        if cells is not None:
            for col in cells.columns:
                props[col] = _cell_value(cells, col, i)
        features.append({
            "id": i + 1,
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [props.get("Lon"), props.get("Lat")]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def _nan_to_null(obj):
    """Julia's JSON output writes missing/NaN properties as ``null``; the
    source layers are JSON (no NaN possible), so a global sweep only
    touches values this port introduced (kV/Structure_*/gen extras)."""
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def roundtrip_gis(case_path: str | Path, final_case_path: str | Path,
                  out_dir: str | Path, cfg: WeldConfig, *,
                  gis_layers: dict[str, Path] | None = None,
                  tl_extra: pd.DataFrame | None = None) -> dict[str, Path]:
    """reformat_GIS.jl equivalent → ``{'tl': …, 'bus': …, 'gen': …}`` paths.

    ``case_path``: the S2 build case (cal_largest_grid.m — carries the FID
    linkage); ``final_case_path``: the finished case (CATS.m lineage —
    carries final R/X/B/ratings). ``gis_layers``/``tl_extra`` default from
    ``cfg.paths`` (keys 'lines'/'added_nodes')."""
    p = cfg.paths
    if gis_layers is None:
        gis_layers = {"lines": p.resolve(p.gis_lines),
                      "added_nodes": p.resolve(p.gis_added_nodes)}
    if tl_extra is None:
        tl_extra = pd.read_csv(p.resolve(p.tl_extra), low_memory=False)

    case = read_matpower_case(case_path)
    final_case = read_matpower_case(final_case_path)
    if case.base_mva != 1.0:
        raise WeldError(
            f"baseMVA {case.base_mva} != 1: the PowerModels per-unit "
            "identity this port relies on does not hold (docstring)")
    tl_layer = json.loads(Path(gis_layers["lines"]).read_text(encoding="utf-8"))
    bus_layer = json.loads(
        Path(gis_layers["added_nodes"]).read_text(encoding="utf-8"))

    tl_out = _reformat_lines(tl_layer, case, final_case, tl_extra, cfg)
    bus_out = _reformat_buses(bus_layer, case, cfg)
    gen_out = _gen_features(case)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, payload in (("tl", tl_out), ("bus", bus_out), ("gen", gen_out)):
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_nan_to_null(payload), f, indent=1)
        paths[name] = path
    return paths
