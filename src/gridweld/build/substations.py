"""Substation / added-node ingestion → bus table.

Origin: substation/added-node cleaning + ``add_substations!``/``add_nodes!``
in ``generate_grid_script.jl`` (lines 56-69) / ``helper_functions.jl``
(algorithm doc 02 §substations). The id rule couples the two files:
added-node ids are offset by the substation ROW COUNT — any substation CSV
edit renumbers every AddedNode. Rule kept; ranges asserted disjoint.

The legacy ``add_*`` functions copy rows verbatim into string-keyed dicts;
here the S2 working state is plain DataFrames with the same column names
(Lat/Lon/FID/Import/Type + 1-based ``index``), which the julia fixture CSVs
mirror directly.

Units: lat/lon WGS84 degrees.
Phase: P2c. Validated vs p1_substations_clean / p1_new_nodes_clean.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The Julia column subset (select! order) shared by both point layers.
POINT_COLUMNS = ["index", "Lat", "Lon", "FID", "Import", "Type"]


def load_substations(path: str | Path) -> pd.DataFrame:
    """Substation CSV → frame: ``index = FID + 1``, Type='Substation'.
    ``Import`` is 'IMPORT' or blank/space (kept verbatim — a missing value
    exports as ``' '``, doc 02)."""
    df = pd.read_csv(path, low_memory=False)
    df["index"] = df["FID"].astype("int64") + 1
    df["Type"] = "Substation"
    df["Import"] = df["Import"].fillna(" ")
    return df[POINT_COLUMNS].copy()


def load_added_nodes(path: str | Path, n_substations: int) -> pd.DataFrame:
    """Added-node CSV → frame: ``index = FID + 1 + n_substations`` (the
    coupling rule), Type='AddedNode'. The legacy commented-out Import
    blanking loop is inactive — missing stays missing → ``' '`` (doc 02)."""
    df = pd.read_csv(path, low_memory=False)
    df["index"] = df["FID"].astype("int64") + 1 + n_substations
    df["Type"] = "AddedNode"
    df["Import"] = df["Import"].fillna(" ")
    return df[POINT_COLUMNS].copy()


def make_bus_table(substations: pd.DataFrame,
                   added_nodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """add_substations! (+ add_nodes!): concatenated bus table, sorted by
    ``index`` for deterministic downstream iteration (DV-P2c-2). Asserts
    the id ranges are disjoint (doc 02 coupling hazard)."""
    frames = [substations] if added_nodes is None else [substations, added_nodes]
    bus = pd.concat(frames, ignore_index=True)
    if bus["index"].duplicated().any():
        dup = bus.loc[bus["index"].duplicated(), "index"].tolist()[:10]
        raise ValueError(f"substation/added-node id ranges collide: {dup}")
    return bus.sort_values("index", ignore_index=True)
