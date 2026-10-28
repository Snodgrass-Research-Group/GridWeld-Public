"""Node extraction, dedup, and bus numbering — backend-independent.

Origin: ``Export_Node_Data_To_CSV`` in
Feeder_Cleanser_and_CSV_File_Generator.py and its hardened Method1.py
version (algorithm doc 01). The exact-float node equality of the older
exporter is superseded by Method1's 6-decimal ``rounded_coord(lon, lat)``
(PD-7 — exact-float silently splits coincident buses; verified against
Method1.py:320-321: key = (round(lon, 6), round(lat, 6))).

Pure numpy/pandas: reused by the arcpy backend (P2d) and the open backend
(P8).

Units: WGS84 degrees; 6 decimals ≈ 0.11 m.
Phase: P2d.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dedup_nodes(start: np.ndarray, end: np.ndarray,
                decimals: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Round (lon, lat) to ``decimals``, assign 1-based bus numbers in
    first-seen order (start point of segment 1, end point of segment 1,
    start of segment 2, ... — the legacy per-row lookup order).

    Parameters
    ----------
    start, end:
        (n_lines, 2) arrays of segment endpoint (lon, lat), degrees.
    decimals:
        Dedup key precision (``gis.node_round_decimals``).

    Returns
    -------
    nodes:
        DataFrame(Longitude, Latitude, Index) — Index 1-based (legacy CSV
        contract; 0-based internal ids appear at S2 ingest). Coordinates
        are the ROUNDED key values (Method1 writes the key, not the raw).
    edges:
        DataFrame(Node_From, Node_To) aligned with the input segment order.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    key_of: dict[tuple[float, float], int] = {}
    lon_out: list[float] = []
    lat_out: list[float] = []
    node_from = np.empty(len(start), dtype=np.int64)
    node_to = np.empty(len(end), dtype=np.int64)

    def _index(lon: float, lat: float) -> int:
        key = (round(lon, decimals), round(lat, decimals))
        idx = key_of.get(key)
        if idx is None:
            idx = len(key_of) + 1
            key_of[key] = idx
            lon_out.append(key[0])
            lat_out.append(key[1])
        return idx

    for i in range(len(start)):
        node_from[i] = _index(start[i, 0], start[i, 1])
        node_to[i] = _index(end[i, 0], end[i, 1])

    nodes = pd.DataFrame({"Longitude": lon_out, "Latitude": lat_out,
                          "Index": np.arange(1, len(lon_out) + 1,
                                             dtype=np.int64)})
    edges = pd.DataFrame({"Node_From": node_from, "Node_To": node_to})
    return nodes, edges
