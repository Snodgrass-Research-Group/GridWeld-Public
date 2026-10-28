"""Island (connected-component) detection and repair — backend-independent.

Origin: ``Island_Counter*`` / ``Island_Destroyer.py`` (nearest-neighbor
stitching) in the EV repo (algorithm doc 01). Fixes ported per doc 01 /
DV-P2d-1: iterative DFS instead of recursive (PD-6 — RecursionError on
long feeders, the legacy "weird feeders"), named lat/lon columns (PD-5 —
the legacy fed the haversine swapped coordinates from the 0/1-column
CSVs), all-pairs island comparison (PD-8 — legacy Nearest_Multiple_
Neighbors only compared islands 1 vs 2 each round), single earth radius
(PD-9 — legacy mixed 6373.0 km and 6371000 m).

Transmission policy (doc 01 §7): stitching is capped by ``max_stitch_m`` —
a transmission island usually means a MISSING LINE, not a geometry slip;
refuse to fabricate long phantom edges. Every stitched edge is logged.

Units: metres for distances (``gis.earth_radius_m``); WGS84 degrees for
coordinates; node ids 0-based here (the 1-based CSV contract is the
caller's, matching the legacy adjacency build which subtracted 1).

Phase: P2d.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def haversine_m(lat1, lon1, lat2, lon2,
                radius_m: float = 6_371_000.0) -> np.ndarray | float:
    """Great-circle distance in metres (degrees in). ONE radius for every
    caller (PD-9)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return radius_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def connected_components(n_nodes: int,
                         edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components via ITERATIVE DFS (PD-6). 0-based node ids;
    components in discovery order, nodes in visit order (the legacy
    Graph_2 contract)."""
    if n_nodes == 0:
        raise ValueError("connected_components: empty node set")
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for f, t in edges:
        adj[int(f)].append(int(t))
        adj[int(t)].append(int(f))
    seen = np.zeros(n_nodes, dtype=bool)
    out: list[list[int]] = []
    for root in range(n_nodes):
        if seen[root]:
            continue
        comp: list[int] = []
        stack = [root]
        seen[root] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        out.append(comp)
    return out


def count_islands(n_nodes: int, edges: Iterable[tuple[int, int]]) -> int:
    """Component count (legacy Island_Counter contract)."""
    return len(connected_components(n_nodes, edges))


def _component_labels(n_nodes: int, edges: pd.DataFrame) -> np.ndarray:
    labels = np.empty(n_nodes, dtype=np.int64)
    pairs = zip(edges["Node_From"].to_numpy(np.int64) - 1,
                edges["Node_To"].to_numpy(np.int64) - 1)
    for ci, comp in enumerate(connected_components(n_nodes, pairs)):
        labels[comp] = ci
    return labels


def nearest_cross_island_pair(nodes: pd.DataFrame, comp_of: np.ndarray
                              ) -> tuple[int, int, float]:
    """Minimum-haversine node pair spanning two different components — ALL
    component pairs considered (PD-8). ``nodes`` needs Longitude/Latitude/
    Index (named columns — PD-5). Returns (Index_a, Index_b, metres)."""
    lat = nodes["Latitude"].to_numpy(float)
    lon = nodes["Longitude"].to_numpy(float)
    ids = nodes["Index"].to_numpy(np.int64)
    best = (-1, -1, np.inf)
    for i in range(len(nodes)):
        other = comp_of != comp_of[i]
        if not other.any():
            continue
        d = haversine_m(lat[i], lon[i], lat[other], lon[other])
        j_rel = int(np.argmin(d))
        j = np.flatnonzero(other)[j_rel]
        if d[j_rel] < best[2] and i < j:
            best = (int(ids[i]), int(ids[j]), float(d[j_rel]))
    return best


def stitch_islands(nodes: pd.DataFrame, edges: pd.DataFrame, *, kv: str,
                   max_stitch_m: float | None = None) -> pd.DataFrame:
    """Append zero-LoadHosting edges until one component remains.

    Returns the edge frame with appended rows (Node_From, Node_To, kV,
    LoadHosting=0). Raises when a needed connection exceeds
    ``max_stitch_m`` (transmission policy) — S2 takes the largest component
    anyway, but stitching decisions must be logged, not silent."""
    out = edges.copy()
    n = len(nodes)
    while True:
        labels = _component_labels(n, out)
        if labels.max() == 0:
            return out
        a, b, dist = nearest_cross_island_pair(nodes, labels)
        if max_stitch_m is not None and dist > max_stitch_m:
            raise ValueError(
                f"island gap {dist:.0f} m exceeds max_stitch_m "
                f"({max_stitch_m:.0f} m) between nodes {a} and {b} — "
                "likely a missing line, not a geometry slip (doc 01 §7)")
        log.info("stitching islands: node %d ↔ node %d (%.1f m)", a, b, dist)
        row = {"Node_From": a, "Node_To": b, "kV": kv, "LoadHosting": 0}
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
