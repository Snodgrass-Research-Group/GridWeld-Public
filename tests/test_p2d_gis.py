"""P2d validation: the portable GIS machinery (node dedup, island
detection/stitching) + arcpy-backend import safety.

The arcpy geoprocessing wrappers themselves need ArcGIS Pro (propy) and
real CATS inputs — that runtime comparison against legacy/Data/GIS/ layers
is not covered here; here we validate everything that runs
without a license, on synthetic cases and the julia fixture graph.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT
from gridweld.gis.islands import (connected_components, count_islands,
                                  haversine_m, nearest_cross_island_pair,
                                  stitch_islands)
from gridweld.gis.nodes import dedup_nodes

FIX = REPO_ROOT / "validation" / "fixtures" / "julia"
needs_julia = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                 reason="julia fixtures not extracted")


# ---------------------------------------------------------------------------
# nodes.dedup_nodes (PD-7: 6-decimal key, first-seen numbering)
# ---------------------------------------------------------------------------

def test_dedup_nodes_first_seen_order():
    start = np.array([[-120.0, 37.0], [-120.5, 37.5]])
    end = np.array([[-120.5, 37.5], [-121.0, 38.0]])
    nodes, edges = dedup_nodes(start, end)
    assert nodes["Index"].tolist() == [1, 2, 3]
    # first-seen: start1 → 1, end1 → 2, start2 == end1 → 2, end2 → 3
    assert edges["Node_From"].tolist() == [1, 2]
    assert edges["Node_To"].tolist() == [2, 3]


def test_dedup_nodes_rounding_merges():
    a = [-120.1234561, 37.0]
    b = [-120.1234564, 37.0]        # same at 6 decimals (≈0.03 m apart)
    c = [-120.1234570, 37.0]        # different 6th decimal
    nodes, edges = dedup_nodes(np.array([a, c]), np.array([b, a]))
    assert len(nodes) == 2          # a≡b merge; c distinct
    assert edges["Node_From"].tolist() == [1, 2]
    assert edges["Node_To"].tolist() == [1, 1]
    # node coordinates are the ROUNDED keys (Method1 writes the key)
    assert nodes["Longitude"].iloc[0] == round(a[0], 6)


# ---------------------------------------------------------------------------
# islands: iterative DFS (PD-6), all-pairs stitching (PD-8), cap policy
# ---------------------------------------------------------------------------

def test_connected_components_iterative_on_long_path():
    n = 20_000                       # a recursive DFS would blow the stack
    edges = [(i, i + 1) for i in range(n - 1)]
    comps = connected_components(n, edges)
    assert len(comps) == 1 and len(comps[0]) == n


def test_count_islands_multi():
    assert count_islands(6, [(0, 1), (1, 2), (3, 4)]) == 3   # {0,1,2},{3,4},{5}


@needs_julia
def test_final_cats_graph_is_one_island():
    """doc 01 validation: the final largest-grid from-to table must be a
    single component."""
    br = pd.read_csv(FIX / "p5_final_largest_branch.csv")
    bus = pd.read_csv(FIX / "p5_final_largest_bus.csv")
    ids = {int(b): i for i, b in enumerate(bus["index"])}
    edges = [(ids[int(f)], ids[int(t)])
             for f, t in zip(br["f_bus"], br["t_bus"])]
    assert count_islands(len(bus), edges) == 1


def test_haversine_m_matches_snapping_km():
    from gridweld.build.snapping import haversine_km
    d_m = haversine_m(37.0, -120.0, 38.0, -121.0)
    d_km = haversine_km(37.0, -120.0, 38.0, -121.0)
    assert d_m == pytest.approx(d_km * 1000.0, rel=1e-12)


def _three_island_nodes():
    """Islands A{1,2} B{3,4} C{5}; the smallest cross-island gap is B–C —
    the legacy 1-vs-2-only comparison (PD-8) would stitch A–B first."""
    return pd.DataFrame({
        "Index": [1, 2, 3, 4, 5],
        "Longitude": [-120.0, -120.001, -119.5, -119.499, -119.4985],
        "Latitude": [37.0, 37.0, 37.0, 37.0, 37.0],
    })


def test_nearest_cross_island_pair_all_pairs():
    nodes = _three_island_nodes()
    edges = pd.DataFrame({"Node_From": [1, 3], "Node_To": [2, 4]})
    from gridweld.gis.islands import _component_labels
    labels = _component_labels(5, edges)
    a, b, dist = nearest_cross_island_pair(nodes, labels)
    assert {a, b} == {4, 5}          # B–C, not the legacy A–B
    assert dist == pytest.approx(haversine_m(37.0, -119.499, 37.0, -119.4985),
                                 rel=1e-9)


def test_stitch_islands_until_connected():
    nodes = _three_island_nodes()
    edges = pd.DataFrame({"Node_From": [1, 3], "Node_To": [2, 4],
                          "kV": ["115", "115"], "LoadHosting": [1, 1]})
    out = stitch_islands(nodes, edges, kv="115")
    assert len(out) == 4             # two stitches added
    added = out.iloc[2:]
    assert (added["LoadHosting"] == 0).all()
    assert (added["kV"] == "115").all()
    pairs = list(zip(out["Node_From"] - 1, out["Node_To"] - 1))
    assert count_islands(5, pairs) == 1


def test_stitch_islands_respects_cap():
    nodes = _three_island_nodes()
    edges = pd.DataFrame({"Node_From": [1, 3], "Node_To": [2, 4],
                          "kV": ["115"] * 2, "LoadHosting": [1, 1]})
    with pytest.raises(ValueError, match="max_stitch_m"):
        stitch_islands(nodes, edges, kv="115", max_stitch_m=10.0)


# ---------------------------------------------------------------------------
# arcpy backend: importable without ArcGIS (lazy arcpy imports, D4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", [
    "gridweld.gis.arcpy_backend.clean",
    "gridweld.gis.arcpy_backend.transmission",
    "gridweld.gis.arcpy_backend.spatial_join",
    "gridweld.gis.arcpy_backend.islands_geom",
    "gridweld.gis.arcpy_backend.driver",
    "gridweld.gis.interface",
])
def test_arcpy_backend_imports_without_arcgis(mod):
    importlib.import_module(mod)


def test_clean_params_frozen():
    from gridweld.gis.interface import CleanParams
    p = CleanParams(feature_to_line_tol_m=1.0, generalize_tol_m=0.0,
                    extend_line_m=10.0, snap_edge_m=5.0,
                    attribute_map_tol_m=0.0)
    with pytest.raises(Exception):
        p.extend_line_m = 99.0       # type: ignore[misc]
