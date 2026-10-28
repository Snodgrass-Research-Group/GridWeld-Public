"""Geometric island reconnection (arcpy) — complements the CSV-level stitcher.

Origin: ``connect_islands_to_main`` in Method1.py (algorithm doc 01). The
port connects each island centroid to the TRUE nearest point on the
mainland via ``queryPointAndDistance`` — the legacy passed a
``measureOnLine`` DISTANCE into ``positionAlongLine(..,
use_percentage=True)``, landing connectors at the wrong spot (PD-11); its
``snap_tolerance`` parameter was unused (not ported).

Phase: P2d (runs under ArcGIS Pro propy).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def connect_islands_geometric(line_fc: str, *, crs_epsg: int = 4326) -> int:
    """Dissolve SINGLE_PART; largest part by length = mainland; draw a
    straight connector from each island centroid to its nearest mainland
    point (queryPointAndDistance — PD-11 fixed); merge connectors back.
    Returns the number of connectors added."""
    import arcpy
    sr = arcpy.SpatialReference(crs_epsg)
    dissolved = f"{line_fc}_dissolved"
    arcpy.management.Dissolve(line_fc, dissolved, multi_part="SINGLE_PART")

    parts = [row[0] for row in arcpy.da.SearchCursor(dissolved, ["SHAPE@"])]
    if len(parts) <= 1:
        arcpy.management.Delete(dissolved)
        return 0
    main = max(parts, key=lambda g: g.length)
    connectors = []
    for part in parts:
        if part is main:
            continue
        centroid = arcpy.PointGeometry(part.centroid, sr)
        near_pt, _dist, _along, _side = main.queryPointAndDistance(centroid)
        connectors.append(arcpy.Polyline(
            arcpy.Array([part.centroid, near_pt.firstPoint]), sr))
    if connectors:
        tmp = f"{line_fc}_connectors"
        arcpy.management.CopyFeatures(connectors, tmp)
        arcpy.management.Append(tmp, line_fc, "NO_TEST")
        arcpy.management.Delete(tmp)
    arcpy.management.Delete(dissolved)
    log.info("added %d geometric island connectors", len(connectors))
    return len(connectors)
