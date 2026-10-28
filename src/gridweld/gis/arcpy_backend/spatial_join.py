"""Midpoint nearest-neighbor attribute transfer (arcpy).

Origin: ``Ruiz_Spatial_Join_ver_2`` — the Method1.py version supersedes the
Feeder_Cleanser one (MID vertices + haversine R = 6371000 m; doc 01
§Method1). The tolerance is an EXPLICIT parameter — the legacy local
``Tol = 0`` silently shadowed the module constant ``Tol = 10000000000``
(PD-10); exact-coincidence matching vs generous-radius matching was an
accident of scoping, not a decision.

Phase: P2d (runs under ArcGIS Pro propy).
"""

from __future__ import annotations

import logging
from typing import Mapping

from ..interface import LayerRef

log = logging.getLogger(__name__)


def _midpoints(fc: str) -> list[tuple[float, float, list]]:
    """(lon, lat, attribute row) per feature MID vertex, WGS84."""
    import arcpy
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("GCS_WGS_1984")
    mid = f"{fc}_mid"
    arcpy.management.FeatureVerticesToPoints(fc, mid, "MID")
    arcpy.management.AddXY(mid)
    rows = [(r[0], r[1]) for r in arcpy.da.SearchCursor(mid, ["POINT_X",
                                                              "POINT_Y"])]
    arcpy.management.Delete(mid)
    return rows


def midpoint_attribute_transfer(target_fc: LayerRef, source_fc: LayerRef,
                                fields: Mapping[str, str], tol_m: float,
                                unmatched_value: int = -1) -> int:
    """Transfer ``fields`` (target_col ← source_col) from the nearest source
    midpoint within ``tol_m`` metres (strict nearest neighbor, haversine —
    ``gridweld.gis.islands.haversine_m``, PD-9's single radius); unmatched
    targets get ``unmatched_value`` (repaired later by the distribution-side
    graph fill passes — out of S1 scope, doc 01). Returns unmatched count.
    """
    import arcpy
    import numpy as np

    from ..islands import haversine_m

    src_pts = _midpoints(source_fc)
    tgt_pts = _midpoints(target_fc)
    src_lon = np.array([p[0] for p in src_pts])
    src_lat = np.array([p[1] for p in src_pts])
    src_rows = list(arcpy.da.SearchCursor(source_fc, list(fields.values())))

    assignments: list[tuple | None] = []
    unmatched = 0
    for lon, lat in tgt_pts:
        d = haversine_m(lat, lon, src_lat, src_lon)
        j = int(np.argmin(d))
        if d[j] <= tol_m:
            assignments.append(src_rows[j])
        else:
            assignments.append(None)
            unmatched += 1

    tgt_cols = list(fields.keys())
    existing = {f.name for f in arcpy.ListFields(target_fc)}
    for col in tgt_cols:
        if col not in existing:
            arcpy.management.AddField(target_fc, col, "DOUBLE")
    with arcpy.da.UpdateCursor(target_fc, tgt_cols) as cur:
        for i, _ in enumerate(cur):
            cur.updateRow(list(assignments[i]) if assignments[i] is not None
                          else [unmatched_value] * len(tgt_cols))
    if unmatched:
        log.info("midpoint transfer: %d/%d targets unmatched at tol %.1f m",
                 unmatched, len(tgt_pts), tol_m)
    return unmatched
