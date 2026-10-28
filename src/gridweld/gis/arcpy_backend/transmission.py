"""Sofia's transmission-specific cleaning steps (arcpy).

Origin: EV-Research-Simulations ``GIS-Code/Sofia_GIS_data_processing.ipynb``
(algorithm doc 01 §Sofia) — the only surviving trace of the uncommitted
ArcPy work behind ``legacy/Data/GIS/``. Ports fix the committed notebook's
defects (DV-P2d-1): true first/last vertices instead of the centroid
CalculateFields expressions (PD-1 — the committed cell cannot have produced
the shipped data), FindIdentical on the computed endpoint fields (PD-2),
FID-to-FID node matching (PD-3 — the legacy compared an IN_FID to a
geometry object, dead code), explicit DeleteIdentical arguments (PD-4).

Units: tolerances metres; lengths US survey miles (``gis.length_units`` —
CalculateGeometryAttributes "Miles (United States)"); coordinates EPSG:4326.

Phase: P2d (runs under ArcGIS Pro propy).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def dedup_added_nodes(nodes_fc: str, tol_m: float = 2.0) -> None:
    """DeleteIdentical on Shape within ``tol_m`` (legacy "2 Meter"), keep
    one node per coincident cluster [notebook cell 12]."""
    import arcpy
    arcpy.management.DeleteIdentical(nodes_fc, ["Shape"], f"{tol_m} Meters")


def drop_nodes_near_substations(nodes_fc: str, subs_fc: str,
                                radius_m: float = 12.0) -> int:
    """Near_analysis vs substations; delete added nodes with 0 < d ≤ radius
    (the substation IS the bus there). Legacy polarity kept: d == 0 and
    d == −1 (nothing within radius) survive [notebook cell 13, commented
    variant — part of the intended flow per doc 01]."""
    import arcpy
    arcpy.analysis.Near(nodes_fc, subs_fc, search_radius=f"{radius_m} Meters")
    deleted = 0
    with arcpy.da.UpdateCursor(nodes_fc, ["NEAR_DIST"]) as cur:
        for (dist,) in cur:
            if dist is not None and dist > 0:
                cur.deleteRow()
                deleted += 1
    log.info("dropped %d added nodes within %.0f m of a substation",
             deleted, radius_m)
    return deleted


def split_tapped_lines(lines_fc: str, nodes_fc: str, out_fc: str,
                       search_radius_m: float = 1.0) -> None:
    """SplitLineAtPoint at (substations ∪ added nodes), 1 m point search
    [notebook cell 14]."""
    import arcpy
    arcpy.management.SplitLineAtPoint(lines_fc, nodes_fc, out_fc,
                                      f"{search_radius_m} Meters")


def recompute_line_attributes(lines_fc: str,
                              length_units: str = "MILES_US") -> None:
    """AddField Length_mi/Lon1/Lat1/Lon2/Lat2 (DOUBLE);
    CalculateGeometryAttributes LENGTH_GEODESIC in US survey miles; endpoint
    columns from the TRUE first/last vertices reprojected to EPSG:4326
    (PD-1 — no centroid expressions)."""
    import arcpy
    for field in ("Length_mi", "Lon1", "Lat1", "Lon2", "Lat2"):
        if field not in [f.name for f in arcpy.ListFields(lines_fc)]:
            arcpy.management.AddField(lines_fc, field, "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        lines_fc, [["Length_mi", "LENGTH_GEODESIC"]],
        length_unit=length_units)
    wgs84 = arcpy.SpatialReference(4326)
    with arcpy.da.UpdateCursor(
            lines_fc, ["SHAPE@", "Lon1", "Lat1", "Lon2", "Lat2"]) as cur:
        for row in cur:
            geo = row[0].projectAs(wgs84)
            first, last = geo.firstPoint, geo.lastPoint
            cur.updateRow([row[0], first.X, first.Y, last.X, last.Y])


def collapse_parallel_segments(lines_fc: str, nodes_fc: str
                               ) -> list[int]:
    """Collapse split-and-rejoin single-pole pairs: FindIdentical on
    (Lon1, Lat1, Lon2, Lat2), ONLY_DUPLICATES, 0 m (PD-2 — the computed
    endpoint fields, not the never-created Start_X set); delete added
    nodes whose XY equals a duplicate's endpoint by FID match (PD-3);
    DeleteIdentical with EXPLICIT endpoint fields keeps one line per pair
    (PD-4). Returns the dropped line OIDs — recorded as parallel-circuit
    candidates for S3, never silently discarded (doc 01 §4)."""
    import arcpy
    fields = ["Lon1", "Lat1", "Lon2", "Lat2"]
    dup_table = f"{lines_fc}_duplicates"
    arcpy.management.FindIdentical(lines_fc, dup_table, fields, "0 Meters",
                                   output_record_option="ONLY_DUPLICATES")
    dup_fids = [row[0] for row in arcpy.da.SearchCursor(dup_table, ["IN_FID"])]
    dup_set = set(dup_fids)
    oid_field = arcpy.Describe(lines_fc).OIDFieldName
    endpoints: set[tuple[float, float]] = set()
    with arcpy.da.SearchCursor(lines_fc, [oid_field, *fields]) as cur:
        for oid, lon1, lat1, lon2, lat2 in cur:
            if oid in dup_set:
                endpoints.add((round(lon1, 6), round(lat1, 6)))
                endpoints.add((round(lon2, 6), round(lat2, 6)))
    with arcpy.da.UpdateCursor(nodes_fc, ["SHAPE@XY"]) as cur:
        for ((x, y),) in cur:
            if (round(x, 6), round(y, 6)) in endpoints:
                cur.deleteRow()
    arcpy.management.DeleteIdentical(lines_fc, fields, "0 Meters")
    log.info("collapsed %d duplicate parallel segments (kept one each)",
             len(dup_fids))
    return dup_fids
