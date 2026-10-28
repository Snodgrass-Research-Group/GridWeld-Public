"""Batched line-layer cleaning (arcpy): the core clean sequence + sweep.

Origin: ``Feeder_Cleanser_and_CSV_File_Generator.py`` step (a)-(f) and
``Island_Destroyer.py`` ``Island_Destroyer_Parameter_Aproach`` (parameter
sweep), EV repo — algorithm doc 01. DV4 (the double-move that clobbered the
Nodes CSV in Automated_Feeder_Cleaner.py) does not recur: exports never
move files, they write final paths directly.

Transmission runs use ``gis.clean.transmission`` knobs (1 m FeatureToLine,
Generalize skipped, snap target = substation VERTEX); distribution uses
``gis.clean.distribution`` (15/30/10/5 m) — doc 01 knob table.

arcpy imports are function-local: the rest of gridweld never needs ArcGIS
(D4); this module runs under ArcGIS Pro's propy.

Units: metres. Phase: P2d.
"""

from __future__ import annotations

import logging

from ..interface import CleanParams, LayerRef

log = logging.getLogger(__name__)


def clean_lines(layer: LayerRef, out_fc: str, p: CleanParams, *,
                snap_targets: LayerRef | None = None) -> LayerRef:
    """FeatureToLine → (Generalize if tol > 0) → ExtendLine → Snap →
    SplitLine → FeatureToLine, tolerances from ``p`` (metres).

    ``snap_targets``: feature class whose VERTEX geometry line ends snap to
    (transmission: the substations layer — doc 01 §2); None = self-EDGE
    snap (feeders).
    """
    import arcpy
    tmp = f"{out_fc}_f2l"
    arcpy.management.FeatureToLine(layer, tmp,
                                   f"{p.feature_to_line_tol_m} Meters",
                                   "ATTRIBUTES")
    if p.generalize_tol_m > 0:                       # transmission skips (0.0)
        arcpy.edit.Generalize(tmp, f"{p.generalize_tol_m} Meters")
    arcpy.edit.ExtendLine(tmp, f"{p.extend_line_m} Meters", "EXTENSION")
    if snap_targets is not None:
        env = [[snap_targets, "VERTEX", f"{p.snap_edge_m} Meters"]]
    else:
        env = [[tmp, "EDGE", f"{p.snap_edge_m} Meters"]]
    arcpy.edit.Snap(tmp, env)
    split = f"{out_fc}_split"
    arcpy.management.SplitLine(tmp, split)
    arcpy.management.FeatureToLine(split, out_fc,
                                   f"{p.feature_to_line_tol_m} Meters",
                                   "ATTRIBUTES")
    for scratch in (tmp, split):
        arcpy.management.Delete(scratch)
    return out_fc


def extract_endpoints(layer: LayerRef, workspace: str
                      ) -> tuple[list[tuple[float, float]],
                                 list[tuple[float, float]]]:
    """START and END vertex (lon, lat) lists in WGS84 —
    FeatureVerticesToPoints + AddXY with outputCoordinateSystem forced to
    GCS_WGS_1984 (the modern export path; the old UTM conversion is the
    PD-13 hazard, not ported)."""
    import arcpy
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("GCS_WGS_1984")
    out: list[list[tuple[float, float]]] = []
    for which in ("START", "END"):
        fc = f"{workspace}/nodes_{which.lower()}"
        arcpy.management.FeatureVerticesToPoints(layer, fc, which)
        arcpy.management.AddXY(fc)
        pts = [(row[0], row[1]) for row in
               arcpy.da.SearchCursor(fc, ["POINT_X", "POINT_Y"])]
        out.append(pts)
        arcpy.management.Delete(fc)
    return out[0], out[1]


def count_layer_islands(layer: LayerRef, workspace: str,
                        decimals: int = 6) -> int:
    """Island count of a cleaned layer: endpoint extraction → dedup_nodes →
    iterative-DFS components (PD-6/PD-7 fixed portable machinery)."""
    from ..islands import count_islands
    from ..nodes import dedup_nodes
    import numpy as np
    start, end = extract_endpoints(layer, workspace)
    nodes, edges = dedup_nodes(np.asarray(start), np.asarray(end), decimals)
    pairs = zip(edges["Node_From"].to_numpy() - 1,
                edges["Node_To"].to_numpy() - 1)
    return count_islands(len(nodes), pairs)


def clean_with_sweep(layer: LayerRef, out_fc: str, p: CleanParams,
                     workspace: str, *, extend_step_m: float = 3.0,
                     snap_step_m: float = 1.5, extend_max_m: float = 50.0
                     ) -> tuple[LayerRef, CleanParams, int]:
    """Island_Destroyer parameter sweep: re-run :func:`clean_lines` with
    extend += 3 m and snap += 1.5 m per retry until one component or
    extend > cap (first retry runs at 13/6.5 m — doc 01). Remaining islands
    are the caller's problem (``gridweld.gis.islands.stitch_islands`` on the
    exported CSVs, subject to ``max_stitch_m`` — doc 01 §7).

    Returns (cleaned layer, the params that produced it, island count).
    """
    current = p
    while True:
        result = clean_lines(layer, out_fc, current)
        n = count_layer_islands(result, workspace)
        if n <= 1 or current.extend_line_m + extend_step_m > extend_max_m:
            if n > 1:
                log.warning("sweep exhausted at extend=%.1f m: %d islands "
                            "remain (stitch or fail per policy)",
                            current.extend_line_m, n)
            return result, current, n
        current = CleanParams(
            feature_to_line_tol_m=current.feature_to_line_tol_m,
            generalize_tol_m=current.generalize_tol_m,
            extend_line_m=current.extend_line_m + extend_step_m,
            snap_edge_m=current.snap_edge_m + snap_step_m,
            attribute_map_tol_m=current.attribute_map_tol_m)
        log.info("island sweep: retrying clean at extend=%.1f snap=%.1f",
                 current.extend_line_m, current.snap_edge_m)
