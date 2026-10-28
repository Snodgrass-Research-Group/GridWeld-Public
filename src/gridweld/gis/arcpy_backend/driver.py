"""Parallel batch driver for the arcpy backend.

Origin: ``run_parallel_feeder_cleaning`` / ``process_one_feeder`` in
Method1.py (algorithm doc 01). ProcessPoolExecutor per item — arcpy is
process-safe, not thread-safe; each worker owns its own workspace GDB
(``safe_del``-style idempotent re-runs). Failures are collected as
``(item, "FAILED: …")`` and never abort the batch; output must be
byte-identical to a serial run (per-item outputs independent — doc 01
validation).

Phase: P2d (runs under ArcGIS Pro propy).
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np

from ...config import WeldConfig
from ..interface import CleanParams

log = logging.getLogger(__name__)


def _params_from_cfg(cfg: WeldConfig, network: str) -> CleanParams:
    knobs = getattr(cfg.gis.clean, network)
    return CleanParams(
        feature_to_line_tol_m=knobs.feature_to_line_tol_m,
        generalize_tol_m=knobs.generalize_tol_m,
        extend_line_m=knobs.extend_line_m,
        snap_edge_m=knobs.snap_edge_m,
        attribute_map_tol_m=knobs.attribute_map_tol_m)


def process_one(item: int, source_gdb: str, out_dir: str,
                cfg: WeldConfig, network: str = "distribution") -> str:
    """extract → clean_with_sweep → export node/branch CSVs for one item.

    Own workspace GDB per worker; the DV4 quarantine hazard (double-move
    clobbering Nodes.csv) cannot recur — files are written once at their
    final paths, never moved.
    """
    import arcpy

    from ..nodes import dedup_nodes
    from .clean import clean_with_sweep, extract_endpoints

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    gdb = out_dir_p / f"work_{item}.gdb"
    if arcpy.Exists(str(gdb)):
        arcpy.management.Delete(str(gdb))
    arcpy.management.CreateFileGDB(str(out_dir_p), gdb.name)
    ws = str(gdb)

    fc = f"{ws}/feeder_{item}"
    arcpy.conversion.ExportFeatures(f"{source_gdb}/FeederDetail", fc,
                                    f"OBJECTID = {item}")
    sweep = cfg.gis.island_sweep
    cleaned, used, n_islands = clean_with_sweep(
        fc, f"{ws}/clean_{item}", _params_from_cfg(cfg, network), ws,
        extend_step_m=sweep.extend_step_m, snap_step_m=sweep.snap_step_m,
        extend_max_m=sweep.extend_max_m)

    start, end = extract_endpoints(cleaned, ws)
    nodes, edges = dedup_nodes(np.asarray(start), np.asarray(end),
                               cfg.gis.node_round_decimals)
    nodes.to_csv(out_dir_p / f"Feeder_{item}_Nodes.csv", index=False)
    edges.to_csv(out_dir_p / f"Feeder_{item}_From_To_Nodes.csv", index=False)
    return (f"ok: {len(nodes)} nodes, {len(edges)} edges, "
            f"{n_islands} island(s), extend={used.extend_line_m}")


def run_batch(items: Sequence[int], source_gdb: str, out_dir: str,
              cfg: WeldConfig, *, network: str = "distribution",
              max_workers: int | None = None) -> list[tuple[int, str]]:
    """ProcessPoolExecutor over :func:`process_one`.
    ``cfg.gis.batch.skip_items`` are skipped up front (known-bad feeders);
    failures collected, not raised."""
    max_workers = max_workers or cfg.gis.batch.max_workers
    todo = [i for i in items if i not in set(cfg.gis.batch.skip_items)]
    results: list[tuple[int, str]] = [
        (i, "SKIPPED (known-bad)") for i in items if i not in todo]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, i, source_gdb, out_dir, cfg,
                               network): i for i in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                results.append((item, fut.result()))
            except Exception:
                results.append((item, f"FAILED: {traceback.format_exc(limit=3)}"))
                log.warning("item %d failed", item)
    return sorted(results)
