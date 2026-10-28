"""QA reports the Julia build script emitted to Plots/ during S2.

Origin: ``generate_grid_script.jl`` side outputs (algorithm doc 07 §QA):
self-loop list, zero-length lines (suspected in-substation transformers),
transformer inventory, per-bus load tables, >15 km snap distances
(generators and transmission endpoints), duplicate-endpoint lines, and the
per-bus loads HTML table. One DataFrame per diagnostic over the
internal-schema S2 WeldModel (``S2Grid.to_model`` columns).

The legacy misspelling ``long_distance_tranmission.csv`` is NORMALIZED to
``long_distance_transmission.csv`` (no byte-level fixture comparison is
planned — noted per doc 07).

Units: km (haversine, shared S2/S3 implementation); MW/MVar.
Phase: P5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import WeldModel


def qa_frames(model: WeldModel, cfg: WeldConfig) -> dict[str, pd.DataFrame]:
    """The diagnostics as DataFrames (pure; write_qa_reports persists)."""
    thr = cfg.export.qa.long_distance_km
    br = model.branches
    out: dict[str, pd.DataFrame] = {}

    out["self_loops"] = br.loc[
        br["from_bus"].to_numpy() == br["to_bus"].to_numpy()].copy()

    same_pt = ((br["lat1"].to_numpy(float) == br["lat2"].to_numpy(float))
               & (br["lon1"].to_numpy(float) == br["lon2"].to_numpy(float)))
    out["tiny_lines"] = br.loc[same_pt & br["lat1"].notna()].copy()

    is_xf = (br["is_transformer"].to_numpy(bool)
             if "is_transformer" in br.columns
             else br["tap"].to_numpy(float) != 0)
    out["transformers"] = br.loc[is_xf].copy()

    loads = model.loads
    per_bus = (loads.groupby("bus")
               .agg(pd_mw=("pd_mw", "sum"), qd_mvar=("qd_mvar", "sum"),
                    n_loads=("load_id", "count"))
               .reset_index())
    out["grid_loads"] = per_bus
    out["largest_grid_loads"] = per_bus     # post-reduce model IS the largest

    gens = model.generators
    gd = gens["gen_bus_distance_km"].to_numpy(float)
    out["long_distance_generators"] = gens.loc[gd > thr].copy()

    fd = br["f_bus_distance_km"].to_numpy(float)
    td = br["t_bus_distance_km"].to_numpy(float)
    far = np.nan_to_num(fd, nan=0.0) > thr
    far |= np.nan_to_num(td, nan=0.0) > thr
    out["long_distance_transmission"] = br.loc[far].copy()

    lo = np.minimum(br["from_bus"].to_numpy(), br["to_bus"].to_numpy())
    hi = np.maximum(br["from_bus"].to_numpy(), br["to_bus"].to_numpy())
    key = pd.DataFrame({"lo": lo, "hi": hi})
    dup = key.duplicated(keep=False) & ~is_xf
    out["transmission_same_endpoints"] = (
        br.loc[dup].assign(lo=lo[dup], hi=hi[dup])
        .sort_values(["lo", "hi"]))
    return out


def write_qa_reports(model: WeldModel, out_dir: Path,
                     cfg: WeldConfig) -> dict[str, Path]:
    """One CSV per diagnostic + ``bus_loads.html`` (the per-bus load table);
    returns name → path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = qa_frames(model, cfg)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    html = out_dir / "bus_loads.html"
    frames["grid_loads"].to_html(html, index=False)
    paths["bus_loads_html"] = html
    return paths
