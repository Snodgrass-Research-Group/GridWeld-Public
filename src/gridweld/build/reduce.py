"""Transmission cleaning, finalizers, and largest-component reduction.

Origin (all algorithm doc 02): transmission tiny-line drop + kV remap
(``generate_grid_script.jl`` lines 212-232), ``add_branches!``, the
finalizers ``add_component_status!`` / ``add_branch_defaults!`` /
``add_source_id!``, ``remove_self_loops!`` / ``get_largest_grid``
(``helper_functions.jl``), and the ``comp_dict_to_dataframe`` audit
flattener.

Ported quirks: kV remap ties resolve to the FIRST (lowest) level (Julia
findmin ≡ np.argmin first occurrence); every bus gets bus_type = 1 (NO
slack at S2 — assigned downstream in S3); ``get_largest_grid`` keeps
branches via **OR** on endpoint membership (replicated, not "fixed" to
AND — harmless because both endpoints share a component);
``identify_blocks`` → scipy connected_components (the dump script's
union-find replacement). The audit flattener stringifies
into a COPY — the legacy in-place mutation before export (DV-P2c-3) is not
reproduced.

Units: kV levels from cfg.build.kv_levels.
Phase: P2c. Validated vs p1_transmission_clean / p1_tiny_lines /
p5_self_loops / p5_final_largest_* fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..config import WeldConfig


def clean_transmission(path: str | Path, cfg: WeldConfig
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transmission CSV → (lines, tiny_lines).

    ``index = FID + 1``; tiny lines = EXACT float equality of endpoints
    (suspected in-substation transformers); after the drop, ``index`` is
    renumbered 1..n (FID keeps provenance); kV remapped onto
    cfg.build.kv_levels by nearest, ties → first/lowest. The ACTIVE code
    uses column ``kV`` (``kV_Actual`` read is commented out); the dormant
    ``kV_Sort > 220`` filter is not ported.
    """
    df = pd.read_csv(path, low_memory=False)
    df["index"] = df["FID"].astype("int64") + 1
    df = df[["index", "Lat1", "Lon1", "Lat2", "Lon2", "kV", "FID"]].copy()

    tiny_mask = (df["Lat1"] == df["Lat2"]) & (df["Lon1"] == df["Lon2"])
    tiny = df.loc[tiny_mask].copy()
    if cfg.build.remove_tiny_lines:
        df = df.loc[~tiny_mask].reset_index(drop=True)
        df["index"] = np.arange(1, len(df) + 1, dtype=np.int64)

    kv_set = np.asarray(cfg.build.kv_levels, dtype=float)
    kv = df["kV"].to_numpy(dtype=float)
    df["kV"] = kv_set[np.abs(kv[:, None] - kv_set[None, :]).argmin(axis=1)]
    return df, tiny


def remove_self_loops(branch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """remove_self_loops!: drop branches with f_bus == t_bus; returns
    (branches, self_loop FID frame — the Plots/self_loops.csv audit)."""
    loop_mask = branch_df["f_bus"].to_numpy() == branch_df["t_bus"].to_numpy()
    loops = pd.DataFrame({"FID": branch_df.loc[loop_mask, "FID"].to_numpy()})
    return branch_df.loc[~loop_mask].reset_index(drop=True), loops


def _blocks(bus_ids: np.ndarray, f_bus: np.ndarray,
            t_bus: np.ndarray) -> np.ndarray:
    """Connected-component label per bus id (identify_blocks equivalent —
    the dump script replaced PowerModelsAnalytics with union-find; scipy
    connected_components is the same computation)."""
    pos = {int(b): i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)
    f = np.array([pos[int(b)] for b in f_bus], dtype=np.int64)
    t = np.array([pos[int(b)] for b in t_bus], dtype=np.int64)
    adj = sp.csr_matrix((np.ones(len(f)), (f, t)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def largest_component(bus: pd.DataFrame, branch: pd.DataFrame,
                      gen: pd.DataFrame, load: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame]:
    """get_largest_grid: keep the block with the most buses; drop gens/loads
    whose bus is outside, buses outside, and branches where NEITHER endpoint
    is inside (the legacy OR-keep, replicated as-is)."""
    bus_ids = bus["index"].to_numpy(dtype=np.int64)
    labels = _blocks(bus_ids, branch["f_bus"].to_numpy(),
                     branch["t_bus"].to_numpy())
    keep_label = int(np.argmax(np.bincount(labels)))
    block = set(bus_ids[labels == keep_label].tolist())

    bus_out = bus[bus["index"].isin(block)].reset_index(drop=True)
    gen_out = gen[gen["gen_bus"].isin(block)].reset_index(drop=True)
    load_out = load[load["load_bus"].isin(block)].reset_index(drop=True)
    br_keep = branch["f_bus"].isin(block) | branch["t_bus"].isin(block)
    return bus_out, branch[br_keep].reset_index(drop=True), gen_out, load_out


def renormalize_loads(load_kept: pd.DataFrame, target_pd: float,
                      target_qd: float) -> pd.DataFrame:
    """Conserve the pre-cut apportionment total onto the surviving loads
    (DEBUGGING D-6, DV-P2c-5).

    ``get_largest_grid`` drops every load outside the kept component, so the
    faithful S2 export sums to ``apportionment_total × component_survival``
    (~0.91 for CATS). The published CATS lineage instead re-apportioned its
    load target onto the final topology with the TOTAL conserved exactly.
    This scales the kept loads' pd (and qd) by ``target / Σ_kept`` so their
    sums equal the pre-cut apportionment totals — the loadsmith peak anchor
    (47,257 MW for the corrected CSV; the raw CSV total in general). P and Q
    are scaled INDEPENDENTLY, each to its own apportionment total (so both
    surviving totals match the input CSV; per-bus power factor shifts
    uniformly). Returns a copy; only invoked when
    ``cfg.loads.renormalize_post_cut`` (default off — see D8/D15 faithfulness).
    """
    out = load_kept.copy()
    kept_pd = float(out["pd"].to_numpy(float).sum())
    kept_qd = float(out["qd"].to_numpy(float).sum())
    if kept_pd != 0.0:
        out["pd"] = out["pd"].to_numpy(float) * (target_pd / kept_pd)
    if kept_qd != 0.0:
        out["qd"] = out["qd"].to_numpy(float) * (target_qd / kept_qd)
    return out


def finalize_frames(bus: pd.DataFrame, branch: pd.DataFrame,
                    gen: pd.DataFrame, load: pd.DataFrame) -> None:
    """add_component_status! + add_branch_defaults! + add_source_id!, on
    frames IN PLACE: bus_type=1 everywhere (no slack at S2), gen_status/
    br_status/status = 1, b_fr = b_to = NaN, transformer default False,
    source_id = '["<type>", "<id>"]' strings (the string-key overwrite the
    legacy add_source_id! performs — doc 02 quirk, replicated)."""
    bus["bus_type"] = 1
    gen["gen_status"] = 1
    load["status"] = 1
    branch["br_status"] = 1
    branch["b_fr"] = np.nan
    branch["b_to"] = np.nan
    if "transformer" not in branch.columns:
        branch["transformer"] = False
    branch["transformer"] = branch["transformer"].map(
        lambda v: bool(v) if pd.notna(v) else False)
    for df, comp in ((bus, "bus"), (branch, "branch"), (gen, "gen"),
                     (load, "load")):
        df["source_id"] = [f'["{comp}", "{i}"]' for i in df["index"]]


def audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """comp_dict_to_dataframe equivalent for the Plots/ audit CSVs: rows
    sorted by index, list/dict-like values stringified INTO THE COPY (the
    legacy mutated the grid in place before export — DV-P2c-3)."""
    out = df.sort_values("index", ignore_index=True).copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (list, dict))).any():
            out[col] = out[col].map(str)
    return out
