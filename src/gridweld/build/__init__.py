"""gridweld.build — S2 network build (algorithm doc 02).

Python equivalent of ``legacy/generate_grid_script.jl`` +
``helper_functions.jl`` + ``prototyping_code/optim_load_assign.jl``. The
stage ORDER is semantically load-bearing (doc 02 §execution order):
substations → generators(+costs) → loads → snap gens (SUBSTATIONS ONLY) →
nodes → import gens → branches → snap branches → MILP loads →
transformers → finalize → self-loops → largest component.

``run_build`` returns :class:`S2Grid` — the four Julia-named frames every
p1–p5 fixture mirrors — plus ``to_model``/`to_case`` bridges to the D6
WeldModel / MATPOWER export (io.matpower).

Phase: P2c. ⚠ DV6 (Q5 "P load is INCORRECT") must be resolved by the author
before the load fixtures are trusted as final — validated against, not
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import MilpResult, WeldModel
from . import costs, generators, loads, reduce, snapping, substations, transformers


@dataclass
class S2Grid:
    """S2 working state: Julia-named frames (1-based ``index`` ids)."""

    bus: pd.DataFrame
    branch: pd.DataFrame
    gen: pd.DataFrame
    load: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> tuple[int, int, int, int]:
        return len(self.bus), len(self.branch), len(self.gen), len(self.load)

    def to_model(self, cfg: WeldConfig) -> WeldModel:
        """S2Grid → WeldModel (0-based internal ids; loads folded into bus
        pd/qd happens at MATPOWER export, not here)."""
        bus = self.bus.sort_values("index", ignore_index=True)
        ext2int = pd.Series(np.arange(len(bus), dtype=np.int64),
                            index=bus["index"].to_numpy(np.int64))
        buses = pd.DataFrame({
            "bus_id": np.arange(len(bus), dtype=np.int64),
            "external_num": bus["index"].to_numpy(np.int64),
            "fid": bus["FID"].to_numpy(),
            "lat": bus["Lat"].to_numpy(float),
            "lon": bus["Lon"].to_numpy(float),
            "node_type": bus["Type"].to_numpy(),
            "import_flag": bus["Import"].astype(str).str.strip().eq("IMPORT"),
            "base_kv": bus["base_kv"].to_numpy(float),
            "bus_type": bus.get("bus_type", pd.Series(1, index=bus.index)),
        })
        br = self.branch
        branches = pd.DataFrame({
            "from_bus": ext2int.loc[br["f_bus"].to_numpy(np.int64)].to_numpy(),
            "to_bus": ext2int.loc[br["t_bus"].to_numpy(np.int64)].to_numpy(),
            "fid": br.get("FID", np.nan),
            "lat1": br.get("Lat1", np.nan), "lon1": br.get("Lon1", np.nan),
            "lat2": br.get("Lat2", np.nan), "lon2": br.get("Lon2", np.nan),
            "voltage_kv_from": br.get("kV", np.nan),
            "voltage_kv_to": br.get("kV", np.nan),
            "tap": br.get("tap", pd.Series(0.0, index=br.index)).fillna(0.0),
            "shift": br.get("shift", pd.Series(0.0, index=br.index)).fillna(0.0),
            "is_transformer": br.get("transformer",
                                     pd.Series(False, index=br.index)),
            "status": br.get("br_status", pd.Series(1, index=br.index)),
            "f_bus_distance_km": br.get("f_bus_distance", np.nan),
            "t_bus_distance_km": br.get("t_bus_distance", np.nan),
        })
        g = self.gen
        gens = pd.DataFrame({
            "gen_id": np.arange(len(g), dtype=np.int64),
            "bus": ext2int.loc[g["gen_bus"].to_numpy(np.int64)].to_numpy(),
            "plant_code": g["PlantCode"].to_numpy(),
            "eia_gen_id": g["GenID"].astype(str).to_numpy(),
            "fuel_type": g["FuelType"].to_numpy(),
            "pmax_mw": g["pmax"].to_numpy(float),
            "pmin_mw": 0.0, "qmax_mvar": 0.0, "qmin_mvar": 0.0,
            "lat": g["Lat"].to_numpy(float),
            "lon": g["Lon"].to_numpy(float),
            "gen_bus_distance_km": g.get("gen_bus_distance", np.nan),
            "status": g.get("gen_status", pd.Series(1, index=g.index)),
            "vg_pu": 0.0, "cost_model": 2, "ncost": 3,
            "startup_usd": g["startup"].to_numpy(float),
            "shutdown_usd": g["shutdown"].to_numpy(float),
            "c2": g["c2"].to_numpy(float),
            "c1": g["c1"].to_numpy(float),
            "c0": g["c0"].to_numpy(float),
        })
        ld = self.load
        load_out = pd.DataFrame({
            "load_id": np.arange(len(ld), dtype=np.int64),
            "bus": ext2int.loc[ld["load_bus"].to_numpy(np.int64)].to_numpy(),
            "pd_mw": ld["pd"].to_numpy(float),        # ⚠ Q5/DV6 unresolved
            "qd_mvar": ld["qd"].to_numpy(float),
            "lat": ld["Lat"].to_numpy(float),
            "lon": ld["Lon"].to_numpy(float),
            "load_bus_distance_km": ld["load_bus_distance"].to_numpy(float),
            "status": ld.get("status", pd.Series(1, index=ld.index)),
        })
        meta = dict(self.meta)
        meta.setdefault("name", cfg.system.name)
        meta.setdefault("base_mva", cfg.system.build_base_mva)
        meta.setdefault("per_unit", True)
        return WeldModel(buses=buses, branches=branches, generators=gens,
                         loads=load_out, meta=meta)


def fold_loads_into_buses(model: WeldModel) -> WeldModel:
    """MATPOWER has no load table: per-bus pd/qd sums land on the bus frame
    (doc 02 output contract) — applied at export time only."""
    out = model.copy()
    pd_sum = out.loads.groupby("bus")["pd_mw"].sum()
    qd_sum = out.loads.groupby("bus")["qd_mvar"].sum()
    ids = out.buses["bus_id"]
    out.buses["pd_mw"] = pd_sum.reindex(ids).fillna(0.0).to_numpy()
    out.buses["qd_mvar"] = qd_sum.reindex(ids).fillna(0.0).to_numpy()
    out.buses["vm_pu"] = 0.0
    out.buses["va_deg"] = 0.0
    return out


def run_build(cfg: WeldConfig, rng: np.random.Generator, *,
              solver: str | None = None
              ) -> tuple[S2Grid, S2Grid, dict[str, Any]]:
    """The full S2 pipeline. Returns (full grid, largest grid, artifacts).

    ``artifacts``: tiny_lines / self_loops audit frames + the MilpResult
    (doc 02 side CSVs Plots/{tiny_lines,self_loops}.csv).
    """
    p = cfg.paths
    subs = substations.load_substations(p.resolve(p.substations))
    nodes = substations.load_added_nodes(p.resolve(p.added_nodes), len(subs))
    gen = generators.load_generators(p.resolve(p.eia_generators),
                                     p.resolve(p.eia_plants), cfg)
    gen = costs.assign_costs(gen, p.resolve(p.cost_curves), cfg)
    gen = gen[generators.GEN_COLUMNS].copy()
    load = loads.load_loads(p.resolve(p.load_apportionment))
    lines, tiny = reduce.clean_transmission(p.resolve(p.transmission), cfg)

    # snap generators BEFORE added nodes exist (substations only, doc 02 §4)
    bus = substations.make_bus_table(subs)
    gen = snapping.snap_points(gen, bus, "gen",
                               warn_km=cfg.snapping.warn_km,
                               earth_radius_km=cfg.geo.earth_radius_km)

    bus = substations.make_bus_table(subs, nodes)
    gen = generators.add_import_generators(gen, nodes, subs, bus, cfg, rng)
    branch = lines.copy()
    branch = snapping.snap_branch_endpoints(
        branch, bus, warn_km=cfg.snapping.warn_km,
        earth_radius_km=cfg.geo.earth_radius_km)

    load, milp = loads.assign_loads(load, bus, branch, gen, cfg,
                                    solver=solver)

    bus, branch, gen, load = transformers.create_transformers(
        bus, branch, gen, load, cfg, rng)
    reduce.finalize_frames(bus, branch, gen, load)
    branch, self_loops = reduce.remove_self_loops(branch)

    full = S2Grid(bus, branch, gen, load,
                  meta={"name": cfg.system.name,
                        "base_mva": cfg.system.build_base_mva,
                        "per_unit": True})
    lb, lbr, lg, ll = reduce.largest_component(bus, branch, gen, load)
    if cfg.loads.renormalize_post_cut:                 # DV-P2c-5 (D-6)
        target_pd = float(load["pd"].to_numpy(float).sum())
        target_qd = float(load["qd"].to_numpy(float).sum())
        ll = reduce.renormalize_loads(ll, target_pd, target_qd)
    largest = S2Grid(lb, lbr, lg, ll, meta=dict(full.meta))
    artifacts = {"tiny_lines": tiny, "self_loops": self_loops, "milp": milp}
    return full, largest, artifacts


__all__ = ["S2Grid", "run_build", "fold_loads_into_buses", "MilpResult"]
