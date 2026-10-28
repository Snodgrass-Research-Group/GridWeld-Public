"""GridWeld orchestrator: checkpointed stage execution with resume.

Origin: the driver scripts — ``generate_grid_script.jl`` (S2 wiring, doc
02), ``legacy/Jonathan_Code/PSERCMain.m`` (S3/S4 wiring incl. the
``codectrl`` checkpoint-.mat control flow, doc 03 §PSERCMain), and
``PSERCMain_v2.m`` (S5 wiring, doc 05). The Reload/ReCalcSNet .mat-flag
mechanism becomes explicit checkpointed stages (pattern: gridforge/
gridcrucible ``core.py`` — copy the pattern, not the code, D5).

Stage order/name contract: :data:`gridweld.config.STAGES`. Each stage is a
``_stage_<name>`` method taking and returning a :class:`WeldModel`; after
each stage the model AND the rng bit-generator state are pickled to
``paths.checkpoint_dir`` so ``run(resume_from=...)`` reproduces the exact
continuation (D10 — one ``numpy.random.Generator`` threads through every
stochastic site).

Frame dialects across the run: S2 stages carry the internal snake_case
frames (``S2Grid.to_model``); ``merge_topology`` onward carries the
MATLAB-named S3+ frames (Num/From/To/RateA/…) that every params/dispatch/
resize/feasibility core consumes — the rename maps below are the single
boundary. ``model.meta`` carries the non-frame state (the S2 case, PF
matrices, logs); the DCOPF meta contract (``load_scenarios``/``pg_uc``) is
documented in :class:`gridweld.feasibility.dcopf.DCOPFUpgrader`.

Bus.Name provenance: the legacy Bus.Name came from old PowerWorld-era runs
and appears nowhere upstream of ReadMatpower — synthesized here as
str(NumExt) at the merge boundary (cosmetic; only the PW exports read it).

Phase: P5.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import STAGES, WeldConfig
from .model import WeldError, WeldModel

log = logging.getLogger(__name__)

#: internal (merge_topology output) → MATLAB S3+ dialect.
BUS_TO_MATLAB = {
    "bus_num": "Num", "external_num": "NumExt", "lat": "Lat", "lon": "Lon",
    "fid": "FID", "node_type": "Category", "import": "Import",
    "bus_type": "Type", "pd_mw": "MW", "qd_mvar": "MVar",
    "base_kv": "kVActual", "kv_class": "kV", "vm_pu": "Vm", "va_deg": "Va",
    "max_mw": "MaxMW", "min_mw": "MinMW", "region": "Region",
}
GEN_TO_MATLAB = {
    "bus": "BusNum", "unit_id": "ID", "pmin_mw": "MinMW", "pmax_mw": "MaxMW",
    "qmin_mvar": "MinMVar", "qmax_mvar": "MaxMVar", "vg_pu": "Vg",
    "fuel_type": "FuelType", "plant_code": "PlantCode",
    "eia_gen_id": "GenID", "region": "Region",
    "c0": "A", "c1": "B", "c2": "C",
}
BRANCH_TO_MATLAB = {
    "fid": "FID", "voltage_kv_from": "kVFrom", "voltage_kv_to": "kVTo",
    "length_mi": "Length", "path_length_mi": "PathLength",
    "n_circuits": "Circuits", "kcmil": "Size",
    "n_cond_per_phase": "CondPerPhase", "capacity_mva": "RateA",
    "gmd_ft": "GMD", "r_pu": "R", "x_pu": "X", "b_pu": "B",
    "overloaded": "Overloaded", "undersized": "Undersized",
    "tower_index": "TowerIndex", "resize": "Resize",
    "conductor_idx": "CondIndex", "max_conductor_idx": "MaxCondIndex",
    "min_conductor_idx": "MinCondIndex", "structure": "Structure",
    "material": "Material", "lat1": "Lat1", "lon1": "Lon1", "lat2": "Lat2",
    "lon2": "Lon2", "from_bus": "From", "to_bus": "To", "tap": "Tap",
}
#: MATLAB → internal (the parameterize → merge boundary).
BRANCH_TO_INTERNAL = {v: k for k, v in BRANCH_TO_MATLAB.items()}


def _matlab_frames_to_internal(bus_df: pd.DataFrame, branch_df: pd.DataFrame
                               ) -> WeldModel:
    """MATLAB S3+ frames → the minimal internal WeldModel the power/
    reactive kernels duck-type (0-based ids)."""
    n = len(bus_df)
    return WeldModel(
        buses=pd.DataFrame({"bus_id": np.arange(n, dtype=np.int64)}),
        branches=pd.DataFrame({
            "from_bus": branch_df["From"].to_numpy(np.int64) - 1,
            "to_bus": branch_df["To"].to_numpy(np.int64) - 1,
            "tap": branch_df["Tap"].to_numpy(float),
            "r_pu": branch_df["R"].to_numpy(float),
            "x_pu": branch_df["X"].to_numpy(float),
            "b_pu": branch_df["B"].to_numpy(float),
            "capacity_mva": branch_df["RateA"].to_numpy(float),
        }))


class GridWeld:
    """Pipeline orchestrator: run/resume the configured stage sequence.

    Parameters
    ----------
    config:
        Fully-validated run configuration (one TOML file == one run).
    """

    def __init__(self, config: WeldConfig) -> None:
        self.config = config
        self.rng: np.random.Generator = np.random.default_rng(
            config.system.rng_seed)
        self._data_cache: dict[str, object] = {}

    # -- shared data tables (catalogs, transformer tables) ------------------

    def _data(self, name: str):
        if name not in self._data_cache:
            from .params.conductors import (crucible_catalog,
                                            load_cats_catalog, matlab_catalog)
            from .params.impedance import load_gmd_tables
            data_dir = Path(__file__).resolve().parent / "data"
            loaders = {
                "cats_catalog": load_cats_catalog,
                "crucible_catalog": crucible_catalog,
                "matlab_catalog": matlab_catalog,
                "gmd_tables": load_gmd_tables,
                "xfdata": lambda: pd.read_parquet(data_dir / "xfdata.parquet"),
                "xfxr": lambda: pd.read_parquet(data_dir / "xfxr.parquet"),
                "form1": lambda: pd.read_parquet(data_dir / "form1.parquet"),
                "tower": lambda: pd.read_parquet(data_dir / "tower.parquet"),
            }
            self._data_cache[name] = loaders[name]()
        return self._data_cache[name]

    # -- checkpoints ---------------------------------------------------------

    def checkpoint_path(self, stage: str) -> Path:
        """Location of a stage's checkpoint under ``paths.checkpoint_dir``
        (index-prefixed so listings sort in execution order)."""
        return (Path(self.config.paths.checkpoint_dir)
                / f"{STAGES.index(stage):02d}_{stage}.pkl")

    def _save_checkpoint(self, stage: str, model: WeldModel) -> None:
        path = self.checkpoint_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"stage": stage, "model": model,
                         "rng_state": self.rng.bit_generator.state}, f)

    def _load_checkpoint(self, stage: str) -> WeldModel:
        path = self.checkpoint_path(stage)
        if not path.exists():
            raise WeldError(f"no checkpoint for stage '{stage}' at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = payload["rng_state"]
        return payload["model"]

    # -- driver --------------------------------------------------------------

    def run(self, resume_from: str | None = None) -> WeldModel:
        """Execute the configured stages in order, checkpointing after each.

        Parameters
        ----------
        resume_from:
            Stage name to resume at (the PRECEDING configured stage's
            checkpoint is loaded, restoring model + rng state), or None to
            run from the first configured stage.
        """
        stages = list(self.config.stages)
        model = WeldModel()
        if resume_from is not None:
            if resume_from not in stages:
                raise WeldError(f"resume_from '{resume_from}' not in "
                                f"configured stages {stages}")
            idx = stages.index(resume_from)
            if idx > 0:
                model = self._load_checkpoint(stages[idx - 1])
            stages = stages[idx:]
        for stage in stages:
            fn = getattr(self, f"_stage_{stage}")
            log.info("stage %s ...", stage)
            t0 = time.perf_counter()
            model = fn(model)
            self._save_checkpoint(stage, model)
            log.info("stage %s done in %.1f s (checkpoint %s)", stage,
                     time.perf_counter() - t0, self.checkpoint_path(stage))
        return model

    # -- stages --------------------------------------------------------------

    def _stage_gis_clean(self, model: WeldModel) -> WeldModel:
        """S1 — arcpy backend, ArcGIS Pro propy runtime ONLY (P2d). The
        evergreen interpreter cannot run it; the CATS cleaned layers are
        checked in at the [paths] keys, so the standard run starts at
        'build'."""
        raise WeldError(
            "gis_clean runs under ArcGIS Pro's propy (gridweld.gis."
            "arcpy_backend, P2d workflow — batch runtime comparison still "
            "pending). Drop 'gis_clean' from [stages] and start at 'build': "
            "the pipeline consumes the cleaned layers at [paths].")

    def _stage_build(self, model: WeldModel) -> WeldModel:
        """S2 (doc 02): full Julia-script equivalent; exports the
        cal_largest_grid-analog case + the S2 QA reports (the Julia script
        emitted them during S2 — doc 07)."""
        from .build import fold_loads_into_buses, run_build
        from .io.matpower import model_to_case, write_matpower_case
        from .viz.plots import write_qa_reports

        cfg = self.config
        _, largest, artifacts = run_build(cfg, self.rng)
        model = largest.to_model(cfg)
        case = model_to_case(fold_loads_into_buses(model))
        out_dir = Path(cfg.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_matpower_case(case, out_dir / "cal_largest_grid.m")
        write_qa_reports(model, out_dir / "qa", cfg)
        model.meta["s2_case"] = case
        model.meta["s2_artifacts"] = artifacts     # frames + MilpResult
        return model

    def _stage_parameterize(self, model: WeldModel) -> WeldModel:
        """S3 steps 3-9 (doc 03): raw GIS CSV → Form 1 seeding → SizeLines
        → RXB (the P3a fixture-exact chain). Output: the MATLAB-named
        parameterized GIS branch table in meta['s3_branch_gis']."""
        from .params.conductors import (init_branch_table, seed_from_form1,
                                        size_lines)
        from .params.impedance import compute_line_impedances

        cfg = self.config
        raw = pd.read_csv(cfg.paths.resolve(cfg.paths.gis_branch_table),
                          low_memory=False, encoding_errors="replace")
        br = init_branch_table(raw, cfg, self.rng)
        br = seed_from_form1(br, self._data("form1"), self._data("tower"),
                             self._data("matlab_catalog"), cfg)
        br = size_lines(br, self._data("matlab_catalog"))
        rxb_in = br.rename(columns={
            "Length": "length_mi", "GMD": "gmd_ft",
            "CondIndex": "conductor_idx", "kVFrom": "voltage_kv_from",
            "Circuits": "n_circuits", "RateA": "capacity_mva", "FID": "fid"})
        rxb_out = compute_line_impedances(rxb_in, self._data("cats_catalog"),
                                          self._data("gmd_tables"), cfg,
                                          matlab_compat=True)
        br["R"] = rxb_out["r_pu"].to_numpy()
        br["X"] = rxb_out["x_pu"].to_numpy()
        br["B"] = rxb_out["b_pu"].to_numpy()
        br["RateA"] = rxb_out["capacity_mva"].to_numpy()
        model.meta["s3_branch_gis"] = br
        return model

    def _stage_merge_topology(self, model: WeldModel) -> WeldModel:
        """S3 step 10 (ReadMatpower): FID-join the parameterized GIS table
        with the S2 case; frames switch to the MATLAB S3+ dialect here."""
        from .io.matpower import merge_topology

        cfg = self.config
        gis = model.meta["s3_branch_gis"].rename(columns=BRANCH_TO_INTERNAL)
        bus, branch, gen, params = merge_topology(gis, model.meta["s2_case"],
                                                  cfg, matlab_compat=True)
        bus = bus.rename(columns=BUS_TO_MATLAB)
        bus["Name"] = bus[("NumExt" if "NumExt" in bus.columns
                           else "Num")].astype(str)     # synthesized (module
        model.buses = bus                               # docstring)
        model.branches = branch.rename(columns=BRANCH_TO_MATLAB)
        model.generators = gen.rename(columns=GEN_TO_MATLAB)
        model.meta["params"] = params
        del model.meta["s3_branch_gis"]
        return model

    def _stage_scenarios(self, model: WeldModel) -> WeldModel:
        """S3 step 11 (CreateSNet via UCED/UCCD): ScenarioSet + the PF
        fields the DCOPF stage needs (load_scenarios/pg_uc — the S5 meta
        contract)."""
        from .dispatch.snet import build_snet

        extras: dict = {}
        model.scenarios = build_snet(model.buses, model.generators,
                                     self.config, extras=extras)
        model.meta["load_scenarios"] = extras["load_scenarios"]
        model.meta["pg_uc"] = extras["pg"]
        return model

    def _stage_size_transformers(self, model: WeldModel) -> WeldModel:
        """S3 step 12 (SizeTransformers): placeholder bus-ties → real
        ratings from DC duty across the deduplicated ('complete') tier."""
        from .params.transformers import size_transformers

        cfg = self.config
        model.branches = size_transformers(
            model.branches, model.scenarios.tiers["complete"],
            len(model.buses), self._data("xfdata"), self._data("xfxr"),
            self._data("matlab_catalog"), self._data("crucible_catalog"),
            cfg, self.rng, self._data("gmd_tables"),
            self._data("cats_catalog"))
        return model

    def _stage_dc_resize(self, model: WeldModel) -> WeldModel:
        """S4 (doc 04 §8): the DCPF → ResizeBranches loop."""
        from .resize.dc_loop import run_dc_resize

        cfg = self.config
        model.branches, dc_log = run_dc_resize(
            model.branches, model.scenarios.tiers["full"], len(model.buses),
            self._data("xfdata"), self._data("xfxr"),
            self._data("crucible_catalog"), self._data("cats_catalog"),
            self._data("gmd_tables"), cfg, self.rng,
            max_passes=cfg.resize.max_passes)
        model.meta["dc_resize_log"] = dc_log
        return model

    def _stage_dcopf(self, model: WeldModel) -> WeldModel:
        """S5 (doc 05): assigning_pmax_pmin (PSERCMain_v2:302-303) then the
        DCOPF feasibility loop."""
        from .build.generators import assign_gen_limits
        from .feasibility.dcopf import DCOPFUpgrader

        cfg = self.config
        eia = pd.read_excel(cfg.paths.resolve(cfg.paths.eia_limits))
        model.generators = assign_gen_limits(model.generators, eia,
                                             state=cfg.eia.state)
        upgrader = DCOPFUpgrader(
            cfg, self._data("cats_catalog"),
            catalog_crucible=self._data("crucible_catalog"),
            xfdata=self._data("xfdata"), xfxr=self._data("xfxr"),
            gmd_tables=self._data("gmd_tables"))
        return upgrader.upgrade(model, self.rng,
                                max_passes=cfg.feasibility.max_passes)

    def _stage_reactive(self, model: WeldModel) -> WeldModel:
        """S4/S5 (doc 04 §9): ReactivePowerPlacement winnowing; Bus.SyncCond
        + the AC-side snet/pv/pq state land in meta for ac_resize."""
        from .resize.reactive import (correct_power_factors,
                                      duplicate_max_scenario)
        from .resize.reactive import place_reactive_support

        cfg = self.config
        scen = model.scenarios
        snet_ac = correct_power_factors(
            duplicate_max_scenario(scen.tiers["complete"], cfg))
        internal = _matlab_frames_to_internal(model.buses, model.branches)
        internal.generators = pd.DataFrame({
            "bus": model.generators["BusNum"].to_numpy(np.int64) - 1})
        slack0 = int(scen.slack_bus) - 1
        result = place_reactive_support(internal, snet_ac, slack0, cfg)
        model.buses["SyncCond"] = result["sync_cond"].astype(float)
        model.meta["snet_ac"] = snet_ac
        model.meta["reactive"] = {k: result[k] for k in
                                  ("pv", "pq", "converged", "iterations")}
        return model

    def _stage_ac_resize(self, model: WeldModel) -> WeldModel:
        """S4/S5 (doc 04 §10): MiddleLoopAC on the ACEngine."""
        from .resize.ac_loop import run_ac_resize

        cfg = self.config
        scen = model.scenarios
        r = model.meta["reactive"]
        model.branches, ac_log = run_ac_resize(
            model.branches, model.meta["snet_ac"], len(model.buses),
            int(scen.slack_bus) - 1, r["pv"], r["pq"], scen.v_setpoints,
            self._data("xfdata"), self._data("xfxr"),
            self._data("crucible_catalog"), self._data("cats_catalog"),
            self._data("gmd_tables"), cfg, self.rng,
            max_passes=cfg.resize.max_passes)
        model.meta["ac_resize_log"] = ac_log
        # D-7 exit record: {exit, returned_pass, best_converged_fraction}
        model.meta["ac_resize_exit"] = dict(ac_log.attrs)
        return model

    def _stage_export(self, model: WeldModel) -> WeldModel:
        """S7 (doc 07): final MATPOWER case + WriteLoads + the PowerWorld
        imports + (when the GIS layers are configured) the GeoJSON
        round-trip."""
        from .io.geojson import roundtrip_gis
        from .io.matpower import write_matpower_case
        from .io.powerworld import (write_load_timeseries,
                                    write_powerworld_imports)

        cfg = self.config
        out_dir = Path(cfg.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        case = self._final_case(model)
        case_path = out_dir / "final_case.m"
        write_matpower_case(case, case_path)

        snet = (model.scenarios.tiers["full"]
                if model.scenarios is not None else None)
        artifacts: dict[str, Path] = {"final_case": case_path}
        if snet is not None:
            mw, mvar = write_load_timeseries(
                snet, out_dir, s_base=cfg.system.s_base_mva, units="mw")
            artifacts["timeseries_mw"] = mw
            artifacts["timeseries_mvar"] = mvar
        artifacts.update(write_powerworld_imports(
            model.buses, model.branches, model.generators, snet,
            out_dir, cfg, ac_stage="ac_resize" in cfg.stages, units="mw"))

        s2_case_path = out_dir / "cal_largest_grid.m"
        layers_ok = all(cfg.paths.resolve(p).exists() for p in
                        (cfg.paths.gis_lines, cfg.paths.gis_added_nodes,
                         cfg.paths.tl_extra))
        if s2_case_path.exists() and layers_ok:
            artifacts.update(roundtrip_gis(s2_case_path, case_path,
                                           out_dir, cfg))
        else:
            log.warning("GeoJSON round-trip skipped (S2 case or GIS layers "
                        "not available)")
        model.meta["export_artifacts"] = artifacts
        return model

    def _final_case(self, model: WeldModel):
        """MATLAB S3+ frames → MATPOWER case (Table2Matpower analog):
        internal 0-based frames for model_to_case; baseMVA = system SBase
        (the S3+ convention — the S2 case stays at build_base_mva)."""
        from .io.matpower import model_to_case

        bus, br, gen = model.buses, model.branches, model.generators
        internal = WeldModel(
            buses=pd.DataFrame({
                "bus_id": bus["Num"].to_numpy(np.int64) - 1,
                "external_num": bus["Num"].to_numpy(np.int64),
                "bus_type": bus["Type"].to_numpy(np.int64),
                "pd_mw": bus["MW"].to_numpy(float),
                "qd_mvar": bus["MVar"].to_numpy(float),
                "vm_pu": bus.get("Vm", pd.Series(1.0, index=bus.index)),
                # DV-07g: the S5 loop's Bus.Va is RADIANS — the MATPOWER VA
                # column (and PowerWorld Vangle) are degree-domain
                "va_deg": np.degrees(
                    bus.get("Va", pd.Series(0.0, index=bus.index))
                    .to_numpy(float)),
                "base_kv": bus["kVActual"].to_numpy(float),
            }),
            branches=pd.DataFrame({
                "from_bus": br["From"].to_numpy(np.int64) - 1,
                "to_bus": br["To"].to_numpy(np.int64) - 1,
                "r_pu": br["R"].to_numpy(float),
                "x_pu": br["X"].to_numpy(float),
                "b_pu": br["B"].to_numpy(float),
                "capacity_mva": br["RateA"].to_numpy(float),
                "tap": br["Tap"].to_numpy(float),
            }),
            generators=pd.DataFrame({
                "bus": gen["BusNum"].to_numpy(np.int64) - 1,
                "pmax_mw": gen["MaxMW"].to_numpy(float),
                "pmin_mw": gen["MinMW"].to_numpy(float),
                "qmax_mvar": gen["MaxMVar"].to_numpy(float),
                "qmin_mvar": gen["MinMVar"].to_numpy(float),
                "vg_pu": gen["Vg"].to_numpy(float),
                "c2": gen["C"].to_numpy(float),
                "c1": gen["B"].to_numpy(float),
                "c0": gen["A"].to_numpy(float),
            }),
            meta={"name": self.config.system.name.replace(" ", "_"),
                  "base_mva": self.config.system.s_base_mva})
        return model_to_case(internal)


def main() -> None:
    """CLI entry point (``gridweld`` console script)."""
    parser = argparse.ArgumentParser(
        prog="gridweld",
        description="Run the GridWeld pipeline from a TOML run file.")
    parser.add_argument("config", nargs="?",
                        default=str(Path("configs") / "cats.toml"),
                        help="run configuration (default: configs/cats.toml)")
    parser.add_argument("--resume-from", metavar="STAGE", default=None,
                        help="resume at STAGE from the preceding checkpoint")
    parser.add_argument("--stages", default=None,
                        help="comma-separated stage subset (canonical order)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = WeldConfig.from_toml(args.config)
    if args.stages:
        cfg.stages = [s.strip() for s in args.stages.split(",")]
        problems = cfg.validate()
        if problems:
            parser.error("; ".join(problems))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose or cfg.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    GridWeld(cfg).run(resume_from=args.resume_from)
