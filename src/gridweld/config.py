"""WeldConfig: TOML-backed run configuration for the GridWeld pipeline.

Replaces FOUR legacy mechanisms:
* the ``options`` struct from ``legacy/Jonathan_Code/Matlab Functions/SetOptions.m``
  (algorithm doc 03 §SetOptions — every constant there appears below),
* the ``codectrl`` flag block hand-edited at the top of ``PSERCMain.m`` /
  ``PSERCMain_v2.m`` (checkpoint .mat control flow becomes explicit ``stages``),
* the constants scattered through ``generate_grid_script.jl`` (doc 02 config table),
* the hard-coded tolerances of the EV-repo GIS scripts (doc 01 config table).

Region-agnostic by design (D8): every CATS/California-specific value lives in
``configs/cats.toml`` and loads over these defaults. The defaults below ARE the
CATS values so a bare ``WeldConfig()`` reproduces the legacy run.

Pattern-copy of ``gridcrucible.config.CrucibleConfig`` (D5: copy the pattern,
not the code). Strict loading: unknown sections/keys are errors.

Randomness (D10): ONE ``numpy.random.Generator`` seeded from ``system.rng_seed``
drives every stochastic site (S2 lat/lon jitter, S3 ``randi(3)`` TowerIndex,
S4 ``randperm`` resize draws). Algorithm doc 02 proposed a separate
``[build] seed``; D10 supersedes it — there is only this one seed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

#: Repo root, anchored on this file (src/gridweld/config.py) so default paths
#: work regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pipeline stage names in execution order (README §2/§4; doc 03 §PSERCMain
#: Python design; doc 05 §stage intent). S6 evaluation is Julia-side
#: (julia/GridWeldEval) and not a Python stage; ``[eval]`` below only carries
#: its config so one TOML file describes the whole run.
STAGES = (
    "gis_clean",          # S1 (doc 01)  — arcpy backend, P2d
    "build",              # S2 (doc 02)  — Julia-script equivalent, P2c
    "parameterize",       # S3 (doc 03 steps 3-9) — conductors/lengths/RXB, P3
    "merge_topology",     # S3 (doc 03 step 10)   — ReadMatpower FID join, P2a/P3
    "scenarios",          # S3 (doc 03 step 11)   — CreateSNet via UCED/UCCD, P3
    "size_transformers",  # S3 (doc 03 step 12)   — SizeTransformers, P3
    "dc_resize",          # S4 (doc 04 §8)        — DCPF → ResizeBranches loop, P3
    "dcopf",              # S5 (doc 05)           — DCOPF feasibility loop, P4
    "reactive",           # S4/S5 (doc 04 §9)     — ReactivePowerPlacement, P3
    "ac_resize",          # S4/S5 (doc 04 §10)    — MiddleLoopAC, P3
    "export",             # S7 (doc 07)           — PowerWorld + GeoJSON round-trip, P5
)


# ---------------------------------------------------------------------------
# [paths] — every file the legacy pipeline hard-coded (doc 02 I/O contract,
# doc 03 §Input data). Fixes Q2/DV2 (Julia read `data/…` vs on-disk `Data/`).
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """[paths] — pipeline inputs/outputs. All relative paths resolve against
    ``root`` (the legacy data tree stays valid for validation runs)."""

    root: Path = REPO_ROOT / "legacy"
    # S2 build inputs (doc 02, the 7 CSVs of generate_grid_script.jl)
    substations: Path = Path("Data/California_Electric_Substations_11_17_21.csv")
    added_nodes: Path = Path("Data/Added_Nodes_11_17_21.csv")
    eia_generators: Path = Path("Data/EIA_Generator_Y2019.csv")
    eia_plants: Path = Path("Data/EIA_Plant_Y2019.csv")
    cost_curves: Path = Path("Data/Generator_Cost_Curves.csv")
    # DV6/D-1: DEFAULT flipped 2026-07-20 to the corrected loadsmith CAISO
    # apportionment (Σ pd = 47,257 MW peak anchor; C4 renormalization). The
    # retired legacy file (Data/California_load_approtionment.csv, a ~×1.25
    # overstatement) stays pinned in the P2c fixture-parser test. Same 8,034
    # points/index/Lat/Lon as legacy — only pd/qd magnitudes differ, so the
    # MILP assignment and all topology are invariant.
    # Files shipped in this repository are absolute (anchored on REPO_ROOT),
    # so they still resolve when ``root`` points at another data checkout.
    load_apportionment: Path = (
        REPO_ROOT / "src" / "gridweld" / "data" / "caiso_loads_corrected.csv")
    transmission: Path = Path("Jonathan_Code/Input Data/CAGISData Removed planned lines.csv")
    # S3 parameterization inputs (doc 03 §Input data)
    form1: Path = Path("Jonathan_Code/Input Data/Form1.csv")
    tower_data: Path = Path("Jonathan_Code/Input Data/TowerData.csv")
    conductor_dir: Path = Path("Jonathan_Code/Data Files/Conductor Data Tables")
    xfdata_csv: Path = REPO_ROOT / "validation" / "fixtures" / "matlab" / "xfdata.csv"  # B1
    xfxr_csv: Path = REPO_ROOT / "validation" / "fixtures" / "matlab" / "xfxr.csv"      # B1
    # S3 raw GIS branch table (PSERCMain reads CAGISData.csv — NOT the
    # "Removed planned lines" S2 input; P3a fixtures pin this)
    gis_branch_table: Path = Path("Jonathan_Code/Input Data/CAGISData.csv")
    # S5 prep (assigning_pmax_pmin, doc 05)
    eia_limits: Path = Path("Jonathan_Code/Aditya_Code/Data Files/"
                            "EIA_Generators_Combined_2010.xlsx")
    # S7 GeoJSON round-trip inputs (reformat_GIS.jl, doc 07)
    gis_lines: Path = Path("Data/GIS/lines.geojson")
    gis_substations: Path = Path("Data/GIS/substations.geojson")
    gis_added_nodes: Path = Path("Data/GIS/added_nodes.geojson")
    tl_extra: Path = Path("Data/CATS_transmission_line_data.csv")
    # outputs
    out_dir: Path = REPO_ROOT / "outputs"
    log_dir: Path = REPO_ROOT / "outputs" / "logs"
    checkpoint_dir: Path = REPO_ROOT / "checkpoints"

    def resolve(self, p: Path) -> Path:
        """Absolute path for an input: relative paths are joined onto ``root``."""
        return p if p.is_absolute() else self.root / p


# ---------------------------------------------------------------------------
# [system] — shared physical/system constants
# ---------------------------------------------------------------------------


@dataclass
class SystemConfig:
    """[system] — constants shared across stages (docs 02/03/04/05)."""

    name: str = "California SynGrid"      # grid["name"], generate_grid_script.jl
    region: str = "CAISO"                 # options.Region, PSERCMain.m
    s_base_mva: float = 100.0             # options.SBase; ALSO the literal 100 in RXB
                                          # Zbase, DCPowerFlow* MW scaling, Table2Matpower
                                          # baseMVA (docs 03/04/05 all share this key)
    build_base_mva: float = 1.0           # grid["baseMVA"]=1 with per_unit=true at S2
                                          # export (doc 02; flagged "Not verified"
                                          # in-source; S6 eval DEPENDS on it — doc 06)
    voltage_levels: list[int] = field(default_factory=lambda: [66, 115, 230, 500])
                                          # options.VoltageLevels ≡ doc 02 kv remap set
    km_per_mile: float = 1.60934          # PSERCMain conversion (⚠ gridforge hard-codes
                                          # 1.61 — do NOT inherit; doc 03 §DistanceCalc)
    rng_seed: int = 42                    # D10 single Generator seed (dump scripts
                                          # used rng(42); keep aligned with fixtures)


@dataclass
class GeoConfig:
    """[geo] — geodesy for snapping/lengths (doc 02 §snapping; doc 03
    §DistanceCalc). NOTE: must stay consistent with ``gis.earth_radius_m``."""

    earth_radius_km: float = 6371.0       # haversine R in lat_lon_distance (Julia)
                                          # and DistanceCalc.m (6371e3 m / 1000)


# ---------------------------------------------------------------------------
# [gis] — S1 cleaning (doc 01 config table, verbatim keys)
# ---------------------------------------------------------------------------


@dataclass
class GISCleanParams:
    """One clean-parameter set (doc 01 §Feeder_Cleanser + knobs table).
    Units: metres unless suffixed otherwise."""

    feature_to_line_tol_m: float = 15.0   # arcpy.FeatureToLine cluster tolerance
    generalize_tol_m: float = 30.0        # Douglas-Peucker; 0.0 = skip (transmission)
    extend_line_m: float = 10.0           # dangle closure
    snap_edge_m: float = 5.0              # self-EDGE snap (transmission: substation VERTEX)
    attribute_map_tol_m: float = 50.0     # Ruiz spatial-join Tol (Method1 used 0 + fill)
    spatial_join_radius_m: float = 10.0   # legacy SpatialJoin_analysis search_radius
    default_kv: str = ""                  # feeders: "12kV"; transmission: "" = error
    dedup_node_tol_m: float = 0.0         # DeleteIdentical "2 Meter" (transmission)
    substation_overlap_radius_m: float = 0.0  # Near_analysis "12" (transmission)
    tap_split_radius_m: float = 0.0       # SplitLineAtPoint "1 Meter" (transmission)


def _distribution_clean() -> GISCleanParams:
    return GISCleanParams(default_kv="12kV")


def _transmission_clean() -> GISCleanParams:
    return GISCleanParams(
        feature_to_line_tol_m=1.0,        # 15 m would fuse double-circuit corridors
        generalize_tol_m=0.0,             # preserve geodesic Length_mi for R/X/B
        attribute_map_tol_m=0.0,
        spatial_join_radius_m=0.0,
        default_kv="",                    # missing transmission kV is an ERROR
        dedup_node_tol_m=2.0,
        substation_overlap_radius_m=12.0,
        tap_split_radius_m=1.0,
    )


@dataclass
class GISCleanConfig:
    """[gis.clean] — per-network-class knob sets (doc 01 knobs table)."""

    distribution: GISCleanParams = field(default_factory=_distribution_clean)
    transmission: GISCleanParams = field(default_factory=_transmission_clean)


@dataclass
class GISIslandSweepConfig:
    """[gis.island_sweep] — Island_Destroyer parameter sweep (doc 01)."""

    extend_step_m: float = 3.0
    snap_step_m: float = 1.5
    extend_max_m: float = 50.0
    max_stitch_m: float = 5000.0          # transmission: refuse longer phantom edges


@dataclass
class GISBatchConfig:
    """[gis.batch] — Method1 parallel driver (doc 01)."""

    max_workers: int = 8                  # ProcessPoolExecutor; arcpy is process-safe
    skip_items: list[int] = field(
        default_factory=lambda: [514, 812, 1007, 1953, 2331])  # known-bad feeders


@dataclass
class GISOptions:
    """[gis] — S1 options (doc 01 §Proposed config extractions)."""

    backend: str = "arcpy"                # 'arcpy' | 'open' (P8 stretch, D4)
    utm_zone: int = 10                    # legacy utm.to_latlon fallback only (PD-13)
    utm_band: str = "S"
    earth_radius_m: float = 6_371_000.0   # unify 6373 km / 6371 km variants (PD-9)
    node_round_decimals: int = 6          # Method1 rounded_coord bus dedup (PD-7)
    status_exclude: list[str] = field(default_factory=lambda: ["Unknown", " "])
                                          # PD-14: "Removed planned lines" actually
                                          # drops Unknown/blank, KEEPS Proposed/Closed
    length_units: str = "us_survey_miles" # CalculateGeometryAttributes "Miles (United States)"
    clean: GISCleanConfig = field(default_factory=GISCleanConfig)
    island_sweep: GISIslandSweepConfig = field(default_factory=GISIslandSweepConfig)
    batch: GISBatchConfig = field(default_factory=GISBatchConfig)


# ---------------------------------------------------------------------------
# [build] + friends — S2 (doc 02 config table, verbatim keys)
# ---------------------------------------------------------------------------


@dataclass
class BuildOptions:
    """[build] — S2 build stage (doc 02 §Proposed config extractions)."""

    remove_tiny_lines: bool = True        # zero-length (suspected in-sub transformer) rows
    export_matpower: bool = True          # export_file("cal_grid.m")
    export_largest: bool = True           # export_file("cal_largest_grid.m")
    export_json: bool = False
    kv_levels: list[int] = field(default_factory=lambda: [66, 115, 230, 500])
                                          # kV remap targets; ties → FIRST/lowest
                                          # (Julia findmin ≡ np.argmin; doc 02)
    jitter_deg: float = 1e-4              # import-gen & split-bus lat/lon wiggle
                                          # amplitude, degrees ((rand()-0.5)*1e-4)
    transformer_tap: float = 1.0          # basic_transformer defaults,
    transformer_shift: float = 0.0        # create_transformers! (doc 02)


@dataclass
class SnappingOptions:
    """[snapping] — nearest-bus snapping (doc 02 §node_assign)."""

    warn_km: float = 15.0                 # `distance_error` warn threshold, km


@dataclass
class LoadAssignOptions:
    """[loads] — MILP load→bus assignment (doc 02 §assign_loads!)."""

    milp_warn_km: float = 50.0            # post-solve distance warning (NOT 15!)
    solver: str = "gurobi"
    renormalize_post_cut: bool = False    # DV-P2c-5 (D-6): when True, scale the
                                          # largest-component loads so Σ pd/qd
                                          # equal the pre-cut apportionment totals
                                          # (the published chain's conservation
                                          # convention at the loadsmith anchor).
                                          # Default False = faithful S2 (survivors
                                          # keep their apportioned values)


@dataclass
class EIAOptions:
    """[eia] — EIA-860 generator filters (doc 02 §generators)."""

    state: str = "CA"
    exclude_status: list[str] = field(default_factory=lambda: ["SB"])  # standby


@dataclass
class CostOptions:
    """[costs] — cost-curve rules (doc 02 §cost-curve assignment).
    Units: c2 $/MWh², c1 $/MWh, c0 $ (MATPOWER descending order)."""

    renewables: list[str] = field(default_factory=lambda: [
        "Onshore Wind Turbine", "Offshore Wind Turbine",
        "Solar Thermal without Energy Storage", "Solar Photovoltaic", "Geothermal",
    ])                                    # keep verbatim incl. Offshore (doc 02 quirk)
    renewable_cost: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    nuclear_linear_usd_per_mwh: float = 113.6 / 16.165   # = 7.027529 (EIA CA 2019)
    import_linear_usd_per_mwh: float = 209.5 / 6.673     # = 31.395175 (9.20 $/MMBtu)
    import_pmax_mw: float = 200.0         # synthetic import gen size
    fallback_donor_map: dict[str, str] = field(default_factory=lambda: {
        "Other Natural Gas": "Natural Gas Steam Turbine",
        "Municipal Solid Waste": "Landfill Gas",
    })
    legacy_nearest_capacity: bool = True  # True reproduces the comparator drift bug
                                          # (doc 02 step 5); False = true argmin
                                          # — needs a DV row before flipping


# ---------------------------------------------------------------------------
# [params] — S3 conductor/impedance/transformer parameterization (doc 03)
# ---------------------------------------------------------------------------


@dataclass
class ConductorOptions:
    """[params] — S3 parameterization (doc 03 config table)."""

    routing_factor: float = 1.28          # PSERCMain:130 — mean route/haversine ratio
    min_length_mi: float = 0.1            # below this, Length := routing_factor × haversine
    stability_angle_deg: float = 45.0     # RateA cap = SBase·sin(angle)/X (RXB/PSERCMain)
    owner_matched_utilities: list[str] = field(
        default_factory=lambda: ["SDG&E", "SCE"])  # Form 1 owner-matched filter
    random_gmd: bool = False              # options.RandomGMD (randi(3) TowerIndex, DV3)
    tower_spacing_choices: list[int] = field(default_factory=lambda: [1, 2, 3])
    # transformer sizing/impedance constants (SizeTransformers/TransformerImpedances)
    xf_initial_mva: float = 2000.0        # placeholder rating before DC duty sweep
    xf_unit_max_mva: float = 2000.0       # above → split into parallel units
    xf_min_mva: float = 20.0              # MinCondIndex clamp floor
    xf_small_mva: float = 50.0            # below → fixed X/R
    xf_small_xr: float = 25.0
    xf_max_index_factor: float = 1.2      # MaxCondIndex = 1.2·max(XFXR.MVA)


# ---------------------------------------------------------------------------
# [dispatch] — S3 scenarios (doc 03 §CreateSNet/UCED/UCCD)
# ---------------------------------------------------------------------------


@dataclass
class DispatchOptions:
    """[dispatch] — scenario generation (doc 03 config table). Field names
    mirror gridcrucible.config.DispatchConfig for D5 interop."""

    mode: str = "Full"                    # options.Dispatch
    load_data: str = "sine"               # GridWeld path (vs gridcrucible 'EPIGRIDS')
    buses: str = "given"                  # bus mode: identity mapping
    load_scenarios: int = 24
    uc_scenarios: int = 2                 # ED + complementary CD (final SetOptions value)
    use_ed: bool = True
    use_cd: bool = True
    reserve: float = 0.2                  # options.Reserve (spinning-reserve fraction)
    sine_amplitude: float = 0.3           # SineLoad = 0.3·sin(θ) + 0.7 (doc 03 step 2)
    sine_offset: float = 0.7
    gen_n_minus_1: bool = False           # options.GenN_1
    gen_contingency_min_mw: float = 2500.0  # options.GenConSize
    hydro_ratio_threshold: float = 1.2    # scale hydro thirds when GenLoadRatio > this
    hydro_thirds: list[float] = field(default_factory=lambda: [2 / 3, 1.0, 1 / 3])
    low_load_floor_mw: float = 0.2        # CreateSNet clamp (no-op in 'given' path,
    low_load_floor_mvar: float = 0.12     # DV-P3-4 — kept for the option to apply intent)
    min_dispatch_mw: float = 0.1          # UCED/UCCD cleanup: drop units at/below


# ---------------------------------------------------------------------------
# [powerflow] — S4 solvers (doc 04 config table)
# ---------------------------------------------------------------------------


@dataclass
class PowerFlowOptions:
    """[powerflow] — DC/AC power-flow settings (doc 04). s_base lives in
    [system] (shared, docs 03/04/05)."""

    ac_tol: float = 1e-8                  # pfsolve.m missing (DV-C): gridforge Newton
    ac_max_iter: int = 30                 # reference constants
    long_line: bool = True                # ACPF.LongLine = 1 (charging shunts in Y-bus)


# ---------------------------------------------------------------------------
# [resize] — S4 resize loops (docs 03+04 config tables)
# ---------------------------------------------------------------------------


@dataclass
class ResizeOptions:
    """[resize] — ResizeBranches + both loops (docs 03/04). Several defaults
    deliberately differ from gridcrucible's (percent_change 1.2 vs 1.3,
    downsize_threshold 0.3 vs 0.5, valve 20 vs 15) — always pass explicitly."""

    upsize: bool = True                   # options.Upsize (set 1 in the loops)
    downsize: bool = True                 # options.Downsize
    percent_resize: float = 0.05          # SetOptions.PercentResize (fraction of paths/pass)
    ac_percent_resize: float = 0.01       # MiddleLoopAC override
    percent_change: float = 1.2           # RateA multiplier: ceil(RateA·1.2) up, /1.2 down
    conductor_margin: float = 1.0         # SetOptions comment says "must be 1.X" but
                                          # value IS 1.0 → overload test = RateA < MaxDCPF
    downsize_threshold: float = 0.3       # SetOptions.DownsizeThreshold (utilization)
    random_resize: bool = True            # randperm eligible set (False = sorted path,
                                          # which is the DV-D table-desync hazard)
    downsize_keep: bool = False           # DV-G: honor `keep` mask in downsize
                                          # (legacy read it nowhere; decision pending)
    ac_dc_substitute: bool = True         # DV-K (D-7 ruling 2026-07-19): non-converged
                                          # scenarios contribute DC |MW| flows to the
                                          # AC envelope; False = faithful zero-fill,
                                          # which is degenerative (doc 04 §10)
    max_circuits: int = 8                 # Circuits clamp in step-circuit pass
    dc_term_overloads: int = 0            # PSERCMain TermCriteria seed
    ac_term_overloads: int = 5            # MiddleLoopAC MLTermCriteria
    valve_after_iters: int = 20           # both loops: TermCriteria += 1 per pass after
    dc_change_voltages: bool = False      # options.changevoltages (lowercase twin)
    ac_change_voltages: bool = True       # options.ChangeVoltages (capital twin —
                                          # MATLAB case-sensitive fields coexist, doc 04)
    max_passes: int = 200                 # hard safety cap on loop passes (legacy loops
                                          # rely on the TermCriteria valve alone)


# ---------------------------------------------------------------------------
# [reactive] — S4 reactive placement (doc 04 §9 config table)
# ---------------------------------------------------------------------------


@dataclass
class ReactiveOptions:
    """[reactive] — ReactivePowerPlacement winnowing (doc 04 §9)."""

    high_q_mvar: float = 10000.0          # ACPF.HighQ (gridcrucible hard-codes 800 —
    low_q_mvar: float = -10000.0          # DV-E: config-drive, default = MATLAB)
    per_switch: float = 0.05              # condenser demotion fraction per iteration
    per_switch_decay: float = 0.90        # shrink on non-convergence
    per_switch_floor: float = 1e-2        # terminate winnowing below this
    v_switch_band: list[float] = field(default_factory=lambda: [0.95, 1.05])
                                          # bank-limited PV→PQ switching band, pu
    v_place_band: list[float] = field(default_factory=lambda: [0.94, 1.06])
                                          # permanent condenser placement band, pu
    support_fraction: float = 0.2         # stop when supported buses < 0.2·n
    ed_scenario_count: int = 42           # magic `1:42` max-load scenario window
    gen_vset: float = 1.0                 # condenser/gen |V| setpoint, pu (supersedes
                                          # the DC-era 1.02 in this stage)
    b_charging_cap: float = 1.0           # OLBranch.B(B>1)=1 series-comp stand-in, pu


# ---------------------------------------------------------------------------
# [feasibility] — S5 DCOPF loop (doc 05 config table)
# ---------------------------------------------------------------------------


@dataclass
class FeasibilityQAOptions:
    """[feasibility.qa] — dead-line diagnostic (doc 05 step 4)."""

    deadline_kv_min: float = 100.0        # kV ≥ this
    deadline_flow_mw: float = 1e-2        # and MaxDCPF < this → reported dead


@dataclass
class FeasibilityOptions:
    """[feasibility] — DCOPF upgrade loop (doc 05). DV1: gurobipy, not
    YALMIP+Mosek. λ3=0 makes line limits vacuous by construction — the knob
    is exposed so the behavior is honest (doc 05 quirk)."""

    w_deviation: float = 0.95             # lambda_1 (L1 setpoint deviation weight)
    w_cost: float = 0.05                  # lambda_2 (quadratic cost weight, pu-space)
    w_violation: float = 0.0              # lambda_3 (line-limit slack weight — 0!)
    term_overloads: int = 2               # TermCriteria seed (v2 starts looser than v1)
    term_relax_after: int = 20            # iterations before TermCriteria grows
    max_workers: int = 0                  # 0 → os.cpu_count(), capped by worker_cap
    numeric_focus: int = 3                # Gurobi NumericFocus (DV1 dump hardening —
                                          # scenario 2 hit status 4 at defaults)
    bar_homogeneous: bool = True          # Gurobi BarHomogeneous=1 (DV1 dump hardening)
    max_passes: int = 200                 # hard safety cap on the S5 loop
    qa: FeasibilityQAOptions = field(default_factory=FeasibilityQAOptions)


# ---------------------------------------------------------------------------
# [io] — MATPOWER bridge defaults (doc 03 §ReadMatpower/Table2Matpower)
# ---------------------------------------------------------------------------


@dataclass
class IOOptions:
    """[io] — MATPOWER read/write defaults (doc 03 config table)."""

    default_line_rxb: list[float] = field(default_factory=lambda: [0.01, 0.1, 0.001])
                                          # pu placeholder until RXB (ReadMatpower step 3)
    default_xf_rxb: list[float] = field(default_factory=lambda: [1e-6, 1e-5, 0.0])
                                          # pu bus-tie placeholder until SizeTransformers
    vmax: float = 1.06                    # Table2Matpower VMAX, pu
    vmin: float = 0.94                    # Table2Matpower VMIN, pu
    gen_voltage_setpoint: float = 1.02    # PF.VBusSet at gen/slack buses, pu (DC era)
    dummy_cost: float = 0.1               # GenCostRandom / missing-gencost coefficient


# ---------------------------------------------------------------------------
# [parallel] — worker sizing (docs 04/05)
# ---------------------------------------------------------------------------


@dataclass
class ParallelOptions:
    """[parallel] — replaces MATLAB parpool heuristics (RAM-gate dropped —
    explicit flags instead; docs 04/05)."""

    ac_parallel: bool = False             # options.ACParallel
    ac_workers_per_core: float = 1.5      # ceil(1.5·numcores)…
    worker_cap: int = 32                  # …capped at 32 (both loops + DCOPF sweep)


# ---------------------------------------------------------------------------
# [export] — S7 writers (doc 07 config table)
# ---------------------------------------------------------------------------


@dataclass
class PowerWorldOptions:
    """[export.powerworld] — WritePowerWorld.m constants (doc 07)."""

    network_tag: str = "CAISO"            # options.Network → filename suffix
    region_tag: str = ""                  # options.Region → filename suffix
    limit_b_factor: float = 1.1           # LimitMVAB = 1.1·RateA
    limit_c_factor: float = 1.15          # LimitMVAC = 1.15·RateA
    commit_threshold_mw: float = 0.1      # GenMWSet commitment cutoff & setpoint zeroing
    shunt_margin: float = 1.3             # switched-shunt Nom = 1.3·|QNet|
    reg_high: float = 1.06                # pu
    reg_low: float = 0.94                 # pu
    block_steps: int = 10
    pu_sniff_factor: float = 40.0         # WriteLoads ×40 auto-pu rule (DV-07c)
    pu_threshold_mw: float = 20000.0      # Aditya-fork PeakLoad `<20000` rule (DV-07c)
    load_sign: str = "negative"           # DV-07e RULED 2026-07-19: loads are the
                                          # NEGATIVE real part of the injection-
                                          # convention SNet (gen +, load −); the
                                          # Aditya fork's >0 pick selected generation
                                          # ('positive' retained for replication)


@dataclass
class ExportGISOptions:
    """[export.gis] — reformat_GIS.jl round-trip constants (doc 07)."""

    feet_to_km: float = 0.0003048
    outdated_fields: list[str] = field(default_factory=lambda: [
        "Length_Fee", "GlobalID", "Legend", "Creator", "Structur_1",
        "kV_Sort", "Engineerin", "Source", "Length_Mil", "Status", "FID",
        "Name", "Structure_", "TLine_Name", "OBJECTID", "Engineer_1",
        "Owner", "Length_mi", "Last_Edito",
    ])                                    # the 19 names, reformat_GIS.jl:51-52
    preserve_added_node_props: bool = True    # DV-07d APPROVED 2026-07-19: keep the
                                              # original GeoJSON properties and only
                                              # add CATS_ID/Type/kV/Comments
                                              # (False = legacy replace, fixture mode)


@dataclass
class ExportQAOptions:
    """[export.qa] — QA report thresholds (doc 07)."""

    long_distance_km: float = 15.0        # snap-distance report (matches [snapping])


@dataclass
class ExportOptions:
    """[export] — S7 writers (doc 07)."""

    powerworld: PowerWorldOptions = field(default_factory=PowerWorldOptions)
    gis: ExportGISOptions = field(default_factory=ExportGISOptions)
    qa: ExportQAOptions = field(default_factory=ExportQAOptions)


# ---------------------------------------------------------------------------
# [eval] — S6 Julia evaluation (doc 06 config table; read by
# julia/GridWeldEval bin/eval.jl, NOT by the Python pipeline)
# ---------------------------------------------------------------------------


@dataclass
class EvalRenewablesOptions:
    """[eval.renewables] — renewable scaling (doc 06 §process_renewable_data)."""

    solar_types: list[str] = field(default_factory=lambda: ["Solar Photovoltaic"])
    wind_types: list[str] = field(default_factory=lambda: ["Onshore Wind Turbine"])
    caiso_year_rows: int = 105408         # 366 d × 288 five-minute intervals (2020)


@dataclass
class EvalOptions:
    """[eval] — S6 evaluation config (doc 06). Paths relative to legacy/ tree
    until P6 relocates them."""

    n_hours: int = 24                     # N in eval.jl
    case_path: str = "Final_Grid/updated models/cal_grid_acpf_v4.m"
    renewable_csv: str = "Renewable Data/Renewable_data_2020_eval.csv"
    load_csv: str = "Load Data/Load_Agg_Post_Assignment.csv"   # ⚠ file LOST (DV-P6-4);
                                          # legacy complex-string matrix OR the
                                          # WriteLoads MW matrix when load_mvar_csv set
    load_mvar_csv: str = ""               # WriteLoads MVar matrix (preferred pair
                                          # contract, DV-P6-4); "" → load_csv is the
                                          # legacy single-file format
    gen_data_csv: str = "Additional Data Files/gen_data.csv"
    solver: str = "ipopt"
    save_solutions: bool = False
    save_summary: bool = False
    solutions_dir: str = "Grid_Evaluation/solutions"
    first_gen_col: int = 5                # DV-P6-3: the legacy 6 was a CONFIRMED
                                          # off-by-one (template gens start at col 5)
    renewable_target: str = "pg"          # DV-P6-5: faithful pg warm-start; "pmax"
                                          # = suspected intent (awaiting the author's ruling)
    renewables: EvalRenewablesOptions = field(default_factory=EvalRenewablesOptions)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass
class WeldConfig:
    """Top-level config: one TOML file == one reproducible run (D8/D10)."""

    # gis_clean needs ArcGIS Pro's Python and runs separately (not by default).
    stages: list[str] = field(
        default_factory=lambda: [s for s in STAGES if s != "gis_clean"])
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    gis: GISOptions = field(default_factory=GISOptions)
    build: BuildOptions = field(default_factory=BuildOptions)
    snapping: SnappingOptions = field(default_factory=SnappingOptions)
    loads: LoadAssignOptions = field(default_factory=LoadAssignOptions)
    eia: EIAOptions = field(default_factory=EIAOptions)
    costs: CostOptions = field(default_factory=CostOptions)
    params: ConductorOptions = field(default_factory=ConductorOptions)
    dispatch: DispatchOptions = field(default_factory=DispatchOptions)
    powerflow: PowerFlowOptions = field(default_factory=PowerFlowOptions)
    resize: ResizeOptions = field(default_factory=ResizeOptions)
    reactive: ReactiveOptions = field(default_factory=ReactiveOptions)
    feasibility: FeasibilityOptions = field(default_factory=FeasibilityOptions)
    io: IOOptions = field(default_factory=IOOptions)
    parallel: ParallelOptions = field(default_factory=ParallelOptions)
    export: ExportOptions = field(default_factory=ExportOptions)
    eval: EvalOptions = field(default_factory=EvalOptions)
    verbose: bool = False

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> WeldConfig:
        """Load a TOML run file over the defaults. Unknown sections/keys are
        ERRORS (typo protection, same policy as gridcrucible); Path-typed
        fields are coerced; nested tables recurse into nested dataclasses."""
        import tomllib

        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeldConfig:
        """Build a config from a nested mapping (the parsed-TOML shape)."""
        cfg = cls()
        for key, val in data.items():
            if key == "stages":
                cfg.stages = list(val)
                continue
            if key == "verbose":
                cfg.verbose = bool(val)
                continue
            if not hasattr(cfg, key) or not isinstance(val, dict):
                raise ValueError(f"unknown config section [{key}]")
            _apply(getattr(cfg, key), val, section=key)
        problems = cfg.validate()
        if problems:
            raise ValueError("invalid config: " + "; ".join(problems))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Nested plain-type mapping (Paths → str) — the from_dict inverse."""
        return _plain(dataclasses.asdict(self))

    def validate(self) -> list[str]:
        """Cross-field checks; returns a list of problems (empty = ok)."""
        v: list[str] = []
        unknown = [s for s in self.stages if s not in STAGES]
        if unknown:
            v.append(f"stages: unknown {unknown}")
        order = [STAGES.index(s) for s in self.stages if s in STAGES]
        if order != sorted(order):
            v.append(f"stages: must follow canonical order {list(STAGES)}")
        if self.system.voltage_levels != sorted(self.system.voltage_levels):
            v.append("system.voltage_levels must be ascending (kV-block bookkeeping)")
        if abs(self.geo.earth_radius_km * 1000.0 - self.gis.earth_radius_m) > 1e-6:
            v.append("geo.earth_radius_km and gis.earth_radius_m disagree (PD-9)")
        if not 0 < self.resize.percent_resize <= 1:
            v.append("resize.percent_resize must be in (0, 1]")
        if not 0 < self.dispatch.reserve < 1:
            v.append("dispatch.reserve must be in (0, 1)")
        weights = (self.feasibility.w_deviation, self.feasibility.w_cost,
                   self.feasibility.w_violation)
        if any(w < 0 for w in weights):
            v.append("feasibility weights must be non-negative")
        if self.gis.backend not in ("arcpy", "open"):
            v.append("gis.backend must be 'arcpy' or 'open'")
        return v


def _apply(target: Any, values: dict[str, Any], section: str) -> None:
    """Recursively apply a mapping onto a dataclass; strict on unknown keys."""
    for key, val in values.items():
        if not hasattr(target, key):
            raise ValueError(f"unknown key {key!r} in [{section}]")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(val, dict):
            _apply(current, val, section=f"{section}.{key}")
        elif isinstance(current, Path):
            setattr(target, key, Path(val))
        else:
            setattr(target, key, val)


def _plain(obj: Any) -> Any:
    """asdict post-pass: Paths → str so the mapping is TOML/JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fieldnames(dc: Any) -> list[str]:
    """Helper for tests/docs: declared field names of a config dataclass."""
    return [f.name for f in fields(dc)]
