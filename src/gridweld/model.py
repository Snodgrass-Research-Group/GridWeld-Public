"""WeldModel: the single shared data structure every GridWeld stage consumes
and returns (decision D6).

Replaces the legacy state that never coexisted in one place: the PowerModels
``grid`` Dict of ``generate_grid_script.jl`` (S2), the MATLAB workspace tables
``Bus`` / ``Branch`` / ``Gen`` (+ ``PF`` struct) of ``PSERCMain.m`` (S3/S4),
and the DCOPF loop state of ``PSERCMain_v2.m`` (S5).

Conventions (D6; same contract as ``gridforge.model.GridModel`` /
``gridcrucible.model.CrucibleModel``):
* bus / branch / generator / load indices are 0-based internal ids; external
  MATPOWER ids appear ONLY at I/O boundaries (``gridweld/io``; the
  ``CreateBranch.m`` int→ext shim becomes writer columns — doc 04 §11)
* column schemas are module constants so io/checkpoints and tests validate
* separate MW / MVAr float columns, never complex, in DataFrames (scenario
  matrices stay complex ndarrays, matching ``PF.SNetFull``)
* branch columns reused by gridcrucible functions keep gridcrucible's names
  (``capacity_mva``, ``conductor_idx``, ``n_circuits``, ``voltage_kv_from``,
  ``resize`` — doc 03 §AssignConductors, D5)
* lengths: ``length_mi`` in statute miles (S3 works in miles; convert km→mi
  once at ingest with ``system.km_per_mile`` = 1.60934, doc 03 §DistanceCalc)

Schemas are the P1 contract; later phases may APPEND columns (never rename)
and must update the constants here in the same commit.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:  # runtime import deferred; gridcrucible is a path dep (D5)
    from gridcrucible.model import ScenarioSet

# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------

#: Raw substation/added-node point layers (S1 output, S2 input — doc 02 §I/O).
SUBSTATION_COLUMNS = [
    "sub_id",        # 0-based internal index (row position == sub_id)
    "fid",           # source-layer FID (provenance; Julia index = FID+1)
    "lat", "lon",    # WGS84 degrees
    "node_type",     # 'Substation' | 'AddedNode' (Julia `Type`)
    "import_flag",   # bool: `Import == "IMPORT"` (out-of-state interchange bus)
]

BUS_COLUMNS = [
    "bus_id",        # 0-based internal index (row position == bus_id)
    "external_num",  # MATPOWER bus number (I/O boundary only; NaN until export)
    "fid",           # provenance FID (split buses inherit it — doc 02)
    "lat", "lon",    # WGS84 degrees (jittered copies at split buses, DV3)
    "node_type",     # 'Substation' | 'AddedNode' (survives bus splitting)
    "import_flag",   # bool
    "base_kv",       # ∈ voltage_levels, or 0 for isolated buses (doc 02)
    "bus_type",      # 1 PQ / 2 PV / 3 ref / 4 isolated (MATPOWER; all 1 at S2 —
                     # slack is assigned by merge_topology, doc 03)
    "pd_mw", "qd_mvar",        # folded loads (MATPOWER has no load table)
    "vm_pu", "va_deg",         # power-flow state
    "max_mw", "min_mw",        # per-bus fleet aggregates (Bus.MaxMW/MinMW;
                               # MinMW exists only with DV-P3-1 fixed)
    "max_mvar", "min_mvar",    # (DV-P3-3: legacy copied MW values here)
    "gen_mw_set",    # Bus.GenMWSet — commitment bookkeeping (doc 05 §5)
    "pnet_mw", "qnet_mvar",    # last-solve net injections (SNetCalc)
    "sync_cond",     # bool: permanent synchronous condenser (doc 04 §9)
]

BRANCH_COLUMNS = [
    "from_bus", "to_bus",      # 0-based internal endpoint ids
    "fid",                     # GIS FID (NaN for created transformer branches)
    "lat1", "lon1", "lat2", "lon2",   # endpoint coordinates, WGS84 degrees
    "voltage_kv_from", "voltage_kv_to",   # gridcrucible names (D5)
    "length_mi",               # real GIS route length, statute miles
    "path_length_mi",          # haversine point-to-point, miles (PSERCMain)
    "kcmil",                   # conductor Size (gridforge name)
    "n_cond_per_phase",        # CondPerPhase
    "n_circuits",              # parallel circuits (gridcrucible name; clamp 1..8)
    "conductor_idx",           # row in pooled ConductorCatalog; ⚠ for
                               # transformers stores per-unit MVA (field
                               # overload kept from TransformerImpedances.m)
    "min_conductor_idx", "max_conductor_idx",   # kV-block bounds (SizeLines)
    "gmd_ft",                  # phase spacing GMD, feet (TowerData; UG = 1)
    "tower_index",             # TowerData Spacing 1|2|3 (randi(3) if absent, DV3)
    "material", "structure",   # Form 1-derived; 'UG'/'Transformer' sentinels
    "r_pu", "x_pu", "b_pu",    # per-unit on s_base (gridforge names)
    "capacity_mva",            # RateA, total across parallel circuits
    "tap", "shift",            # tap==0 ⇔ line (TL mask), tap≠0 ⇔ transformer (XF)
    "is_transformer",          # bool (doc 02 `transformer` flag)
    "status",                  # br_status (1 = in service)
    "resize",                  # bool: flagged for AssignConductors re-snap
    "overloaded", "undersized",  # bools: at conductor bound, circuits must step
    "keep",                    # bool: protected from downsize (DV-G)
    "max_dcpf_mw",             # envelope |flow| across scenarios (Branch.MaxDCPF)
    "max_angle_diff_deg",      # envelope |Δθ| across scenarios
]

GENERATOR_COLUMNS = [
    "gen_id",        # 0-based internal index (row position == gen_id)
    "bus",           # 0-based bus id (gen_bus; snapped to SUBSTATIONS only — doc 02)
    "plant_code",    # EIA-860 Plant Code (0 for synthetic imports)
    "eia_gen_id",    # EIA Generator ID string ('0' for imports)
    "fuel_type",     # EIA Technology ('IMPORT' for imports)
    "pmax_mw", "pmin_mw",
    "qmax_mvar", "qmin_mvar",
    "lat", "lon",    # plant coords (imports: bus coords + jitter, DV3)
    "gen_bus_distance_km",     # snap distance (NaN for imports)
    "status",        # gen_status (1 = in service)
    "vg_pu",         # voltage setpoint
    "cost_model",    # 2 = polynomial (MATPOWER)
    "ncost",         # 3
    "startup_usd", "shutdown_usd",
    "c2", "c1", "c0",          # $/MWh², $/MWh, $ — MATPOWER descending order;
                               # source CSV is ascending A,B,C (doc 02 reversal)
]

LOAD_COLUMNS = [
    "load_id",       # 0-based internal index (row position == load_id)
    "bus",           # 0-based bus id (MILP assignment — doc 02 §assign_loads!)
    "pd_mw", "qd_mvar",        # ⚠ Q5/DV6: pd inherits the "P load is INCORRECT"
                               # defect pending the author's ruling (P2c gate)
    "lat", "lon",    # WGS84 degrees
    "load_bus_distance_km",    # MILP assignment distance (warn > 50 km)
    "status",        # 1 = in service
]


# ---------------------------------------------------------------------------
# Frozen result bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerFlowResult:
    """One DC or AC power-flow solve (doc 04 §3-§6).

    Sign convention: flows use the +1/X primitive (DV-B — DCPowerFlowFull's
    negated BPrim is NOT reproduced); angles are EXACT degrees (Q9 — no
    arctan compression)."""

    converged: bool
    v_mag_pu: np.ndarray | None = None       # (n_bus,)
    v_angle_deg: np.ndarray | None = None    # (n_bus,) exact degrees
    flow_mva: np.ndarray | None = None       # (n_branch,) DC: MW; AC: max(|S_send|,|S_rec|)
    angle_diff_deg: np.ndarray | None = None  # (n_branch,) |Δθ| across each branch
    snet_calc_mva: np.ndarray | None = None  # (n_bus,) complex net injections (AC)
    losses_mw: float = float("nan")
    iterations: int = 0


@dataclass(frozen=True)
class DispatchResult:
    """One UCED/UCCD solve (doc 03 §UCED/§UCCD).

    ⚠ Objective is the cplexqp-convention QP value (½·C quadratic term,
    DV-P3-5), not the physical cost — report both when implemented."""

    x_mw: np.ndarray                 # (n_committed,) unit dispatch, MW
    fleet_index: np.ndarray          # surviving (committed) generator gen_ids
    objective: float                 # QP objective (½C convention)
    iterations: int = 0


@dataclass(frozen=True)
class MilpResult:
    """MILP load→bus assignment outcome (doc 02 §assign_loads!). Validation
    compares OBJECTIVE, never per-load argmin (degenerate optima)."""

    objective_km: float              # Σ assigned haversine distances
    assignment: np.ndarray           # (n_load,) 0-based bus id per load
    distance_km: np.ndarray          # (n_load,) assigned distances
    gap: float = 0.0                 # relative MIP gap at termination
    runtime_s: float = float("nan")


@dataclass(frozen=True)
class DCOPFResult:
    """Single-scenario DCOPF solve (doc 05 §solve_dcopf, verbatim contract)."""

    theta_rad: np.ndarray            # (n_bus,) bus angles
    pg_mw: np.ndarray                # (n_gen,) GenDispatch
    delp_pu: np.ndarray              # (n_gen,) setpoint deviation
    delta_pu: np.ndarray             # (n_branch,) line-limit slack (λ3=0 → free)
    flow_mw: np.ndarray              # (n_branch,)
    angle_diff_rad: np.ndarray       # (n_branch,)
    snet_mva: np.ndarray             # (n_bus,) complex, MATLAB DC convention (Q9)
    objective: float
    status: int                      # gurobipy status code


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WeldError(Exception):
    """Base class for gridweld pipeline errors."""


class SingularNetworkError(WeldError):
    """Disconnected/singular network where the legacy code silently bailed or
    swallowed the failure (DV-H/DV-I: replaces `condest > 1e16` / `sprank`
    proxies and MATLAB try/catch skips with an exact connectivity check)."""


class DCOPFInfeasibleError(WeldError):
    """DCOPF scenario did not solve to optimality (doc 05: replaces the
    legacy `disp(sol.info); pause` interactive hang). Carries scenario index
    and solver status in the message."""


class SchemaError(WeldError):
    """A DataFrame violates its module-constant column schema."""


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class WeldModel:
    """The pipeline state. Every stage takes and returns one of these (D6).

    ``scenarios`` reuses ``gridcrucible.model.ScenarioSet`` (doc 03
    §CreateSNet: tiers dict full/complete/advanced/intermediate/basic +
    slack/gen/load bus bookkeeping — MATLAB ``PF``)."""

    substations: pd.DataFrame = field(default_factory=pd.DataFrame)
    buses: pd.DataFrame = field(default_factory=pd.DataFrame)
    branches: pd.DataFrame = field(default_factory=pd.DataFrame)
    generators: pd.DataFrame = field(default_factory=pd.DataFrame)
    loads: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenarios: "ScenarioSet | None" = None
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys (grow the list, document here): name, base_mva, per_unit,
    # region, rng_seed, stage_history, slack_bus, pg_matrix (g×K MW, doc 05),
    # snet_opf (n×K complex, doc 05), self_loop_fids, tiny_line_fids

    # -- derived ------------------------------------------------------------

    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    def incidence(self) -> sp.csc_matrix:
        """Signed incidence A (n_buses × n_branches): +1 at from, −1 at to.

        Origin: ``BuildANet.m`` (doc 04 §1) with EXPLICIT n_buses — the
        max-id inference (quirk Q-a) is not reproduced."""
        m = self.n_branches
        rows = np.concatenate([self.branches["from_bus"].to_numpy(np.int64),
                               self.branches["to_bus"].to_numpy(np.int64)])
        cols = np.tile(np.arange(m, dtype=np.int64), 2)
        vals = np.concatenate([np.ones(m), -np.ones(m)])
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.n_buses, m))

    def line_mask(self) -> np.ndarray:
        """TL mask: tap == 0 (PSERCMain convention, doc 03 step 13)."""
        return self.branches["tap"].to_numpy() == 0

    def transformer_mask(self) -> np.ndarray:
        """XF mask: tap != 0 (complement of :meth:`line_mask`)."""
        return self.branches["tap"].to_numpy() != 0

    #: Minimum columns a non-empty table must carry at ANY pipeline point;
    #: the full module-constant schemas apply to stage-complete models
    #: (``validate(full=True)``) — I/O-boundary models (e.g. a parsed S2
    #: case, P2a) legitimately lack later-stage columns.
    CORE_COLUMNS = {
        "substations": ["sub_id", "lat", "lon"],
        "buses": ["bus_id"],
        "branches": ["from_bus", "to_bus", "tap"],
        "generators": ["gen_id", "bus"],
        "loads": ["load_id", "bus"],
    }

    def validate(self, *, full: bool = False) -> list[str]:
        """Schema + invariant check; returns a list of violations (empty=ok).

        ``full=True`` additionally requires every column of the module
        schemas (end-of-stage contract); the default checks the core
        columns and the id-range invariants only.
        """
        v: list[str] = []
        for tbl, required in (("substations", SUBSTATION_COLUMNS),
                              ("buses", BUS_COLUMNS),
                              ("branches", BRANCH_COLUMNS),
                              ("generators", GENERATOR_COLUMNS),
                              ("loads", LOAD_COLUMNS)):
            df = getattr(self, tbl)
            if len(df):
                need = required if full else self.CORE_COLUMNS[tbl]
                missing = [c for c in need if c not in df.columns]
                if missing:
                    v.append(f"{tbl}: missing columns {missing}")
        br = self.branches
        if len(br) and {"from_bus", "to_bus"} <= set(br.columns):
            fb = br["from_bus"].to_numpy(np.int64)
            tb = br["to_bus"].to_numpy(np.int64)
            n = self.n_buses
            if n and (fb.min(initial=0) < 0 or tb.min(initial=0) < 0
                      or fb.max(initial=-1) >= n or tb.max(initial=-1) >= n):
                v.append(f"branches: endpoint out of range [0, {n})")
        for tbl, col in (("generators", "bus"), ("loads", "bus")):
            df = getattr(self, tbl)
            if len(df) and col in df.columns and self.n_buses:
                ids = df[col].to_numpy(np.int64)
                if ids.min(initial=0) < 0 or ids.max(initial=-1) >= self.n_buses:
                    v.append(f"{tbl}.{col}: id out of range [0, {self.n_buses})")
        if self.scenarios is not None:
            for name, arr in self.scenarios.tiers.items():
                if arr.shape[0] != self.n_buses:
                    v.append(f"scenarios[{name}]: {arr.shape[0]} rows != "
                             f"{self.n_buses} buses")
        return v

    def copy(self) -> "WeldModel":
        return _copy.deepcopy(self)
