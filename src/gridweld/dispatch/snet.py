"""CreateSNet: net-injection scenario matrix (bus × scenario, complex MW+jMVar).

Origin: ``CreateSNet.m`` (re-verified 2026-07-17), GridWeld path:
Buses='given', LoadData='sine', Dispatch='Full', one region, empty
ImportBuses. In that path (source line refs):

1. Per-bus fleet aggregates when missing [16-19]; ⚠ DV-P3-3: MATLAB assigns
   Bus.MaxMVar/MinMVar from MaxMW/MinMW (copy-paste) — ``matlab_compat``
   keeps it, False aggregates the MVar columns (CATS gen MVar are all 0, so
   both give zeros).
2. Sine loads [184-196]: ``θ = linspace(0, 2π, LoadScenarios)``;
   ``SineLoad = 0.3·sin(θ) + 0.7``; column i = SineLoad_i ·
   complex(Bus.MW, Bus.MVar) over ALL buses.
3. Import-bus schedule and the low-load 0.2+0.12j clamp are NO-OPS in
   'given' (``LoadScenarioBuses = []`` [263]) — DV-P3-4 replicated;
   ``clamp_low_loads=True`` applies the evident intent to load buses.
4. Guards [265-281]: maxscenario = argmax of column load sums;
   GenLoadRatio = ΣMaxMW/peak; hydro capacity via CONTAINS('Hydro') —
   note the dispatchers themselves match 'Hydro' EXACTLY (no-op for CATS);
   over-/under-scaling of the whole load matrix.
5. ED block then CD block [320-365]: per scenario column, hydro thirds
   (⅔/1/⅓) when GenLoadRatio > threshold; SNetFull col = zero-padded
   ``accumarray(BusNum, dispatch) − LoadScenarios col`` (complex load
   subtracted → imag = −Q load).
6. Tier subsampling per UC block (Complete/Advanced/Intermediate/Basic) —
   REUSE (D5): gridcrucible ``data.scenarios._commitment_subsample`` /
   ``_diff_num_ed`` / ``_diff_num_cd`` (ported from THIS file); signature
   rows = [slack; GenList] with slack = argmax Bus.MaxMW (the DV-P3-1
   values — PSERCMain step 11) and GenList = sorted unique gen buses minus
   slack; VBusSet = ``io.gen_voltage_setpoint`` at gen+slack buses.
7. The unconditional LoadImportBase.csv side effect is NOT here — rerouted
   to ``gridweld.io.powerworld`` opt-in (DV-P3-10).

Units: SNet complex MW/MVar (per-unit division happens in the solvers);
bus ids 1-based internal (the S3 frame convention).
Phase: P3. Validated vs s5_pf.mat / s5_snetfull fixtures.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from gridcrucible.data.scenarios import (_commitment_subsample, _diff_num_cd,
                                         _diff_num_ed)
from gridcrucible.model import ScenarioSet

from ..config import WeldConfig
from .uced import uccd, uced


def bus_fleet_aggregates(bus_df: pd.DataFrame, gen_df: pd.DataFrame, *,
                         matlab_compat: bool = True) -> pd.DataFrame:
    """CreateSNet.m:16-19 per-bus fleet aggregates (only when absent —
    ReadMatpower usually filled MaxMW already). DV-P3-3: compat assigns the
    MVar aggregates from the MW ones."""
    out = bus_df.copy()
    n = len(out)
    bus_1b = gen_df["BusNum"].to_numpy(int)
    agg = {
        "MaxMW": np.bincount(bus_1b - 1, weights=gen_df["MaxMW"], minlength=n),
        "MinMW": np.bincount(bus_1b - 1, weights=gen_df["MinMW"], minlength=n),
    }
    if matlab_compat:
        agg["MaxMVar"] = agg["MaxMW"]
        agg["MinMVar"] = agg["MinMW"]
    else:
        agg["MaxMVar"] = np.bincount(bus_1b - 1, weights=gen_df["MaxMVar"],
                                     minlength=n)
        agg["MinMVar"] = np.bincount(bus_1b - 1, weights=gen_df["MinMVar"],
                                     minlength=n)
    for col, vals in agg.items():
        if col not in out.columns:
            out[col] = vals
    return out


def sine_load_scenarios(bus_df: pd.DataFrame, cfg: WeldConfig) -> np.ndarray:
    """Step 2: (n_bus × LoadScenarios) complex loads."""
    theta = np.linspace(0.0, 2 * np.pi, cfg.dispatch.load_scenarios)
    sine = (cfg.dispatch.sine_amplitude * np.sin(theta)
            + cfg.dispatch.sine_offset)
    base = (bus_df["MW"].to_numpy(float)
            + 1j * bus_df["MVar"].to_numpy(float))
    return base[:, None] * sine[None, :]


def bus_roles(bus_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: WeldConfig, *,
              matlab_compat: bool = False
              ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """PSERCMain.m:318-325 bus-role bookkeeping.

    DV-P3-12 (the author's ruling 2026-07-19: **corrected is the DEFAULT**):
    the legacy ``GenList(2:end)`` drops the FIRST (smallest) gen bus on the
    wrong assumption it is the slack ("1st Bus.Num is the slack" comment),
    while the TRUE slack (argmax Bus.MaxMW — the DV-P3-1 values) stays
    INSIDE GenList; the dropped gen bus lands in LoadList (a gen bus typed
    PQ downstream). Default (``matlab_compat=False``): GenList = all gen
    buses minus the TRUE slack; nothing is mis-dropped.
    ``matlab_compat=True`` replicates the quirk exactly (s5_pf.mat fixture
    parity). VBusSet = gen_voltage_setpoint at [slack; GenList] either way.
    """
    max_mw = bus_df["MaxMW"].to_numpy(float)
    slack = int(np.argmax(max_mw)) + 1        # first max (MATLAB max), 1-based
    gen_buses = np.sort(gen_df["BusNum"].astype(int).unique())
    if matlab_compat:
        gen_list = gen_buses[1:]              # GenList(2:end) — see docstring
    else:
        gen_list = np.setdiff1d(gen_buses, [slack])
    all_buses = np.arange(1, len(bus_df) + 1)
    load_list = np.setdiff1d(all_buses, np.concatenate([gen_list, [slack]]))
    vset = np.ones(len(bus_df))
    vset[np.concatenate([[slack], gen_list]) - 1] = cfg.io.gen_voltage_setpoint
    return slack, gen_list, load_list, vset


def _dispatch_block(loads: np.ndarray, fleet: pd.DataFrame, dispatch_fn,
                    cfg: WeldConfig, gen_load_ratio: float,
                    gurobi_env=None,
                    pg_out: list[np.ndarray] | None = None) -> np.ndarray:
    """Step 5 (one UC block). Hydro thirds when GenLoadRatio > threshold —
    the schedule is passed but is a NO-OP inside the CATS dispatchers
    (exact-'Hydro' match; kept for fidelity). When ``pg_out`` is given, the
    UNIT-level dispatch matrix (g × n_scen MW, zeros for uncommitted units
    — MATLAB ``PF.Pg``'s block) is appended to it (the S5 DCOPF needs PF.Pg
    surfaced)."""
    n_buses, n_scen = loads.shape
    thirds = cfg.dispatch.hydro_thirds
    out = np.zeros((n_buses, n_scen), complex)
    pg_block = np.zeros((len(fleet), n_scen))
    for i in range(n_scen):
        if gen_load_ratio > cfg.dispatch.hydro_ratio_threshold:
            hydro = thirds[0] if i < np.floor(n_scen / 3) else \
                thirds[1] if i < np.floor(2 * n_scen / 3) else thirds[2]
        else:
            hydro = 1.0
        x, fleet_out = dispatch_fn(fleet, hydro, loads[:, i].real,
                                   cfg.dispatch.reserve, gurobi_env)
        pg_block[fleet.index.get_indexer(fleet_out.index), i] = x
        gen_bus = np.bincount(fleet_out["BusNum"].to_numpy(int) - 1,
                              weights=x, minlength=n_buses)
        out[:, i] = gen_bus - loads[:, i]
    if pg_out is not None:
        pg_out.append(pg_block)
    return out


def build_snet(bus_df: pd.DataFrame, gen_df: pd.DataFrame, cfg: WeldConfig, *,
               gurobi_env=None, matlab_compat: bool = True,
               bus_roles_compat: bool = False,
               clamp_low_loads: bool = False,
               uced_fn: Callable = uced, uccd_fn: Callable = uccd,
               extras: dict | None = None) -> ScenarioSet:
    """CreateSNet.m for the GridWeld path. ``bus_df``/``gen_df`` are the
    post-ReadMatpower S3 frames (1-based BusNum; Bus MW/MVar folded loads).

    Returns a gridcrucible ``ScenarioSet`` (D5): tiers full/complete/
    advanced/intermediate/basic, slack/gen/load bus lists (1-based, matching
    MATLAB PF), v_setpoints. When ``extras`` (a dict) is passed it is filled
    with the PF fields the ScenarioSet does not carry (the S5 DCOPF
    contract): ``load_scenarios`` (n × LoadScenarios complex
    MW, post-guard-scaling ≡ PF.LoadScenarios) and ``pg`` (g × totalscenarios
    MW unit-level dispatch ≡ PF.Pg).
    """
    bus = bus_fleet_aggregates(bus_df, gen_df, matlab_compat=matlab_compat)
    loads = sine_load_scenarios(bus, cfg)
    n_buses = len(bus)

    max_scen = int(loads.real.sum(axis=0).argmax())
    if clamp_low_loads:                                   # DV-P3-4 opt-in
        load_rows = np.flatnonzero(
            ~np.isin(np.arange(1, n_buses + 1),
                     gen_df["BusNum"].astype(int).unique()))
        low = load_rows[loads.real[load_rows, max_scen] < 0.2]
        loads[low, :] = 0.2 + 0.12j

    cap = float(gen_df["MaxMW"].sum())
    cap_min = float(gen_df["MinMW"].sum())
    hydro_cap = float(gen_df.loc[gen_df["FuelType"].astype(str)
                                 .str.contains("Hydro", case=False),
                                 "MaxMW"].sum())
    peak = float(loads.real[:, max_scen].sum())
    if cap - (2 / 3) * hydro_cap <= 1.01 * peak:
        loads = loads * (cap / (1.01 * peak))
    elif cap_min >= 0.99 * peak:
        loads = loads * (1.01 * cap_min / peak)
    gen_load_ratio = cap / float(loads.real[:, max_scen].sum())

    blocks: list[np.ndarray] = []
    pg_blocks: list[np.ndarray] = []
    if cfg.dispatch.use_ed:
        blocks.append(_dispatch_block(loads, gen_df, uced_fn, cfg,
                                      gen_load_ratio, gurobi_env,
                                      pg_out=pg_blocks))
    if cfg.dispatch.use_cd:
        blocks.append(_dispatch_block(loads, gen_df, uccd_fn, cfg,
                                      gen_load_ratio, gurobi_env,
                                      pg_out=pg_blocks))
    snet_full = np.hstack(blocks)
    if extras is not None:
        extras["load_scenarios"] = loads
        extras["pg"] = np.hstack(pg_blocks)

    # DV-P3-12: corrected roles by default; bus_roles_compat=True replicates
    # the GenList(2:end) quirk for s5 fixture parity
    slack, gen_list, load_list, vset = bus_roles(bus, gen_df, cfg,
                                                 matlab_compat=bus_roles_compat)
    gen_rows = np.concatenate([[slack], gen_list]) - 1     # 0-based rows
    ed_diff = _diff_num_ed(n_buses, False)
    tiers: dict[str, list[np.ndarray]] = {t: [] for t in
                                          ("complete", "advanced",
                                           "intermediate", "basic")}
    for bi, block in enumerate(blocks):
        if bi == 0 and cfg.dispatch.use_ed:
            d = ed_diff
        else:
            d = _diff_num_cd(n_buses, False, gen_load_ratio,
                             [cfg.system.region], ed_diff)
        tiers["complete"].append(_commitment_subsample(block, gen_rows, d))
        adv = _commitment_subsample(block, gen_rows, d + 1)
        tiers["advanced"].append(adv)
        tiers["intermediate"].append(
            adv if n_buses < 4000
            else _commitment_subsample(block, gen_rows, d + 2))
        tiers["basic"].append(
            tiers["intermediate"][-1] if n_buses < 2000
            else _commitment_subsample(block, gen_rows, d + 3))

    return ScenarioSet(
        tiers={"full": snet_full,
               **{t: np.hstack(v) for t, v in tiers.items()}},
        slack_bus=slack, gen_buses=gen_list, load_buses=load_list,
        v_setpoints=vset)
