"""PowerWorld import-CSV writers.

Origin: ``WriteLoads.m`` (identical in both forks) and the **Aditya fork**
of ``WritePowerWorld.m`` (``Aditya_Code/Matlab Functions/`` — the S5
lineage, doc 07). Fork facts replicated here (they differ from the
Jonathan_Code original that doc 07 described): SubstationImport is built FROM THE
BUS TABLE (no Sub table in the fork's signature); ``Branch.DeviceType``
is derived from ``strcmpi(Branch.Structure,'Transformer')``; the PeakLoad
per-unit sniff is ``sum|real(SNet(:,1))| < 20000`` (not ×40); LoadBuses
are ``real(SNet(:,1)) > 0`` (DV-07e — suspected fork bug, config key
``export.powerworld.load_sign`` makes it explicit); GenImport uses the
internal ``Gen.BusNum`` and drops UnitType; SwitchedShunts only at the AC
stage (``codectrl.ACPF`` → the ``ac_stage`` flag).

Multi-unit-bus quirks replicated verbatim (DV-07a pending):
``intersect``/``setdiff`` return ONE generator row per bus, so only the
FIRST unit at a bus receives the bus-level MWSetPoint / 'Open' status.

DV-07c: the sniffs live behind ``units='auto'`` (warn when triggered);
'pu'/'mw' are explicit.

Frame dialect: MATLAB-named S3+ frames (Bus Num/Name/kV/Lat/Lon/[Vm/Va/
PNet/QNet/GenMWSet/SyncCond/MW/MVar], Branch From/To/R/X/B/RateA/Length/
Structure, Gen BusNum/ID/FuelType/Min-Max MW-MVar/A/B/C).
Units: MW/MVar; CSVs headerless (WriteLoads) / headed (imports).
Phase: P5. Validated vs legacy PowerWorld Data CSVs (schema exact; values
for every state-independent column).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..config import WeldConfig
from ..model import WeldError

log = logging.getLogger(__name__)


def _resolve_pu(snet: np.ndarray, *, s_base: float,
                units: Literal["pu", "mw", "auto"], auto_is_pu: bool,
                rule: str) -> np.ndarray:
    """Common DV-07c scaling: 'pu' always scales by s_base, 'mw' never,
    'auto' applies the caller's legacy sniff verdict (warn on trigger)."""
    if units == "pu" or (units == "auto" and auto_is_pu):
        if units == "auto":
            log.warning("auto per-unit sniff triggered (%s) — scaling SNet "
                        "by SBase=%g (DV-07c)", rule, s_base)
        return snet * s_base
    return snet


def write_load_timeseries(snet: np.ndarray, out_dir: Path, *, s_base: float,
                          units: Literal["pu", "mw", "auto"] = "auto",
                          bus_mw: np.ndarray | None = None,
                          sniff_factor: float = 40.0
                          ) -> tuple[Path, Path]:
    """WriteLoads.m: ``|Re(SNet)|ᵀ`` / ``|Im(SNet)|ᵀ`` → TimeSeriesMWLoad
    .csv / TimeSeriesMVarLoad.csv (rows = scenarios, columns = buses, no
    header). ``snet``: (n_bus × n_scen) complex — the caller picks PF.SNet
    or PF.SNetComplete (the legacy isfield branch). 'auto' reproduces the
    ×``sniff_factor`` rule ``Σ|Re(SNet(:,1))|·40 < Σ Bus.MW`` and needs
    ``bus_mw``; ⚠ abs() discards sign — generation-dominated buses are
    written as if they were loads (legacy behavior, doc 07)."""
    snet = np.asarray(snet)
    if units == "auto":
        if bus_mw is None:
            raise ValueError("units='auto' needs bus_mw for the ×40 sniff "
                             "(DV-07c); pass units='mw' or 'pu' instead")
        auto_is_pu = (np.abs(np.real(snet[:, 0])).sum() * sniff_factor
                      < float(np.asarray(bus_mw).sum()))
    else:
        auto_is_pu = False
    snet = _resolve_pu(snet, s_base=s_base, units=units,
                       auto_is_pu=auto_is_pu, rule=f"x{sniff_factor:g} rule")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mw_path = out_dir / "TimeSeriesMWLoad.csv"
    mvar_path = out_dir / "TimeSeriesMVarLoad.csv"
    np.savetxt(mw_path, np.abs(np.real(snet)).T, fmt="%.15g", delimiter=",")
    np.savetxt(mvar_path, np.abs(np.imag(snet)).T, fmt="%.15g", delimiter=",")
    return mw_path, mvar_path


# ---------------------------------------------------------------------------
# per-kind pure builders (WritePowerWorld.m, Aditya fork)
# ---------------------------------------------------------------------------


def bus_import(bus_df: pd.DataFrame, *,
               va_unit: str = "rad") -> pd.DataFrame:
    """PWBus [fork lines 17-33]: Vpu/Vangle only when Bus.Vm exists (the
    try/catch); SubName duplicates Name (SubNumber linkage is the legacy
    'Fix this' comment — name-only, kept).

    DV-07g (the author's ruling 2026-07-19): PowerWorld's Vangle expects
    DEGREES; the S5 pipeline's Bus.Va is RADIANS (the Vpol convention),
    so the default converts here — the historical BusImport CSVs shipped
    radians (wrong). ``va_unit='deg'`` passes a degree-domain frame
    through unchanged."""
    n = len(bus_df)
    out = pd.DataFrame({
        "Number": bus_df["Num"].to_numpy(np.int64),
        "Name": bus_df["Name"].astype(str).to_numpy(),
        "NomkV": bus_df["kV"].to_numpy(float),
        "AreaNumber": np.ones(n, dtype=np.int64),
        "ZoneNumber": np.ones(n, dtype=np.int64),
        "NomB": np.zeros(n),
        "NomG": np.zeros(n),
        "Latitude": bus_df["Lat"].to_numpy(float),
        "Longitude": bus_df["Lon"].to_numpy(float),
    })
    if "Vm" in bus_df.columns:
        va = bus_df["Va"].to_numpy(float)
        out["Vpu"] = bus_df["Vm"].to_numpy(float)
        out["Vangle"] = np.degrees(va) if va_unit == "rad" else va
    out["SubName"] = bus_df["Name"].astype(str).to_numpy()
    return out


def substation_import(bus_df: pd.DataFrame) -> pd.DataFrame:
    """PWSub [fork lines 42-45]: built from the BUS table (the fork has no
    Sub argument — one 'substation' per bus)."""
    return pd.DataFrame({
        "Number": bus_df["Num"].to_numpy(np.int64),
        "Name": bus_df["Name"].astype(str).to_numpy(),
        "Latitude": bus_df["Lat"].to_numpy(float),
        "Longitude": bus_df["Lon"].to_numpy(float),
    })


def branch_import(branch_df: pd.DataFrame, cfg: WeldConfig) -> pd.DataFrame:
    """PWBranch [fork lines 4-6, 55-69]: DeviceType from
    strcmpi(Structure,'Transformer'); LimitMVAB/C = 1.1/1.15 × RateA."""
    pw = cfg.export.powerworld
    xf = (branch_df["Structure"].astype(str).str.strip().str.casefold()
          == "transformer").to_numpy()
    rate = branch_df["RateA"].to_numpy(float)
    return pd.DataFrame({
        "BusNumFrom": branch_df["From"].to_numpy(np.int64),
        "BusNumTo": branch_df["To"].to_numpy(np.int64),
        "Circuit": np.ones(len(branch_df), dtype=np.int64),
        "BranchDeviceType": np.where(xf, "Transformer", "Line"),
        "IsXF": np.where(xf, "YES", "NO"),
        "R": branch_df["R"].to_numpy(float),
        "X": branch_df["X"].to_numpy(float),
        "B": branch_df["B"].to_numpy(float),
        "LimitMVAA": rate,
        "LimitMVAB": pw.limit_b_factor * rate,
        "LimitMVAC": pw.limit_c_factor * rate,
        "LineLength": branch_df["Length"].to_numpy(float),
    })


def peak_load_import(bus_df: pd.DataFrame, snet: np.ndarray | None,
                     cfg: WeldConfig, *,
                     units: Literal["pu", "mw", "auto"] = "auto"
                     ) -> pd.DataFrame:
    """PWLoad [fork lines 77-101]: scenario 1 is assumed the peak snapshot;
    the fork's `<20000` sniff (DV-07c) and `real(SNet(:,1)) > 0` load pick
    (DV-07e — ``export.powerworld.load_sign``); SMW==0 rows deleted."""
    pw = cfg.export.powerworld
    n = len(bus_df)
    out = pd.DataFrame({
        "BusNum": bus_df["Num"].to_numpy(np.int64),
        "ID": np.ones(n, dtype=np.int64),
        "Status": ["Closed"] * n,
        "SMW": np.zeros(n),
        "SMvar": np.zeros(n),
    })
    if snet is not None:
        col0 = np.asarray(snet)[:, 0]
        auto_is_pu = (units == "auto"
                      and np.abs(np.real(col0)).sum() < pw.pu_threshold_mw)
        col0 = _resolve_pu(col0, s_base=cfg.system.s_base_mva, units=units,
                           auto_is_pu=auto_is_pu,
                           rule=f"<{pw.pu_threshold_mw:g} MW rule")
        if pw.load_sign == "positive":                 # Aditya fork (DV-07e)
            load_rows = np.real(col0) > 0
        elif pw.load_sign == "negative":               # injection convention
            load_rows = np.real(col0) < 0
        else:
            raise ValueError(f"load_sign must be 'positive' or 'negative', "
                             f"got {pw.load_sign!r}")
        out.loc[load_rows, "SMW"] = np.abs(np.real(col0[load_rows]))
        out.loc[load_rows, "SMvar"] = np.abs(np.imag(col0[load_rows]))
    else:
        out["SMW"] = bus_df["MW"].to_numpy(float)
        out["SMvar"] = bus_df["MVar"].to_numpy(float)
    return out[out["SMW"] != 0].reset_index(drop=True)


def gen_import(bus_df: pd.DataFrame, gen_df: pd.DataFrame,
               cfg: WeldConfig) -> pd.DataFrame:
    """PWGen [fork lines 109-140] — emitted only when Bus.PNet exists
    (caller gates). DV-07a APPROVED 2026-07-19: the bus-level |PNet|/
    |QNet| setpoints are split across ALL units at the bus proportional
    to MWMax (legacy `intersect` quirk gave everything to the first unit),
    and 'Open' applies to EVERY unit at a decommitted bus (legacy
    `setdiff` opened only the first). Per-bus totals equal the legacy
    bus-level values. The committed-bus length guard stays (a committed
    bus with no generator rows crashed the legacy assignment).
    ⚠ Cost columns are labeled Cubic but carry the quadratic A/B/C fit."""
    pw = cfg.export.powerworld
    bus_num = gen_df["BusNum"].to_numpy(np.int64)
    commit = bus_df.loc[bus_df["GenMWSet"].to_numpy(float)
                        > pw.commit_threshold_mw, "Num"].to_numpy(np.int64)
    decom = np.setdiff1d(np.unique(bus_num), commit)
    com = np.intersect1d(bus_num, commit)
    if len(com) != len(commit):
        raise WeldError(
            f"{len(commit) - len(com)} committed buses have no generator "
            "rows — the legacy MWSetPoint(ComGenIndex)=|PNet(GenBusCommit)| "
            "assignment would have crashed here (length mismatch)")

    g = len(gen_df)
    max_mw = gen_df["MaxMW"].to_numpy(float)
    status = np.array(["Closed"] * g, dtype=object)
    status[np.isin(bus_num, decom)] = "Open"           # every unit (DV-07a)

    bus_cap = pd.Series(max_mw).groupby(bus_num).transform("sum").to_numpy()
    unit_count = pd.Series(np.ones(g)).groupby(bus_num).transform("sum")
    frac = np.where(bus_cap > 0, max_mw / np.where(bus_cap > 0, bus_cap, 1),
                    1.0 / unit_count.to_numpy())       # 0-MWMax bus: equal
    at_com = np.isin(bus_num, com)
    pnet = bus_df.set_index("Num")["PNet"]
    qnet = bus_df.set_index("Num")["QNet"]
    mw_set = np.zeros(g)
    mw_set[at_com] = (np.abs(pnet.loc[bus_num[at_com]].to_numpy(float))
                      * frac[at_com])
    mw_set[mw_set < pw.commit_threshold_mw] = 0.0      # numerical tolerance
    mvar_set = np.zeros(g)
    mvar_set[at_com] = (np.abs(qnet.loc[bus_num[at_com]].to_numpy(float))
                        * frac[at_com])
    return pd.DataFrame({
        "BusNum": bus_num,
        "ID": gen_df["ID"].to_numpy(np.int64),
        "Status": status,
        "FuelType": gen_df["FuelType"].astype(str).to_numpy(),
        "MWMin": gen_df["MinMW"].to_numpy(float),
        "MWMax": max_mw,
        "MvarMin": gen_df["MinMVar"].to_numpy(float),
        "MvarMax": gen_df["MaxMVar"].to_numpy(float),
        "FixedCostInd": gen_df["A"].to_numpy(float),
        "CubicB": gen_df["B"].to_numpy(float),
        "CubicC": gen_df["C"].to_numpy(float),
        "CostModel": ["Cubic"] * g,
        "MWSetPoint": mw_set,
        "MvarSetpoint": mvar_set,
        "VoltSet": np.ones(g),
        "AGC": ["YES"] * g,
        "AVR": ["YES"] * g,
        "PartFact": max_mw / max_mw.sum(),
    })


def switched_shunts_import(bus_df: pd.DataFrame,
                           cfg: WeldConfig) -> pd.DataFrame:
    """PWSS [fork lines 150-172] — AC-stage only (caller gates on
    ``ac_stage``, the fork's ``codectrl.ACPF``). PowerWorld injection-
    positive convention: capacitive rows (QNet > 0) get the main block on
    step 2, inductive on step 1, ∓1 MVar/step token on the opposite side;
    QNet == 0 rows keep zero blocks."""
    pw = cfg.export.powerworld
    sc = np.flatnonzero(bus_df["SyncCond"].to_numpy(float) != 0)
    q = bus_df["QNet"].to_numpy(float)[sc]
    cap = q > 0                                        # ShuntB
    ind = q < 0                                        # ShuntG
    m = len(sc)
    step1 = np.zeros(m)
    step1[ind] = -np.ceil(pw.shunt_margin * np.abs(q[ind]) / pw.block_steps)
    step1[cap] = -1.0
    step2 = np.zeros(m)
    step2[ind] = 1.0
    step2[cap] = np.ceil(pw.shunt_margin * np.abs(q[cap]) / pw.block_steps)
    nom = np.zeros(m)
    nom[cap] = np.ceil(np.abs(q[cap]))
    nom[ind] = -np.ceil(np.abs(q[ind]))
    nums = bus_df["Num"].to_numpy(np.int64)[sc]
    return pd.DataFrame({
        "BusNum": nums,
        "ID": np.ones(m, dtype=np.int64),
        "Status": ["Closed"] * m,
        "RegBusNum": nums,
        "ShuntMode": ["Continuous"] * m,
        "RegulationType": ["Volt"] * m,
        "RegHigh": np.full(m, pw.reg_high),
        "RegLow": np.full(m, pw.reg_low),
        "RegTarget": np.ones(m),
        "BlockMvarPerStep1": step1,
        "BlockNumberStep1": np.full(m, pw.block_steps, dtype=np.int64),
        "BlockMvarPerStep2": step2,
        "BlockNumberStep2": np.full(m, pw.block_steps, dtype=np.int64),
        "MvarNom": nom,
    })


def write_powerworld_imports(bus_df: pd.DataFrame, branch_df: pd.DataFrame,
                             gen_df: pd.DataFrame, snet: np.ndarray | None,
                             out_dir: Path, cfg: WeldConfig, *,
                             ac_stage: bool = False,
                             units: Literal["pu", "mw", "auto"] = "auto",
                             va_unit: str = "rad") -> dict[str, Path]:
    """WritePowerWorld.m (Aditya fork): write every applicable import kind;
    returns kind → path. File names ``<Kind>Import{network_tag}
    {region_tag}.csv`` (both tags empty → the legacy catch-branch names).
    Gen only when ``Bus.PNet`` exists; shunts only when ``Bus.QNet`` exists
    AND ``ac_stage`` (the fork's codectrl.ACPF gate)."""
    pw = cfg.export.powerworld
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{pw.network_tag}{pw.region_tag}"
    frames: dict[str, pd.DataFrame] = {
        "Bus": bus_import(bus_df, va_unit=va_unit),
        "Substation": substation_import(bus_df),
        "Branch": branch_import(branch_df, cfg),
        "PeakLoad": peak_load_import(bus_df, snet, cfg, units=units),
    }
    if "PNet" in bus_df.columns:
        frames["Gen"] = gen_import(bus_df, gen_df, cfg)
    if "QNet" in bus_df.columns and ac_stage:
        frames["SwitchedShunts"] = switched_shunts_import(bus_df, cfg)
    paths: dict[str, Path] = {}
    for kind, frame in frames.items():
        path = out_dir / f"{kind}Import{suffix}.csv"
        frame.to_csv(path, index=False)
        paths[kind] = path
    return paths
