"""EIA-860 generator ingestion + synthetic import generators + unit limits.

Origin: generator/plant cleaning (``generate_grid_script.jl`` lines 74-81,
207) + ``add_import_generators!`` (``helper_functions.jl`` lines 103-132)
— algorithm doc 02 — and ``assigning_pmax_pmin.m`` (S5 prep,
PSERCMain_v2.m:302-303 — algorithm doc 05; CALL_TREE reclassified
ORPHAN→ACTIVE 2026-07-15).

Ported fixes (DV-P2c-3): ``add_import_generators!`` filtered the GLOBAL
``new_nodes`` instead of its ``nodes`` parameter — explicit arguments here.
Jitter draws come from the single seeded Generator (D10/DV3); jittered
coordinates are excluded from exact fixture comparison (|Δ| ≤ 5e-5).

⚠ Generators whose PlantCode is missing from the plant file keep missing
Lat/Lon and would crash snapping — reported here instead of crashing there.

Units: pmax MW; lat/lon degrees; import cost $/MWh linear.
Phase: P2c. Validated vs p1_generators_clean / p3_post_assign_branches_gen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import WeldConfig

#: Julia select! order after cost processing (cost kept as c2/c1/c0 columns).
GEN_COLUMNS = ["index", "Lat", "Lon", "FuelType", "pmax", "PlantCode",
               "GenID", "model", "startup", "shutdown", "ncost",
               "c2", "c1", "c0"]


def load_generators(gen_path: str | Path, plant_path: str | Path,
                    cfg: WeldConfig) -> pd.DataFrame:
    """EIA-860 generator + plant sheets → unit table.

    Filters: State == cfg.eia.state; Status ∉ cfg.eia.exclude_status.
    Renames per the I/O contract; left-join plant Lat/Lon on PlantCode
    (pandas merge preserves left order — the DataFrames.jl leftjoin
    order guarantee caveat in doc 02); ``index`` assigned AFTER the join.
    """
    gens = pd.read_csv(gen_path, low_memory=False)
    plants = pd.read_csv(plant_path, low_memory=False)

    plants = plants[plants["Lat"].notna()].rename(
        columns={"Plant Code": "PlantCode"})
    plants = plants[["PlantCode", "Lat", "Lon"]]

    gens = gens[gens["State"] == cfg.eia.state]
    gens = gens[~gens["Status"].isin(cfg.eia.exclude_status)]
    gens = gens.rename(columns={
        "Plant Code": "PlantCode", "Generator ID": "GenID",
        "Technology": "FuelType", "Nameplate Capacity (MW)": "pmax"})
    gens = gens.merge(plants, on="PlantCode", how="left", sort=False)
    gens = gens.reset_index(drop=True)
    gens["index"] = np.arange(1, len(gens) + 1, dtype=np.int64)

    missing = gens["Lat"].isna()
    if missing.any():
        raise ValueError(
            "generators with no plant coordinates (would crash node_assign): "
            f"PlantCodes {sorted(gens.loc[missing, 'PlantCode'].unique())[:20]}")
    return gens


def add_import_generators(gen_df: pd.DataFrame, added_nodes: pd.DataFrame,
                          substations: pd.DataFrame, bus_df: pd.DataFrame,
                          cfg: WeldConfig,
                          rng: np.random.Generator) -> pd.DataFrame:
    """One synthetic import generator per Import=='IMPORT' bus.

    Order matters for assigned ids: added-node imports FIRST, then
    substation imports, each in row order (helper_functions.jl:110).
    pmax = cfg.costs.import_pmax_mw; linear cost
    cfg.costs.import_linear_usd_per_mwh; Lat/Lon = bus coords + seeded
    jitter of amplitude cfg.build.jitter_deg (DV3). ``gen_bus`` is set
    directly (imports are never snapped → no gen_bus_distance, exports NaN).
    """
    node_ids = pd.concat([
        added_nodes.loc[added_nodes["Import"] == "IMPORT", "index"],
        substations.loc[substations["Import"] == "IMPORT", "index"],
    ]).to_numpy(dtype=np.int64)

    bus_pos = bus_df.set_index("index")
    rows = []
    next_id = int(gen_df["index"].max())
    for bus_id in node_ids:
        next_id += 1
        rows.append({
            "index": next_id,
            "gen_bus": int(bus_id),
            "pmax": cfg.costs.import_pmax_mw,
            "FuelType": "IMPORT",
            "Lat": bus_pos.at[bus_id, "Lat"]
                   + (rng.random() - 0.5) * cfg.build.jitter_deg,
            "Lon": bus_pos.at[bus_id, "Lon"]
                   + (rng.random() - 0.5) * cfg.build.jitter_deg,
            "GenID": 0,
            "PlantCode": 0,
            "c2": 0.0,
            "c1": cfg.costs.import_linear_usd_per_mwh,
            "c0": 0.0,
            "model": 2,
            "startup": 0.0,
            "shutdown": 0.0,
            "ncost": 3,
        })
    return pd.concat([gen_df, pd.DataFrame(rows)], ignore_index=True)


#: raw-xlsx spaced headers → the MATLAB readtable-normalized names the
#: original code (and this port) key on.
_EIA_LIMITS_RENAME = {
    "Plant Code": "PlantCode", "Generator ID": "GeneratorID",
    "Number of Units": "NumberOfUnits",
    "Plant Minimum Load (MW)": "PlantMinimumLoad_MW_",
    "Min MVar": "MinMVar", "Max MVar": "MaxMVar",
}


def _strcmpi_key(value) -> str:
    """MATLAB ``strcmpi(string(a), string(b))`` equivalence key:
    numeric→string drops a trailing .0 (string(34.0) == \"34\");
    comparison is case-insensitive."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).casefold()


def assign_gen_limits(gen_df: pd.DataFrame, eia_860: pd.DataFrame, *,
                      state: str = "CA") -> pd.DataFrame:
    """``assigning_pmax_pmin.m``, faithful sequential port: fill
    Gen.MinMW/MinMVar/MaxMVar from the EIA generator workbook so the S5
    DCOPF has real unit limits (doc 05 §generator limit assignment).

    Semantics replicated verbatim (the function is ORDER-DEPENDENT):

    * rows are visited in order; a row already covered by an earlier
      multi-unit broadcast (``comp_flag``) is skipped for matching but
      still gets the final MinMW ≥ MaxMW → 0 clamp at its OWN iteration
      (so a later broadcast can leave an un-reclamped MinMW — kept);
    * (PlantCode, GeneratorID) matches replay in EIA row order (last match
      wins; the MATLAB inner loop has no break);
    * a multi-unit match divides plant limits by NumberOfUnits, zeroes
      Min_MW_per_gen against the CURRENT row's MaxMW only, then broadcasts
      to EVERY generator at the same BUS (not plant) and marks them done;
    * no plant match → fuel-type fallback: mean over EIA rows whose
      Technology strcmpi-equals the row's FuelType (no Technology match →
      values left untouched);
    * all string comparisons are MATLAB strcmpi on string() conversions
      (case-insensitive; numeric 34.0 == \"34\").

    ``eia_860``: the EIA_Generators_Combined workbook as a DataFrame (raw
    spaced headers or readtable-normalized names both accepted). ``state``:
    the State filter (original hard-codes 'CA'; pass cfg.eia.state).
    Units: MW / MVar. Returns a copy of ``gen_df``."""
    gen = gen_df.copy()
    eia = eia_860.rename(columns=_EIA_LIMITS_RENAME)
    eia = eia[eia["State"].astype(str).str.strip().str.casefold()
              == state.casefold()].reset_index(drop=True)

    e_plant = [_strcmpi_key(v) for v in eia["PlantCode"]]
    e_genid = [_strcmpi_key(v) for v in eia["GeneratorID"]]
    e_tech = [_strcmpi_key(v) for v in eia["Technology"]]
    e_units = eia["NumberOfUnits"].to_numpy(float)
    e_min_mw = eia["PlantMinimumLoad_MW_"].to_numpy(float)
    e_min_mvar = eia["MinMVar"].to_numpy(float)
    e_max_mvar = eia["MaxMVar"].to_numpy(float)
    matches: dict[tuple[str, str], list[int]] = {}
    for j, key in enumerate(zip(e_plant, e_genid)):
        matches.setdefault(key, []).append(j)
    tech_rows: dict[str, list[int]] = {}
    for j, t in enumerate(e_tech):
        tech_rows.setdefault(t, []).append(j)

    n = len(gen)
    min_mw = gen["MinMW"].to_numpy(float).copy()
    min_mvar = gen["MinMVar"].to_numpy(float).copy()
    max_mvar = gen["MaxMVar"].to_numpy(float).copy()
    max_mw = gen["MaxMW"].to_numpy(float)
    busnum = gen["BusNum"].to_numpy(np.int64)
    g_plant = [_strcmpi_key(v) for v in gen["PlantCode"]]
    g_genid = [_strcmpi_key(v) for v in gen["GenID"]]
    g_fuel = [_strcmpi_key(v) for v in gen["FuelType"]]
    data_flag = np.zeros(n, dtype=bool)
    comp_flag = np.zeros(n, dtype=bool)

    for i in range(n):
        if not comp_flag[i]:
            for j in matches.get((g_plant[i], g_genid[i]), []):
                if e_units[j] == 1:
                    min_mw[i] = e_min_mw[j]
                    min_mvar[i] = e_min_mvar[j]
                    max_mvar[i] = e_max_mvar[j]
                    data_flag[i] = True
                else:
                    per_min_mw = e_min_mw[j] / e_units[j]
                    if per_min_mw >= max_mw[i]:
                        per_min_mw = 0.0
                    mates = busnum == busnum[i]
                    min_mw[mates] = per_min_mw
                    min_mvar[mates] = e_min_mvar[j] / e_units[j]
                    max_mvar[mates] = e_max_mvar[j] / e_units[j]
                    data_flag[mates] = True
                    comp_flag[mates] = True
            if not data_flag[i]:
                rows = tech_rows.get(g_fuel[i], [])
                if rows:
                    min_mw[i] = float(np.mean(e_min_mw[rows]))
                    min_mvar[i] = float(np.mean(e_min_mvar[rows]))
                    max_mvar[i] = float(np.mean(e_max_mvar[rows]))
                data_flag[i] = True
            comp_flag[i] = True
        if min_mw[i] >= max_mw[i]:
            min_mw[i] = 0.0

    gen["MinMW"] = min_mw
    gen["MinMVar"] = min_mvar
    gen["MaxMVar"] = max_mvar
    return gen
