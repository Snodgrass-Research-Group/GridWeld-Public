"""Conductor catalog + Form 1 seeding + catalog snapping (S3, P3a).

Origin (all re-verified against the MATLAB sources 2026-07-17):
* PSERCMain.m lines 55-253 — branch-table init (NaN parameter columns,
  Circuits from the string ``Circuit`` column, ``randi(3)`` TowerIndex when
  absent), pooled ConductorData (P2a artifact), short-length fix
  (``Length < 0.1`` mi → 1.28 × haversine/1.60934), PathLength, kV snap,
  and the Form 1 matching loop → :func:`seed_from_form1`.
* ``SizeLines.m`` → :func:`size_lines` (initial nearest-MVA snap + per-kV
  index bounds; the whole-column ``Resize`` zeroing inside the loop is
  harmless-wasteful — done once here).
* ``AssignConductors.m`` → :func:`assign_conductors` — REUSE-AS-IS via
  gridcrucible ``network.conductors.assign_conductors`` (line-for-line
  port) behind a column-name adapter.

The S3 stage works on the RAW MATLAB-named frames (FID/kV/Owner/Length/…)
— the same contract the s2/s3 fixtures dump; the model-name boundary is
``io.matpower.merge_topology`` (D6).

Units: lengths statute miles; ratings MVA; sizes kcmil; GMD feet.
Phase: P3. Validated vs s2_branch_post_form1 (TowerIndex injected) and the
SizeLines columns of s2_branch_post_rxb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gridforge.network.conductors import ConductorCatalog

from ..build.snapping import haversine_km
from ..config import WeldConfig

#: Committed pooled-catalog parquet (built by gridweld.io.matlab_data from
#: ConductorData{66,115,230,500}.mat; verified against the MATLAB dump's
#: s2_conductor_data.csv — 101 rows).
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data" / "conductors.parquet"

#: parquet (gridforge) names → MATLAB pooled-table header.
_TO_MATLAB = {"mva": "MVA", "kv": "kV", "kcmil": "Size",
              "cond_per_phase": "CondPhase", "ra_ohm_per_mi": "Ra",
              "xa_ohm_per_mi": "Xa", "xap_mohm_mi": "XaP",
              "cond_index": "CondIndex"}


def load_cats_catalog(source: str | Path | pd.DataFrame = DEFAULT_CATALOG
                      ) -> ConductorCatalog:
    """gridforge ``ConductorCatalog`` over the CATS pooled table (P2a
    artifact). Built object-directly because the gridforge file loader only
    scans its own 69/138-family kV levels — CATS uses {66,115,230,500}.
    ``conductor_idx`` semantics are identical (1-based pooled row position ≡
    PSERCMain ``CondIndex = (1:height)'``).

    Origin: PSERCMain.m step 4; doc 03 §SizeLines reuse verdict (D5).
    """
    pooled = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    cat = ConductorCatalog.__new__(ConductorCatalog)
    cat.pooled = pooled.reset_index(drop=True)
    cat.by_kv = {int(kv): sub.reset_index(drop=True)
                 for kv, sub in pooled.groupby("kv", sort=True)}
    cat.cutoffs = {}
    cat.max_length_km = None
    return cat


def matlab_catalog(source: str | Path | pd.DataFrame = DEFAULT_CATALOG
                   ) -> pd.DataFrame:
    """The pooled table with the MATLAB header (MVA/kV/Size/CondPhase/…),
    plain 0-based row index — what the S3 loops index via ``CondIndex``
    (1-based row position)."""
    pooled = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    return pooled.rename(columns=_TO_MATLAB).reset_index(drop=True)


def crucible_catalog(source: str | Path | pd.DataFrame = DEFAULT_CATALOG
                     ) -> pd.DataFrame:
    """The pooled table in gridcrucible's master-catalog shape (columns
    mva/kv/size/cond_phase, index = 1-based cond_index) for
    ``gridcrucible.network.conductors`` reuse (D5)."""
    pooled = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    cat = pooled.rename(columns={"kcmil": "size", "cond_per_phase": "cond_phase"})
    cat = cat.reset_index(drop=True)
    cat.index = pd.Index(np.arange(1, len(cat) + 1), name="cond_index")
    return cat


def build_catalog(conductor_dir: Path, voltage_levels: list[int]) -> pd.DataFrame:
    """Pool ConductorData<kV>.mat over the modeled levels; CondIndex := row
    position (PSERCMain step 4). Delegates to the P2a converter — the legacy
    pooled ConductorData.mat is a build artifact, rebuilt, never read."""
    import tempfile

    from ..io.matlab_data import convert_conductor_mats
    out = Path(tempfile.mkdtemp()) / "conductors.parquet"
    return convert_conductor_mats(conductor_dir, out, voltage_levels)


# ---------------------------------------------------------------------------
# PSERCMain preamble + Form 1 matching loop
# ---------------------------------------------------------------------------

def _is_empty(v) -> bool:
    return pd.isna(v) or str(v).strip() == ""


def _ci_eq(series: pd.Series, value) -> pd.Series:
    """strcmpi: case-insensitive equality with NaN-safe fill."""
    return series.fillna("").astype(str).str.casefold() == str(value).casefold()


def init_branch_table(raw: pd.DataFrame, cfg: WeldConfig,
                      rng: np.random.Generator, *,
                      tower_index: np.ndarray | None = None) -> pd.DataFrame:
    """PSERCMain.m lines 55-152: NaN parameter columns, Circuits (string
    ``Circuit`` → numeric, NaN → 1), TowerIndex (``randi(3)`` via the seeded
    Generator — DV3 — unless injected for fixture parity), short-length fix,
    PathLength [mi], ``Resize = 1``, kVFrom/kVTo snapped to voltage classes."""
    br = raw.copy()
    n = len(br)
    if "Circuit" in br.columns and "Circuits" not in br.columns:
        br["Circuits"] = pd.to_numeric(br["Circuit"], errors="coerce")
        br = br.drop(columns=["Circuit"])
    if "Circuits" not in br.columns:
        br["Circuits"] = 1.0
    br["Circuits"] = pd.to_numeric(br["Circuits"], errors="coerce").fillna(1.0)

    for col in ("Size", "CondPerPhase", "RateA", "GMD", "R", "X", "B"):
        br[col] = np.nan
    br["Overloaded"] = 0.0
    br["Undersized"] = 0.0
    if tower_index is not None:
        br["TowerIndex"] = np.asarray(tower_index, dtype=np.int64)
    elif "TowerIndex" not in br.columns:
        choices = np.asarray(cfg.params.tower_spacing_choices, dtype=np.int64)
        br["TowerIndex"] = rng.integers(1, len(choices) + 1, size=n)

    length = pd.to_numeric(br["Length"], errors="coerce").to_numpy(float)
    hav_mi = haversine_km(br["Lat1"].to_numpy(float), br["Lon1"].to_numpy(float),
                          br["Lat2"].to_numpy(float), br["Lon2"].to_numpy(float),
                          cfg.geo.earth_radius_km) / cfg.system.km_per_mile
    short = length < 0.1
    length[short] = cfg.params.routing_factor * hav_mi[short]
    br["Length"] = length
    br["PathLength"] = hav_mi
    br["Resize"] = 1.0

    levels = np.asarray(cfg.system.voltage_levels, dtype=float)
    kv = pd.to_numeric(br["kV"], errors="coerce").to_numpy(float)
    snapped = levels[np.abs(kv[:, None] - levels[None, :]).argmin(axis=1)]
    br["kVFrom"] = snapped
    br["kVTo"] = snapped
    return br


def _rate_from_exact(lp: pd.DataFrame, size: float, cpp: float) -> float | None:
    """MATLAB scalar-assignment semantics: exactly ONE matching row or bust."""
    m = lp[(lp["Size"] == size) & (lp["CondPhase"] == cpp)]
    return float(m["MVA"].iloc[0]) if len(m) == 1 else None


def seed_from_form1(branch_df: pd.DataFrame, form1: pd.DataFrame,
                    tower: pd.DataFrame, catalog: pd.DataFrame,
                    cfg: WeldConfig) -> pd.DataFrame:
    """PSERCMain.m lines 148-252: the Form 1 matching loop.

    Per line: candidate Form 1 rows at the same voltage class — owner-matched
    only for cfg.params.owner_matched_utilities (SDG&E/SCE), UG rows only for
    underground lines, with progressive relaxation on empty; then optional
    Material/TowerType narrowing (revert on empty); the nearest-``Length``
    row donates missing Material/Structure/Size/CondPhase (Size snapped to
    the kV block's catalog sizes); UG lines get Material=Structure=Type='UG'
    and GMD=1 ft; overhead GMD from the TowerData exact-key lookup (no match
    → NaN, the legacy try/catch — RXB raises on those up front); ampacity
    from the catalog with the three-step fallback cascade (each step demands
    EXACTLY one matching row — MATLAB scalar assignment).

    ``branch_df`` must come from :func:`init_branch_table`. Returns a copy.
    """
    br = branch_df.copy()
    owners = {u.casefold() for u in cfg.params.owner_matched_utilities}
    f_volt = pd.to_numeric(form1["Voltage"], errors="coerce")
    f_ug = form1["MaterialType"].fillna("").astype(str).str.casefold() == "ug"
    f_util = form1["Utility"].fillna("").astype(str).str.casefold()
    f_mat = form1["MaterialType"].fillna("").astype(str).str.casefold()
    f_tow = form1["TowerType"].fillna("").astype(str).str.casefold()
    t_volt = pd.to_numeric(tower["Voltage"], errors="coerce")
    t_mat = tower["Material"].fillna("").astype(str).str.casefold()
    t_con = tower["Construction"].fillna("").astype(str).str.casefold()
    t_spc = pd.to_numeric(tower["Spacing"], errors="coerce")

    for i in br.index:
        kv = float(br.at[i, "kVFrom"])
        owner = str(br.at[i, "Owner"]) if not pd.isna(br.at[i, "Owner"]) else ""
        is_ug = str(br.at[i, "Type"]).casefold() == "ug"
        ug_mask = f_ug if is_ug else ~f_ug

        cand = None
        if owner.casefold() in owners:
            cand = form1[(f_volt == kv) & (f_util == owner.casefold()) & ug_mask]
            if cand.empty:
                cand = form1[(f_volt == kv) & ug_mask]
        else:
            cand = form1[(f_volt == kv) & ug_mask]
        if cand.empty:
            cand = form1[f_volt == kv]
        if cand.empty:
            raise ValueError(f"Form 1 has no rows at {kv} kV (line FID "
                             f"{br.at[i, 'FID']}) — legacy would crash here")
        lp = catalog[catalog["kV"] == kv]

        mat = br.at[i, "Material"]
        if not _is_empty(mat):
            narrowed = cand[f_mat.loc[cand.index] == str(mat).casefold()]
            if not narrowed.empty:
                cand = narrowed
        struct = br.at[i, "Structure"]
        if not _is_empty(struct):
            narrowed = cand[f_tow.loc[cand.index] == str(struct).casefold()]
            if not narrowed.empty:
                cand = narrowed

        lengths = pd.to_numeric(cand["Length"], errors="coerce").to_numpy(float)
        j = cand.index[int(np.nanargmin(np.abs(float(br.at[i, "Length"])
                                               - lengths)))]
        if _is_empty(br.at[i, "Material"]):
            br.at[i, "Material"] = form1.at[j, "MaterialType"]
        if _is_empty(br.at[i, "Structure"]):
            br.at[i, "Structure"] = form1.at[j, "TowerType"]
        if pd.isna(br.at[i, "Size"]):
            size = float(form1.at[j, "Size"])
            k = int(np.argmin(np.abs(size - lp["Size"].to_numpy(float))))
            br.at[i, "Size"] = float(lp["Size"].iloc[k])
        if pd.isna(br.at[i, "CondPerPhase"]):
            br.at[i, "CondPerPhase"] = float(form1.at[j, "CondPhase"])

        # --- GMD [PSERCMain:214-230] ---
        if is_ug:
            br.at[i, "Material"] = "UG"
            br.at[i, "Structure"] = "UG"
            br.at[i, "Type"] = "UG"
            br.at[i, "GMD"] = 1.0
        else:
            hit = tower[(t_volt == kv)
                        & (t_mat == str(br.at[i, "Material"]).casefold())
                        & (t_con == str(br.at[i, "Structure"]).casefold())
                        & (t_spc == float(br.at[i, "TowerIndex"]))]
            if len(hit) == 1:                 # try/catch: no match → GMD NaN
                br.at[i, "GMD"] = float(hit["GMD"].iloc[0])

        # --- ampacity cascade [PSERCMain:232-251] ---
        size = float(br.at[i, "Size"])
        cpp = float(br.at[i, "CondPerPhase"])
        rate = _rate_from_exact(lp, size, cpp)
        if rate is None:
            m = lp[lp["Size"] == size]
            if len(m) == 1:
                rate = float(m["MVA"].iloc[0])
        if rate is None:
            tmp = lp[lp["CondPhase"] == cpp]
            if tmp.empty:
                k = int(np.argmin(np.abs(size - lp["Size"].to_numpy(float))))
                br.at[i, "Size"] = float(lp["Size"].iloc[k])
                br.at[i, "CondPerPhase"] = float(lp["CondPhase"].iloc[k])
            else:
                k = int(np.argmin(np.abs(size - tmp["Size"].to_numpy(float))))
                br.at[i, "Size"] = float(tmp["Size"].iloc[k])
            rate = _rate_from_exact(lp, float(br.at[i, "Size"]),
                                    float(br.at[i, "CondPerPhase"]))
            if rate is None:                  # uncaught in MATLAB — hard stop
                raise ValueError(
                    f"ampacity cascade found no unique catalog row for line "
                    f"FID {br.at[i, 'FID']} (Size {br.at[i, 'Size']}, "
                    f"CondPerPhase {br.at[i, 'CondPerPhase']}, {kv} kV)")
        br.at[i, "RateA"] = rate
    return br


# ---------------------------------------------------------------------------
# SizeLines + AssignConductors
# ---------------------------------------------------------------------------

def size_lines(branch_df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """SizeLines.m: per kV block, ``CondIndex`` = pooled index of
    argmin |MVA − RateA| (first-row tie-break); ``Max/MinCondIndex`` = block
    extents; Size/CondPerPhase/RateA snapped to the chosen row's exact table
    values; ``Resize`` zeroed. Returns a copy (MATLAB-named frame)."""
    br = branch_df.copy()
    cond = np.zeros(len(br), dtype=np.int64)
    cmax = np.zeros(len(br), dtype=np.int64)
    cmin = np.zeros(len(br), dtype=np.int64)
    kv_arr = br["kVFrom"].to_numpy(float)
    rate = br["RateA"].to_numpy(float)
    for kv in np.unique(kv_arr):
        blk = catalog[catalog["kV"] == kv]
        if blk.empty:
            raise ValueError(f"no conductors at {kv} kV in the catalog")
        mva = blk["MVA"].to_numpy(float)
        ci = blk["CondIndex"].to_numpy(np.int64)
        rows = np.flatnonzero(kv_arr == kv)
        pick = np.abs(mva[None, :] - rate[rows, None]).argmin(axis=1)
        cond[rows] = ci[pick]
        cmax[rows] = ci.max()
        cmin[rows] = ci.min()
    br["CondIndex"] = cond
    br["MaxCondIndex"] = cmax
    br["MinCondIndex"] = cmin
    br["Size"] = catalog["Size"].to_numpy(float)[cond - 1]
    br["CondPerPhase"] = catalog["CondPhase"].to_numpy(float)[cond - 1]
    br["RateA"] = catalog["MVA"].to_numpy(float)[cond - 1]
    br["Resize"] = 0.0
    return br


#: MATLAB branch header ⇄ gridcrucible candidate columns (assign_conductors).
_CRUCIBLE_RENAME = {"RateA": "capacity_mva", "CondIndex": "conductor_idx",
                    "Circuits": "n_circuits", "kVFrom": "voltage_kv_from",
                    "kVTo": "voltage_kv_to", "Size": "kcmil",
                    "CondPerPhase": "cond_per_phase", "Resize": "resize",
                    "MaxCondIndex": "max_conductor_idx",
                    "MinCondIndex": "min_conductor_idx",
                    "Overloaded": "overloaded", "Undersized": "undersized"}


def assign_conductors(branch_df: pd.DataFrame, catalog_crucible: pd.DataFrame,
                      change_voltages: bool = False) -> pd.DataFrame:
    """AssignConductors.m — REUSE-AS-IS (gridcrucible
    ``network.conductors.assign_conductors``, a line-for-line port: per-
    circuit rating, nearest-MVA within the kV block — or globally with a
    ``min_kv`` floor when voltages may change — snap Size/CondPerPhase,
    RateA = Circuits × table MVA, clear Resize). Operates on the
    MATLAB-named frame through a column adapter (D5). Returns a copy."""
    from gridcrucible.network.conductors import assign_conductors as _ac
    view = branch_df.rename(columns=_CRUCIBLE_RENAME)
    _ac(view, catalog_crucible, change_voltages)
    return view.rename(columns={v: k for k, v in _CRUCIBLE_RENAME.items()})
