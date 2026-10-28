""".mat / legacy-CSV → parquet/npz one-time converters (S3 input data).

Origin (algorithm doc 03 §Input data):
* ``ConductorData{66,115,230,345,500,765}.mat`` — plain numeric arrays
  (scipy-readable), columns MVA,kV,Size,CondPhase,Ra,Xa,XaP,CondIndex.
  Pooled per PSERCMain step 4: concatenate the configured kV levels in
  order, then renumber CondIndex = row position (1-based) — the identical
  construction gridcrucible's ``migrate_conductor_tables`` documents.
* ``TowerData.csv`` (81 rows) — lookup key must be UNIQUE per
  (Voltage, Material, Construction, Spacing); validated here.
* ``Form1.csv`` (292 rows) — ⚠ column ``'State '`` has a trailing space;
  names are stripped and ``Material Type``/``Tower Type`` renamed.
* ``xfdata.csv`` / ``xfxr.csv`` — dump-script CSV exports (blocker B1: the
  .mat originals are MATLAB ``table`` objects scipy cannot read; identical
  blocker + fix as gridtemper/gridcrucible ``export_opaque_tables.m``).
* ``GMDDataCAISO.m`` — only the Xd/XdPrime vectors are consumed (the
  1000-row inverse-CDF GMD draw is commented out in RXB.m); exported as an
  npz in gridforge ``GMDTables`` layout so ``compute_rxb`` reads it as-is.

Output schema reconciliation (D5): the pooled conductor parquet uses
gridforge's column names (``mva, kv, kcmil, cond_per_phase, ra_ohm_per_mi,
xa_ohm_per_mi, xap_mohm_mi, cond_index``) so
``gridforge.network.conductors.ConductorCatalog`` and ``compute_rxb``
consume it without adaptation (see ``gridweld.params.conductors``).

Units: MVA ratings; kV classes; kcmil sizes; Ra/Xa in Ω/mi; XaP in MΩ·mi;
GMD in feet; Xd in Ω/mi; XdPrime in MΩ·mi.

Phase: P2a.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

#: MATLAB array column order of ConductorData<kV>.mat → gridforge names.
CONDUCTOR_COLUMNS = ["mva", "kv", "kcmil", "cond_per_phase",
                     "ra_ohm_per_mi", "xa_ohm_per_mi", "xap_mohm_mi",
                     "cond_index"]

#: MATLAB-side header of the pooled table (fixture s2_conductor_data.csv).
CONDUCTOR_MATLAB_HEADER = ["MVA", "kV", "Size", "CondPhase", "Ra", "Xa",
                           "XaP", "CondIndex"]


def _write_parquet(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def _read_csv_lenient(path: str | Path, **kw) -> pd.DataFrame:
    """UTF-8 first, cp1252 fallback (Form1.csv's 'State ' column actually
    ends in a 0xA0 non-breaking space — Excel-era encoding)."""
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kw)


def convert_conductor_mats(conductor_dir: str | Path, out_path: str | Path,
                           voltage_levels: list[int]) -> pd.DataFrame:
    """ConductorData<kV>.mat set → one pooled parquet.

    ``cond_index`` is renumbered to the 1-based POOLED row position
    (PSERCMain step 4: ``CondIndex = (1:height)'``) — the per-file legacy
    ids are discarded, matching both PSERCMain and gridforge semantics.
    """
    conductor_dir = Path(conductor_dir)
    frames = []
    for kv in voltage_levels:
        mat = sio.loadmat(conductor_dir / f"ConductorData{kv}.mat")
        arr = mat[f"ConductorData{kv}"]
        if arr.shape[1] != len(CONDUCTOR_COLUMNS):
            raise ValueError(f"ConductorData{kv}: expected "
                             f"{len(CONDUCTOR_COLUMNS)} columns, got {arr.shape[1]}")
        df = pd.DataFrame(arr, columns=CONDUCTOR_COLUMNS)
        if not (df["kv"] == kv).all():
            raise ValueError(f"ConductorData{kv}: kV column disagrees with filename")
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled["cond_index"] = np.arange(1, len(pooled) + 1)
    _write_parquet(pooled, Path(out_path))
    return pooled


def convert_tower_data(csv_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """TowerData.csv → parquet; asserts a unique lookup key per
    (Voltage, Material, Construction, Spacing). ``Source``/``Notes`` are
    mixed-type commentary — carried as strings, never consumed."""
    df = _read_csv_lenient(csv_path)
    df.columns = [c.strip() for c in df.columns]
    key = ["Voltage", "Material", "Construction", "Spacing"]
    dup = df.duplicated(subset=key)
    if dup.any():
        raise ValueError(f"TowerData: non-unique lookup keys:\n{df.loc[dup, key]}")
    for col in ("Source", "Notes"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    _write_parquet(df, Path(out_path))
    return df


def convert_form1(csv_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Form1.csv → parquet. Column names stripped ('State ' trailing space)
    and ``Material Type``/``Tower Type`` → MaterialType/TowerType (the
    readtable-normalized names PSERCMain's Form 1 loop uses)."""
    df = _read_csv_lenient(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Material Type": "MaterialType",
                            "Tower Type": "TowerType",
                            "Conductor Type": "ConductorType"})
    _write_parquet(df, Path(out_path))
    return df


def convert_xf_tables(xfdata_csv: str | Path, xfxr_csv: str | Path,
                      out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """XFData/XFXR dump-script CSVs → parquet (blocker B1 resolution).

    XFData: Pri/Sec/Ter [kV], XHL/XHT/XLT [% impedance on own rating].
    XFXR: MVA → X/R ratio curve (interp1 source in TransformerImpedances.m).
    """
    out_dir = Path(out_dir)
    xfdata = pd.read_csv(xfdata_csv)
    xfxr = pd.read_csv(xfxr_csv)
    for name, df, required in (("xfdata", xfdata, {"Pri", "Sec", "XHL"}),
                               ("xfxr", xfxr, {"MVA", "XR"})):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if not xfxr["MVA"].is_monotonic_increasing:
        raise ValueError("xfxr: MVA column must be ascending for interp1")
    _write_parquet(xfdata, out_dir / "xfdata.parquet")
    _write_parquet(xfxr, out_dir / "xfxr.parquet")
    return xfdata, xfxr


_GMD_VEC = re.compile(r"(Xd|XdPrime)\s*=\s*\[([^\]]+)\];", re.S)


def convert_gmd_tables(gmd_m_path: str | Path, out_path: str | Path) -> dict:
    """GMDData<REGION>.m → npz in gridforge ``GMDTables`` layout.

    Only ``xd`` (Ω/mi) and ``xd_prime`` (MΩ·mi), both indexed by integer GMD
    feet (1-based), are real data — RXB.m's inverse-CDF ``GMDTable`` draw is
    commented out for GridWeld (GMD comes from TowerData), so ``gmd_table``
    is a zeros placeholder that satisfies the ``GMDTables`` constructor and
    must never be drawn from (``compute_rxb`` is always called with
    all-nonzero ``gmd_ft``).

    Regex reuse: gridcrucible ``io.matlab_import._GMD_VEC``.
    """
    text = Path(gmd_m_path).read_text(encoding="utf-8", errors="replace")
    vecs = {name: np.array([float(x) for x in re.split(r"[;,\s]+", body.strip())])
            for name, body in _GMD_VEC.findall(text)}
    if set(vecs) != {"Xd", "XdPrime"} or len(vecs["Xd"]) != len(vecs["XdPrime"]):
        raise ValueError(f"{gmd_m_path}: bad Xd/XdPrime vectors")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, gmd_table=np.zeros((1000, 6)),
                        xd=vecs["Xd"], xd_prime=vecs["XdPrime"])
    return {"xd_len": len(vecs["Xd"]), "path": str(out_path)}


def migrate_all(cfg, data_dir: str | Path | None = None) -> dict[str, str]:
    """Run every converter against the legacy tree at ``cfg.paths.root``.

    Outputs land in ``data_dir`` (default: ``src/gridweld/data`` next to
    this package — small, committed artifacts, same pattern as gridforge's
    ``data_store``). Idempotent overwrite; returns {artifact: path}.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir = Path(data_dir)
    p = cfg.paths
    written: dict[str, str] = {}
    pooled = convert_conductor_mats(p.resolve(p.conductor_dir),
                                    data_dir / "conductors.parquet",
                                    list(cfg.system.voltage_levels))
    written["conductors.parquet"] = f"{len(pooled)} rows"
    written["tower.parquet"] = f"{len(convert_tower_data(p.resolve(p.tower_data), data_dir / 'tower.parquet'))} rows"
    written["form1.parquet"] = f"{len(convert_form1(p.resolve(p.form1), data_dir / 'form1.parquet'))} rows"
    xfdata, xfxr = convert_xf_tables(p.resolve(p.xfdata_csv),
                                     p.resolve(p.xfxr_csv), data_dir)
    written["xfdata.parquet"] = f"{len(xfdata)} rows"
    written["xfxr.parquet"] = f"{len(xfxr)} rows"
    gmd_m = p.resolve(Path("Jonathan_Code/Data Files/GMD Statistics and lookup tables/"
                           f"GMDData{cfg.system.region}.m"))
    written["gmd_tables.npz"] = str(convert_gmd_tables(
        gmd_m, data_dir / "gmd_tables.npz")["xd_len"]) + " xd entries"
    return written
