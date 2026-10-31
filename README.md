# GridWeld

**gridweld** welds public GIS and tabular data (HIFLD/CEC transmission
shapefiles, EIA-860, FERC Form 1, CAISO data) into a geographically accurate
synthetic transmission test system. Its first target is reproducing **CATS —
the California Test System** (Sofia Taylor, Noah Rhodes, Aditya Rangarajan,
Jonathan Snodgrass, Bernie Lesieutre, Line Roald — UW–Madison).

This is a Python consolidation of the original three-language CATS pipeline
(ArcGIS/ArcPy → Julia → MATLAB; its repository, WISPO-POP/WildfireMapData, is
not public) into one package. It belongs to a family of synthetic-grid tools:
[GridForge](https://github.com/Snodgrass-Research-Group/GridForge-Public)
(topology), [GridCrucible](https://github.com/Snodgrass-Research-Group/GridCrucible-Public)
(KCL-KVL), [GridTemper](https://github.com/Snodgrass-Research-Group/GridTemper-Public)
(case tempering), [LoadSmith](https://github.com/Snodgrass-Research-Group/LoadSmith-Public)
(load profiles) and **GridWeld** (real-GIS test systems).

## Install

Python 3.12 or 3.13 (`.python-version` pins 3.13), managed with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

`gridforge` and `gridcrucible` install from their public repositories (git
dependencies in `pyproject.toml`; `uv.lock` pins the commits). To work against
local clones instead, run `uv pip install -e ../GridForge-Public -e ../GridCrucible-Public`
after syncing.

With pip:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

pip reads the same git dependencies. (The PyPI project named `gridforge` is an
unrelated table-formatting library; do not install it.)

You also need a Gurobi licence: the size-limited licence bundled with the pip
package is too small for the CATS models. Gurobi is used by the `build` stage
(load-to-bus MILP, unless `[loads] solver = "assignment"`), the `scenarios`
stage (dispatch QP) and the `dcopf` stage. ArcGIS Pro is needed only for the
arcpy GIS-cleaning code in `src/gridweld/gis/arcpy_backend/`, which the default
run does not use.

## Input data

The CATS inputs are not in this repository, and the repository that holds them
(WISPO-POP/WildfireMapData) is not public. To request them, contact the CATS
authors named above. The published CATS model is public at
[WISPO-POP/CATS-CaliforniaTestSystem](https://github.com/WISPO-POP/CATS-CaliforniaTestSystem),
but it does not include this input tree.

Once you have a copy of the inputs, point the run at it: copy
`configs/cats.toml` (for example to `configs/cats_local.toml`) and add under
`[paths]`:

```toml
root = "C:/path/to/WildfireMapData"
```

`root` must hold the original CATS tree (`Data/`, `Jonathan_Code/`, ...). The
files shipped in `src/gridweld/data/` (including the corrected CAISO load
apportionment) are found whatever `root` is.

## Usage

Console script — runs the pipeline from a TOML run file (set `[paths] root`
first, see **Input data**):

```bash
gridweld                      # defaults to configs/cats.toml
gridweld configs/cats.toml --verbose
gridweld --stages build,parameterize        # run a subset
gridweld --resume-from dc_resize            # resume from a checkpoint
```

Python API:

```python
from gridweld import WeldConfig, WeldModel, STAGES
from gridweld.pipeline import GridWeld

cfg = WeldConfig.from_toml("configs/cats.toml")
GridWeld(cfg).run()
```

## Pipeline stages

Canonical order (`gridweld.config.STAGES`):

`gis_clean` (separate, ArcGIS Pro only) → `build` → `parameterize` → `merge_topology` → `scenarios` →
`size_transformers` → `dc_resize` → `dcopf` → `reactive` → `ac_resize` →
`export`

Each stage is a `WeldModel → WeldModel` step; the model and RNG state are
checkpointed after every stage so `--resume-from` reproduces the exact
continuation.

## Package layout

```
src/gridweld/
  config.py        WeldConfig + option dataclasses + STAGES
  model.py         WeldModel (pandas-frame data model) + schemas
  pipeline.py      GridWeld orchestrator (checkpointed stages, CLI)
  gis/             S1 GIS cleaning (arcpy + open backends), islands, nodes
  build/           S2 build: substations, snapping, costs, generators, loads
  params/          S3 conductors, impedance (RXB), transformers
  dispatch/        UCED/UCCD dispatch, SNet scenario injections
  power/           Ybus, DC/AC power-flow primitives
  resize/          DC/AC resize loops, reactive placement
  feasibility/     S5 DCOPF feasibility
  io/              MATPOWER, MATLAB .mat, GeoJSON, PowerWorld I/O
  viz/             metrics + plots
  data/            shipped parquet/csv inputs (conductors, tower, Form 1, …)
configs/cats.toml  California/CATS run configuration
tests/             pytest suite
```

## Development

```bash
uv run pytest -m "not slow"       # test suite (slow markers skipped)
uv run ruff check src tests       # lint
```

Fixture-backed regression tests (reference dumps of the original Julia and
MATLAB runs) and tests that read the original CATS tree skip when those files
are absent; 110 tests run on a fresh clone.

Labels in the code such as D5 (design decision), P3 (conversion phase) and
DV-P3-2 (deviation from the MATLAB code) refer to the conversion notes, which
are not part of this repository; each deviation is explained where it is
implemented.

## Running

Run from the repo root with the environment active, or put `uv run` in front of each command. The default config argument is the relative path `configs/cats.toml`. By default the `outputs/` and `checkpoints/` folders go to the repo root, whatever the working folder.

1. Full run with your config:

   ```bat
   gridweld configs\cats_local.toml --verbose
   ```

2. Resume after a failed or interrupted stage:

   ```bat
   gridweld configs\cats_local.toml --resume-from dcopf
   ```

   This loads the checkpoint of the stage before `dcopf` (model and random-number state) and continues from there.
3. To assign loads to buses without Gurobi's MILP, set `solver = "assignment"` under `[loads]`. It uses `scipy.optimize.linear_sum_assignment`; the `build/loads.py` docstring states it reaches the same optimum objective. The `scenarios` and `dcopf` stages still need Gurobi.

## Inputs and outputs

Inputs (relative to `[paths] root` unless noted):

- `build`: `Data/California_Electric_Substations_11_17_21.csv`, `Data/Added_Nodes_11_17_21.csv`, `Data/EIA_Generator_Y2019.csv`, `Data/EIA_Plant_Y2019.csv`, `Data/Generator_Cost_Curves.csv`, `Jonathan_Code/Input Data/CAGISData Removed planned lines.csv`, and the shipped load file `src/gridweld/data/caiso_loads_corrected.csv`.
- `parameterize`: `Jonathan_Code/Input Data/CAGISData.csv`, plus the shipped tables in `src/gridweld/data/` (`conductors.parquet`, `form1.parquet`, `tower.parquet`, `gmd_tables.npz`). `xfdata.parquet` and `xfxr.parquet` are used from `size_transformers` onward.
- `dcopf`: `Jonathan_Code/Aditya_Code/Data Files/EIA_Generators_Combined_2010.xlsx` (read with pandas and openpyxl).
- `export` (optional GeoJSON round-trip): `Data/GIS/lines.geojson`, `Data/GIS/added_nodes.geojson`, `Data/CATS_transmission_line_data.csv`. The round-trip is skipped with a warning if any of them, or `outputs/cal_largest_grid.m` from `build`, is missing.

Outputs (in `outputs/` at the repo root, set by `[paths] out_dir`):

- `build`: `cal_largest_grid.m` (MATPOWER case of the largest connected network) and `qa/`, with one CSV per check (`self_loops`, `tiny_lines`, `transformers`, `grid_loads`, `largest_grid_loads`, `long_distance_generators`, `long_distance_transmission`, `transmission_same_endpoints`) plus `bus_loads.html`.
- `export`: `final_case.m`; `TimeSeriesMWLoad.csv` and `TimeSeriesMVarLoad.csv`; PowerWorld import tables `BusImportCAISO.csv`, `SubstationImportCAISO.csv`, `BranchImportCAISO.csv`, `PeakLoadImportCAISO.csv`, plus `GenImportCAISO.csv` and `SwitchedShuntsImportCAISO.csv` when the bus table carries the `PNet`/`QNet` columns that the `dcopf` stage adds (switched shunts only when `ac_resize` is in the stage list); `tl.json`, `bus.json` and `gen.json` from the GeoJSON round-trip.
- After every stage: `checkpoints/NN_<stage>.pkl` (pickled model and random-number state; `[paths] checkpoint_dir`).

Both `outputs/` and `checkpoints/` are gitignored.

## Known issues

- The `gis_clean` stage always raises `WeldError`: the arcpy code runs only
  under ArcGIS Pro's Python and the pipeline does not call it. It is not in the
  default stage list.
- The open-source GIS backend (`gis.backend = "open"`,
  `src/gridweld/gis/open_backend/`) is a docstring-only placeholder.
- The `[eval]` section of `configs/cats.toml` is read by a Julia evaluation
  module that is not part of this repository.

## Data sources

The data files in `src/gridweld/data/` are licensed under CC BY 4.0 (see
[LICENSE-DATA](LICENSE-DATA)); the original sources keep their own terms.

| Files | Contents | Original source |
|---|---|---|
| `caiso_loads_corrected.csv` (+ `.provenance.json`) | CAISO load apportionment, corrected to the 47,257 MW state peak | Built from LoadSmith's CAISO bus table (census-tract load shares) |
| `conductors.parquet`, `tower.parquet` | Conductor catalogue and tower geometry | CATS MATLAB tables (`Jonathan_Code/`) |
| `form1.parquet` | Transmission-line statistics | FERC Form 1 (public), via the CATS tables |
| `gmd_tables.npz` | Conductor-spacing (GMD) tables for CAISO | CATS `GMDDataCAISO.m` |
| `xfdata.parquet`, `xfxr.parquet` | Typical transformer reactances and X/R ratios | CATS MATLAB tables |

## License

* Code: MIT — see [LICENSE](LICENSE).
* Data in `src/gridweld/data/`: CC BY 4.0 — see [LICENSE-DATA](LICENSE-DATA).

Copyright (c) 2026 Jonathan M. Snodgrass for the Python consolidation. The
algorithms come from the CATS pipeline by the authors named above; please cite
CATS as well (see [CITATION.cff](CITATION.cff)).

## Acknowledgment

This work was supported by the Power Systems Engineering Research Center
(PSERC) under Project S-91, "Generating Value from Detailed, Realistic
Synthetic Electric Grids."

## Related repositories

- WISPO-POP/WildfireMapData (not public): the original CATS pipeline and input
  data.
- [WISPO-POP/CATS-CaliforniaTestSystem](https://github.com/WISPO-POP/CATS-CaliforniaTestSystem):
  the published CATS model (MATPOWER and GIS files) that `gridweld` aims to
  reproduce.
- [GridForge](https://github.com/Snodgrass-Research-Group/GridForge-Public):
  the `gridforge` package (conductor catalogue, line impedance, Ybus, DC and
  AC power flow) that `gridweld` imports.
- [GridCrucible](https://github.com/Snodgrass-Research-Group/GridCrucible-Public):
  the `gridcrucible` package (`ScenarioSet`, conductor resizing, UCCD dispatch
  and its QP, scenario helpers) that `gridweld` imports.
- [LoadSmith](https://github.com/Snodgrass-Research-Group/LoadSmith-Public):
  source of the CAISO bus table behind `caiso_loads_corrected.csv`.
