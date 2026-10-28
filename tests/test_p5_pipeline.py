"""P5 pipeline validation: checkpoint/resume mechanics (fast, fake stages),
the MATLAB-frames → MATPOWER final-case conversion on the dcopf fixture
state, CLI plumbing, and the end-to-end CATS smoke run (slow — reduced
scenario counts + capped loop passes via config; gis_clean excluded, its
cleaned CATS layers are checked in)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridweld.config import REPO_ROOT, WeldConfig
from gridweld.model import WeldError, WeldModel
from gridweld.pipeline import GridWeld

FIX = REPO_ROOT / "validation" / "fixtures" / "dcopf"
needs_dcopf = pytest.mark.skipif(not (FIX / "manifest.txt").exists(),
                                 reason="dcopf fixtures not extracted")


def _tmp_cfg(tmp_path) -> WeldConfig:
    cfg = WeldConfig()
    cfg.paths.out_dir = tmp_path / "outputs"
    cfg.paths.log_dir = tmp_path / "outputs" / "logs"
    cfg.paths.checkpoint_dir = tmp_path / "checkpoints"
    return cfg


def test_checkpoint_roundtrip_restores_rng(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    gw = GridWeld(cfg)
    model = WeldModel(buses=pd.DataFrame({"Num": [1, 2]}),
                      meta={"marker": 42})
    gw.rng.random(5)                              # advance the stream
    expected_next = gw.rng.bit_generator.state
    gw.rng = np.random.default_rng()
    gw.rng.bit_generator.state = expected_next
    probe = gw.rng.random()

    gw2 = GridWeld(cfg)
    gw2.rng.random(5)
    gw2._save_checkpoint("build", model)
    gw3 = GridWeld(cfg)
    loaded = gw3._load_checkpoint("build")
    assert loaded.meta["marker"] == 42
    assert gw3.rng.random() == probe              # exact stream continuation


def test_run_resume_skips_completed_stages(tmp_path, monkeypatch):
    cfg = _tmp_cfg(tmp_path)
    cfg.stages = ["build", "parameterize", "merge_topology"]
    calls: list[str] = []

    def fake(stage):
        def _f(self, model):
            calls.append(stage)
            model.meta.setdefault("trail", []).append(stage)
            return model
        return _f

    for s in cfg.stages:
        monkeypatch.setattr(GridWeld, f"_stage_{s}", fake(s))
    gw = GridWeld(cfg)
    model = gw.run()
    assert calls == cfg.stages
    assert model.meta["trail"] == cfg.stages

    calls.clear()
    gw2 = GridWeld(cfg)
    model = gw2.run(resume_from="merge_topology")
    assert calls == ["merge_topology"]            # prior stages from disk
    assert model.meta["trail"] == cfg.stages

    with pytest.raises(WeldError):
        GridWeld(cfg).run(resume_from="dcopf")    # not in configured stages


def test_gis_clean_stage_raises_without_arcpy(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    with pytest.raises(WeldError, match="propy"):
        GridWeld(cfg)._stage_gis_clean(WeldModel())


@needs_dcopf
def test_final_case_conversion(tmp_path):
    """MATLAB S3+ frames (the dcopf fixture state) → MATPOWER case: ids,
    ratings, the A/B/C ↔ gencost column mapping, baseMVA = SBase."""
    from gridweld.io.matpower import read_matpower_case, write_matpower_case

    cfg = _tmp_cfg(tmp_path)
    model = WeldModel(
        buses=pd.read_csv(FIX / "in_bus.csv", low_memory=False),
        branches=pd.read_csv(FIX / "in_branch.csv", low_memory=False),
        generators=pd.read_csv(FIX / "in_gen.csv", low_memory=False))
    case = GridWeld(cfg)._final_case(model)
    assert case.base_mva == 100.0
    assert case.bus.shape[0] == 8848 and case.gen.shape[0] == 2229
    assert np.allclose(case.branch[:, 5],
                       model.branches["RateA"].to_numpy(float))
    assert np.allclose(case.arrays["gencost"][:, 4],
                       model.generators["C"].to_numpy(float))
    assert np.allclose(case.arrays["gencost"][:, 6],
                       model.generators["A"].to_numpy(float))
    path = write_matpower_case(case, tmp_path / "final_case.m")
    back = read_matpower_case(path)
    assert np.allclose(back.branch[:, 3], case.branch[:, 3])   # X roundtrip


def test_cli_help_and_bad_stage(monkeypatch, capsys):
    from gridweld.pipeline import main
    monkeypatch.setattr("sys.argv", ["gridweld", "--help"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    monkeypatch.setattr(
        "sys.argv",
        ["gridweld", str(REPO_ROOT / "configs" / "cats.toml"),
         "--stages", "nonsense"])
    with pytest.raises(SystemExit):
        main()


# ---------------------------------------------------------------------------
# the end-to-end CATS smoke run (P5 deliverable)
# ---------------------------------------------------------------------------


@needs_dcopf
@pytest.mark.slow
def test_cats_smoke_run(tmp_path):
    """build → … → export on the CATS config at smoke scale (4 load
    scenarios → 8 SNetFull columns; loop passes capped; assignment-reduction
    MILP — DV-P2c-4). Asserts stage handoffs, the S5 meta contract, export
    artifacts, and checkpoint resume.

    ``ac_resize`` is EXCLUDED: the smoke grid is deliberately under-resized
    (2 DC passes, 1 DCOPF pass), so the Newton flat-start diverges on every
    scenario and run_ac_resize collapse-returns the input state unchanged
    (DV-K exit semantics, D-7) — nothing to validate here. The legacy AC
    stages never ran on this vintage either (the dump's stage-8 crash,
    DV-P3-11); the AC pieces carry their own P3 unit/property tests.
    ``reactive`` stays — its winnowing loop degrades gracefully and proves
    the stage handoff."""
    cfg = _tmp_cfg(tmp_path)
    cfg.stages = ["build", "parameterize", "merge_topology", "scenarios",
                  "size_transformers", "dc_resize", "dcopf", "reactive",
                  "export"]
    cfg.loads.solver = "assignment"               # provably-equivalent MILP
    cfg.dispatch.load_scenarios = 4
    cfg.resize.max_passes = 2
    cfg.feasibility.max_passes = 1
    cfg.feasibility.max_workers = 1
    gw = GridWeld(cfg)
    model = gw.run()

    n = len(model.buses)
    assert model.buses["Num"].tolist() == list(range(1, n + 1))
    assert model.scenarios is not None
    assert model.scenarios.tiers["full"].shape == (n, 8)
    assert model.meta["load_scenarios"].shape == (n, 4)
    assert model.meta["pg_uc"].shape == (len(model.generators), 8)
    assert model.meta["pg"].shape == (len(model.generators), 8)   # OPF Pg
    assert "snet_full_opf" in model.meta
    assert (model.branches["RateA"].to_numpy(float) > 0).all()
    assert model.buses["SyncCond"].notna().all()

    assert "reactive" in model.meta

    art = model.meta["export_artifacts"]
    for key in ("final_case", "timeseries_mw", "Bus", "Branch", "PeakLoad",
                "Gen", "tl", "bus", "gen"):
        assert key in art and art[key].exists(), key
    assert "SwitchedShunts" not in art             # ac_resize not configured
    for stage in cfg.stages:
        assert gw.checkpoint_path(stage).exists(), stage

    # resume: re-export from the ac_resize checkpoint
    model2 = GridWeld(cfg).run(resume_from="export")
    assert model2.meta["export_artifacts"]["final_case"].exists()
