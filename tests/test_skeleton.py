"""P1 skeleton gate: every module imports, config round-trips, model basics.

Deliberately does NOT test behavior — stubs raise NotImplementedError with
their phase tag until the owning phase lands.
"""

from __future__ import annotations

import importlib
import pkgutil

import pandas as pd
import pytest

import gridweld
from gridweld.config import REPO_ROOT, STAGES, WeldConfig
from gridweld.model import WeldModel

CATS_TOML = REPO_ROOT / "configs" / "cats.toml"


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------

def _all_modules() -> list[str]:
    names = ["gridweld"]
    for m in pkgutil.walk_packages(gridweld.__path__, prefix="gridweld."):
        names.append(m.name)
    return names


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_reused_packages_import() -> None:
    """D5 path deps resolve (external/ populated by scripts/bootstrap_deps)."""
    import gridcrucible  # noqa: F401
    import gridforge  # noqa: F401


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_cats_toml_is_noop_over_defaults() -> None:
    """WeldConfig defaults ARE the CATS values (config.py contract), so
    loading configs/cats.toml must reproduce a bare WeldConfig()."""
    assert WeldConfig.from_toml(CATS_TOML) == WeldConfig()


def test_config_dict_round_trip() -> None:
    cfg = WeldConfig.from_toml(CATS_TOML)
    assert WeldConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="unknown config section"):
        WeldConfig.from_dict({"nonsense": {"x": 1}})


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        WeldConfig.from_dict({"resize": {"percent_resiez": 0.05}})


def test_config_rejects_bad_stage_order() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        WeldConfig.from_dict({"stages": ["build", "gis_clean"]})


def test_stages_cover_pipeline() -> None:
    assert STAGES[0] == "gis_clean"
    assert STAGES[-1] == "export"
    assert len(STAGES) == len(set(STAGES))


def test_paths_resolve_against_root() -> None:
    cfg = WeldConfig()
    resolved = cfg.paths.resolve(cfg.paths.substations)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(cfg.paths.root))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def test_empty_model_validates() -> None:
    assert WeldModel().validate() == []


def test_incidence_and_masks() -> None:
    model = WeldModel(
        buses=pd.DataFrame({"bus_id": [0, 1, 2]}),
        branches=pd.DataFrame({
            "from_bus": [0, 1],
            "to_bus": [1, 2],
            "tap": [0.0, 1.0],
        }),
    )
    a = model.incidence()
    assert a.shape == (3, 2)
    dense = a.toarray()
    assert dense[0, 0] == 1 and dense[1, 0] == -1        # branch 0: 0 → 1
    assert dense[1, 1] == 1 and dense[2, 1] == -1        # branch 1: 1 → 2
    assert model.line_mask().tolist() == [True, False]   # tap == 0 ⇔ line
    assert model.transformer_mask().tolist() == [False, True]


def test_model_flags_out_of_range_endpoints() -> None:
    model = WeldModel(
        buses=pd.DataFrame({"bus_id": [0, 1]}),
        branches=pd.DataFrame({"from_bus": [0], "to_bus": [5], "tap": [0.0]}),
    )
    assert any("out of range" in p for p in model.validate())


# P0–P5 stubs are all implemented — the phase-tag test retired with them
# (P5). The S6 evaluation module is Julia-side (no Python stub).
