"""GIS backend protocol + shared parameter types.

Origin: the common surface of the EV-repo cleaners (algorithm doc 01 —
Feeder_Cleanser_and_CSV_File_Generator.py, Method1.py, Sofia's notebook),
abstracted so ``backend="arcpy" | "open"`` is a config switch (D4/D8).

Phase: P2d (interface lands with the arcpy backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

#: A backend-opaque handle to a feature layer (arcpy: catalog path string;
#: open backend: a GeoDataFrame wrapper). Kept loose until P2d.
LayerRef = Any


@dataclass(frozen=True)
class CleanParams:
    """One clean run's tolerances (doc 01 knobs table). Units: metres.
    Values come from ``WeldConfig.gis.clean.<distribution|transmission>``."""

    feature_to_line_tol_m: float
    generalize_tol_m: float          # 0.0 = skip Generalize (transmission)
    extend_line_m: float
    snap_edge_m: float
    attribute_map_tol_m: float


@dataclass(frozen=True)
class JoinParams:
    """Attribute-transfer parameters (Ruiz midpoint join / SpatialJoin).
    ``tol_m`` explicit — never the shadowed module global (PD-10)."""

    tol_m: float
    unmatched_value: int = -1


class GISCleanBackend(Protocol):
    """What every GIS backend must provide (doc 01 §Feeder_Cleanser design)."""

    def clean_lines(self, layer: LayerRef, p: CleanParams) -> LayerRef:
        """FeatureToLine → (Generalize) → ExtendLine → Snap → SplitLine →
        FeatureToLine sequence with the given tolerances."""
        ...

    def extract_endpoints(self, layer: LayerRef) -> tuple[Any, Any]:
        """START and END vertex point tables (WGS84 lon/lat), for node dedup."""
        ...

    def transfer_attributes(self, target: LayerRef, source: LayerRef,
                            p: JoinParams) -> None:
        """Nearest-neighbor attribute transfer (midpoint-based, doc 01 §Method1)."""
        ...
