"""GridWeld: real-GIS synthetic test-system builder.

Python consolidation of the CATS (California Test System) pipeline —
ArcGIS/ArcPy GIS cleaning (S1), Julia network build (S2), MATLAB
parameterization + dispatch + resizing (S3/S4), MATLAB DCOPF feasibility
(S5), and exports (S7) — into one package. Evaluation (S6) stays in Julia
(`julia/GridWeldEval/`, decision D3).

Completes the family: gridforge (topology), gridcrucible (KCL-KVL),
gridtemper (TX OPF feasibility), gridweld (real-GIS test systems).

Labels such as D5 (design decision), P3 (conversion phase) and DV-P3-2
(deviation from the MATLAB code) refer to the conversion notes, which are
not part of this package.
"""

from .config import STAGES, WeldConfig
from .model import WeldModel

__version__ = "0.1.0"
__all__ = ["WeldConfig", "WeldModel", "STAGES"]
