"""gridweld.params — S3 electrical parameterization (algorithm doc 03).

Conductor selection from FERC Form 1 + tower geometry (``conductors.py``),
R/X/B with real GIS lengths (``impedance.py``), transformer sizing and
impedances (``transformers.py``). Heavy reuse of gridforge/gridcrucible
per the doc 03 reuse verdicts (D5).

Phase: P2b (impedance) / P3 (conductors, transformers).
"""
