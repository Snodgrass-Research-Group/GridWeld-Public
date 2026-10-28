"""gridweld.io — case/data I/O at the external-id boundary (D6).

MATPOWER case bridge (``matpower.py``), one-time .mat→parquet converters
(``matlab_data.py``), PowerWorld import writers (``powerworld.py``), and
the GeoJSON round-trip (``geojson.py``). External MATPOWER ids appear ONLY
here; the CreateBranch.m int→ext shim becomes writer columns (doc 04 §11).

Phase: P2a (matpower, matlab_data) / P5 (powerworld, geojson).
"""
