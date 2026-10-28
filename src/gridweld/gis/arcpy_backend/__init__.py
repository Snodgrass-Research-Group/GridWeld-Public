"""arcpy backend for gridweld.gis — runs under ArcGIS Pro's Python (propy).

Port of the EV-repo cleaners per algorithm doc 01: the batched clean
(``clean.py``), Sofia's transmission-specific steps (``transmission.py``),
the midpoint attribute transfer (``spatial_join.py``), the geometric island
fix (``islands_geom.py``), and the parallel driver (``driver.py``).

``import arcpy`` happens ONLY inside these modules (D4): everything else in
gridweld runs without ArcGIS.

Phase: P2d.
"""
