"""gridweld.gis — S1 GIS cleaning (algorithm doc 01).

Backend-swappable design (D4): ``interface.py`` defines the Protocol;
``arcpy_backend/`` is the P2d port (runs under ArcGIS Pro's ``propy``);
``open_backend/`` is the P8 stretch (geopandas/shapely/networkx). The
portable pieces — ``nodes.py`` (dedup → bus numbering) and ``islands.py``
(components + stitching) — are pure numpy/pandas and shared by both.

arcpy imports live ONLY inside ``arcpy_backend/`` so the rest of gridweld
never needs ArcGIS.
"""
