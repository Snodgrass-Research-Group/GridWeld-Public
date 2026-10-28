"""Haversine snapping: point components and branch endpoints → nearest bus.

Origin: ``lat_lon_distance`` / ``node_assign`` / ``edge_assign`` /
``assign_generators!`` / ``assign_branches!`` in ``helper_functions.jl``
(algorithm doc 02 §snapping) and ``DistanceCalc.m`` (algorithm doc 03 —
same formula, R = 6371 km). ONE shared haversine for S2 snapping and S3
lengths.

Ported fixes (DV-P2c-3): correct >warn_km warnings (the legacy edge_assign
warning interpolated an undefined variable — UndefVarError if it ever
fired). Deterministic tie-breaks: candidate buses are scanned in the given
row order — callers pass frames sorted by ``index`` (Julia Dict-hash order
is unreproducible, DV-P2c-2).

⚠ Generators are snapped BEFORE added nodes join the bus table, so they
attach to substations only — the driver preserves that call order (doc 02
execution order step 4).

Units: degrees in, km out; warn thresholds km.
Phase: P2c. Validated vs p2/p3 julia fixtures (ids exact, distances 1e-9).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Chunk rows so the (chunk × n_bus) distance matrix stays ~150 MB.
_CHUNK = 2000


def haversine_km(lat1, lon1, lat2, lon2,
                 earth_radius_km: float = 6371.0) -> np.ndarray | float:
    """Vectorized haversine (broadcasting), R = cfg.geo.earth_radius_km —
    6371 km matches lat_lon_distance (Julia) and DistanceCalc.m (6371e3 m)
    to 1e-12. Args in degrees; result km."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return earth_radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_bus(lat: np.ndarray, lon: np.ndarray, buses: pd.DataFrame, *,
                earth_radius_km: float = 6371.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """(bus ``index`` id, distance km) of the nearest bus per point.

    First-occurrence argmin over the bus rows AS ORDERED — pass the bus
    frame sorted by ``index`` for deterministic tie-breaks (DV-P2c-2).
    """
    bus_lat = buses["Lat"].to_numpy(dtype=float)
    bus_lon = buses["Lon"].to_numpy(dtype=float)
    bus_ids = buses["index"].to_numpy(dtype=np.int64)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ids = np.empty(len(lat), dtype=np.int64)
    dist = np.empty(len(lat), dtype=float)
    for lo in range(0, len(lat), _CHUNK):
        hi = min(lo + _CHUNK, len(lat))
        d = haversine_km(lat[lo:hi, None], lon[lo:hi, None],
                         bus_lat[None, :], bus_lon[None, :], earth_radius_km)
        k = d.argmin(axis=1)
        ids[lo:hi] = bus_ids[k]
        dist[lo:hi] = d[np.arange(hi - lo), k]
    return ids, dist


def snap_points(comp_df: pd.DataFrame, buses: pd.DataFrame, comp_type: str,
                *, warn_km: float = 15.0,
                earth_radius_km: float = 6371.0) -> pd.DataFrame:
    """node_assign: adds ``<comp_type>_bus`` / ``<comp_type>_bus_distance``
    (km). Warns beyond ``warn_km`` (legacy ``distance_error`` = 15)."""
    out = comp_df.copy()
    ids, dist = nearest_bus(out["Lat"].to_numpy(), out["Lon"].to_numpy(),
                            buses, earth_radius_km=earth_radius_km)
    out[f"{comp_type}_bus"] = ids
    out[f"{comp_type}_bus_distance"] = dist
    for i in np.flatnonzero(dist > warn_km):
        log.warning("Closest bus to %s %s is %.3f km away",
                    comp_type, out["index"].iloc[i], dist[i])
    return out


def snap_branch_endpoints(branch_df: pd.DataFrame, buses: pd.DataFrame, *,
                          warn_km: float = 15.0,
                          earth_radius_km: float = 6371.0) -> pd.DataFrame:
    """edge_assign: independent nearest-bus snap of (Lat1,Lon1) → ``f_bus``
    and (Lat2,Lon2) → ``t_bus``, with distances (km)."""
    out = branch_df.copy()
    f_ids, f_dist = nearest_bus(out["Lat1"].to_numpy(), out["Lon1"].to_numpy(),
                                buses, earth_radius_km=earth_radius_km)
    t_ids, t_dist = nearest_bus(out["Lat2"].to_numpy(), out["Lon2"].to_numpy(),
                                buses, earth_radius_km=earth_radius_km)
    out["f_bus"] = f_ids
    out["f_bus_distance"] = f_dist
    out["t_bus"] = t_ids
    out["t_bus_distance"] = t_dist
    for name, dist in (("f_bus", f_dist), ("t_bus", t_dist)):
        for i in np.flatnonzero(dist > warn_km):
            log.warning("Closest %s to branch %s is %.3f km away",
                        name, out["index"].iloc[i], dist[i])
    return out
