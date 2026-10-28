"""Reactive power placement: condenser winnowing.

Origin: ``ReactivePowerPlacement.m`` (+ ``PVPQSwitching.m`` from
Aditya_Code) — algorithm doc 04 §9. No MATLAB fixture exists: the dump's
stage 8 CRASHED upstream ("Index in position 2 exceeds array bounds. Index
must not exceed 36") — root cause CONFIRMED during P3: the magic
``1:42`` max-load search window (``ed_scenario_count``) indexes
``PF.SNetComplete``, which the CATS run deduplicates to **36** columns
(s5_pf.mat). The port clamps the window to the matrix width with a warning
(DV-P3-11) instead of crashing.

Structure (doc 04 §9, the three copy-paste solve blocks collapsed into
one helper as gridcrucible did):
1. Cap line charging ``B(B>1) = 1`` (series-compensation stand-in).
2. Power-factor correction of the load scenarios (bands 0.9→0.97,
   0.95→0.98, 0.96→0.99; Q rebuilt from P).
3. Max-|P| scenario within the (clamped) ED window duplicated as COLUMN 0
   (the AC loop drops it later).
4. Bus typing: slack = PF slack (argmax MaxMW); every other bus PV at
   1.0 pu (``reactive.gen_vset`` — supersedes the DC-era 1.02).
5. Winnowing: per iteration demote the ceil(per_switch·n) smallest-|Q|
   condenser candidates to PQ; non-convergence → revert + per_switch ×=
   decay (floor per_switch_floor ends the loop); |V| outside v_place_band →
   PERMANENT condenser; Q-limit / band violations → PV→PQ clamp (the
   PVPQSwitching two-fixed-passes logic); stop when candidates + placed <
   support_fraction·n.

Reuse (D5): the winnowing schedule/constants mirror gridcrucible
``network.shunts.place_reactive_compensation``; Q banks are CONFIG-driven
with MATLAB defaults ±10000 MVAr (DV-E — gridcrucible hard-codes
800/−1200). Solves run on gridweld's own ACEngine (P2b) — caller-owned bus
typing is exactly what this stage needs.

Units: Q MVar; V pu; fractions dimensionless.
Phase: P3 (no fixture — property/unit tests only; doc 04 validation tier).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..config import WeldConfig
from ..model import WeldModel
from ..power.ac import ACEngine

log = logging.getLogger(__name__)


def correct_power_factors(snet: np.ndarray) -> np.ndarray:
    """Doc 04 §9 step 2: sequential band re-assignment (net effect):
    PF ≤ 0.9 → 0.97; (0.9, 0.95] → 0.98; (0.95, 0.96] → 0.99;
    Q rebuilt as |P|·√(1/pf² − 1)·sign(Q)."""
    out = snet.copy()
    p, q = out.real, out.imag
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.abs(p) / np.abs(out)
    pf = np.where(np.isfinite(pf), pf, 1.0)
    new_pf = np.where(pf <= 0.9, 0.97,
                      np.where(pf <= 0.95, 0.98,
                               np.where(pf <= 0.96, 0.99, pf)))
    changed = new_pf != pf
    q_new = np.where(changed,
                     np.abs(p) * np.sqrt(1.0 / new_pf ** 2 - 1.0)
                     * np.where(q < 0, -1.0, 1.0), q)
    return p + 1j * q_new


def duplicate_max_scenario(snet: np.ndarray, cfg: WeldConfig) -> np.ndarray:
    """Doc 04 §9 step 3: argmax total |P| within the first
    ``ed_scenario_count`` columns — CLAMPED to the width (DV-P3-11; the
    legacy 1:42 crashed on SNetComplete's 36 columns) — duplicated as
    column 0."""
    window = min(cfg.reactive.ed_scenario_count, snet.shape[1])
    if window < cfg.reactive.ed_scenario_count:
        log.warning("ed_scenario_count window %d clamped to SNet width %d "
                    "(the legacy 1:42 crashed here — DV-P3-11)",
                    cfg.reactive.ed_scenario_count, snet.shape[1])
    best = int(np.abs(snet.real[:, :window]).sum(axis=0).argmax())
    return np.concatenate([snet[:, [best]], snet], axis=1)


def place_reactive_support(model: WeldModel, snet: np.ndarray, slack: int,
                           cfg: WeldConfig) -> dict:
    """The winnowing loop (doc 04 §9 steps 4-6) on ACEngine solves.

    ``snet``: complex net injections MW+jMVar (bus × scenario), column 0 =
    the duplicated max-load scenario used for winnowing solves; ``slack``:
    0-based slack bus. Returns {'sync_cond': bool array, 'pv': ids,
    'pq': ids, 'converged': bool, 'iterations': int, 'v_mag': array}.
    """
    n = model.n_buses
    branches = model.branches.copy()
    b = branches["b_pu"].to_numpy(float).copy()
    b[b > cfg.reactive.b_charging_cap] = cfg.reactive.b_charging_cap
    branches["b_pu"] = b
    work = WeldModel(buses=model.buses, branches=branches,
                     generators=model.generators, meta=dict(model.meta))

    snet = correct_power_factors(snet)
    col0 = snet[:, 0]

    gen_buses = np.unique(model.generators["bus"].to_numpy(np.int64))
    candidates = np.setdiff1d(np.arange(n), np.concatenate([gen_buses,
                                                            [slack]]))
    permanent = np.zeros(n, dtype=bool)
    is_pv = np.ones(n, dtype=bool)            # start: EVERYONE PV
    is_pv[slack] = False
    per_switch = cfg.reactive.per_switch
    vset = np.full(n, cfg.reactive.gen_vset)
    eng = ACEngine(work, s_base=cfg.system.s_base_mva)
    v0 = None
    result = None
    it = 0
    v_lo, v_hi = cfg.reactive.v_place_band

    while per_switch >= cfg.reactive.per_switch_floor:
        it += 1
        pv = np.flatnonzero(is_pv)
        pq = np.setdiff1d(np.arange(n), np.concatenate([pv, [slack]]))
        res = eng.solve(col0, slack, pv, pq, v0=v0, v_setpoints=vset)
        if not res.converged:
            per_switch *= cfg.reactive.per_switch_decay
            v0 = None                          # cold restart, smaller step
            continue
        result = res
        v0 = res.v_mag_pu * np.exp(1j * np.radians(res.v_angle_deg))

        q = res.snet_calc_mva.imag
        over_q = (q > cfg.reactive.high_q_mvar) | (q < cfg.reactive.low_q_mvar)
        if over_q.any():
            is_pv[over_q & ~permanent
                  & np.isin(np.arange(n), candidates)] = False

        vm = res.v_mag_pu
        newly_perm = ((vm >= v_hi) | (vm <= v_lo)) \
            & np.isin(np.arange(n), candidates)
        permanent |= newly_perm
        is_pv[newly_perm] = True

        active = np.flatnonzero(is_pv & np.isin(np.arange(n), candidates)
                                & ~permanent)
        if (len(active) + permanent.sum()
                < cfg.reactive.support_fraction * n):
            break
        k = min(math.ceil(per_switch * n), len(active))
        if k == 0:
            break
        demote = active[np.argsort(np.abs(q[active]))[:k]]
        is_pv[demote] = False

    permanent |= is_pv & np.isin(np.arange(n), candidates)
    is_pv[permanent] = True
    pv_final = np.flatnonzero(is_pv | np.isin(np.arange(n), gen_buses))
    pv_final = np.setdiff1d(pv_final, [slack])
    return {
        "sync_cond": permanent,
        "pv": pv_final,
        "pq": np.setdiff1d(np.arange(n), np.concatenate([pv_final, [slack]])),
        "converged": result is not None and result.converged,
        "iterations": it,
        "v_mag": None if result is None else result.v_mag_pu,
    }
