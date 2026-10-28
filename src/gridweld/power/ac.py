"""AC power flow: Newton solves with caller-owned bus typing + Y-bus caching.

Origin: ``ACPowerFlowSimple.m`` / ``ACPowerFlowFinal.m`` — algorithm doc 04
§5-§6. ⚠ The Newton kernel ``pfsolve.m`` exists in NEITHER repo (DV-C): its
tolerances are unrecoverable; gridforge's textbook full-Newton constants
(tol=1e-8 max-mismatch, max_iter=30) are the reference, validated against
pandapower rather than MATLAB fixtures.

Reuse (D5): gridforge ``scale.power_flow.ACPowerFlow`` supplies the Newton
pieces — its static ``_mismatch``/``_jacobian`` are consumed directly so the
kernel is byte-identical; ``gridforge.network.impedance.ybus`` supplies the
operators. This engine adds what doc 04 requires and gridforge lacks:
explicit slack/pv/pq lists (the reactive stage re-types buses every
iteration), warm starts, Y-bus caching across scenario sweeps
(ACPowerFlowFinal's ReCalcYBus/YBusOld plumbing), and SNetCalc bus
injections + loss stats on the result. The dead AddReactiveSupport path
(quirk Q-f: undefined functions + invalid save call) is not ported (DV-A).

Flows use the series-only primitive (charging injection excluded at the
terminals — consistent with MATLAB; changing this would skew send/receive
asymmetry). PV→PQ Q-limit switching (PVPQSwitching.m, Aditya_Code) lands
with the P3 reactive stage.

Units: V pu, angles deg; flows MVA (max of |send|,|rec|); injections
MW+jMVar (net INJECTION convention; flip upstream per MiddleLoopAC's
positive-load guard — units are explicit here, the Q-i ×20000 sniff is
dropped).

Phase: P2b (kernel + engine) / P3 (Q-limit switching).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from gridforge.network.impedance import ybus
from gridforge.scale.power_flow import ACPowerFlow

from ..model import PowerFlowResult, SingularNetworkError, WeldModel


class ACEngine:
    """Cached-Y-bus AC solver (replaces the ACPF struct's ReCalcYBus/YBusOld
    plumbing): invalidate on impedance change, warm-start between scenarios."""

    def __init__(self, model: WeldModel, *, s_base: float = 100.0,
                 long_line: bool = True, tol: float = 1e-8,
                 max_iter: int = 30) -> None:
        self.model = model
        self.s_base = s_base
        self.long_line = long_line
        self.tol = tol
        self.max_iter = max_iter
        self._ybus_set = None

    def invalidate(self) -> None:
        """Drop the cached Y-bus (call after any impedance change)."""
        self._ybus_set = None

    def _operators(self):
        if self._ybus_set is None:
            br = self.model.branches
            a = self.model.incidence().tocsc()
            self._ybus_set = ybus(
                a, br["from_bus"].to_numpy(np.int64),
                br["to_bus"].to_numpy(np.int64),
                br["r_pu"].to_numpy(float), br["x_pu"].to_numpy(float),
                br["b_pu"].to_numpy(float))
            self._a = a
        return self._a, self._ybus_set

    def solve(self, snet_mva: np.ndarray, slack: int, pv: np.ndarray,
              pq: np.ndarray, v0: np.ndarray | None = None, *,
              v_setpoints: np.ndarray | None = None) -> PowerFlowResult:
        """One Newton solve with caller-owned bus typing.

        ``snet_mva``: complex net injections, MW+jMVar. ``pv``/``pq``:
        0-based bus id arrays partitioning the non-slack buses. ``v0``:
        warm-start complex voltages (default: setpoint magnitude, 0°).
        """
        model = self.model
        n = model.n_buses
        a, ys = self._operators()
        y = (ys.y_long if self.long_line else ys.y_short).tocsr()

        pv = np.asarray(pv, dtype=np.int64)
        pq = np.asarray(pq, dtype=np.int64)
        pvpq = np.concatenate([pv, pq])
        if len(pvpq) + 1 != n or len(np.unique(np.concatenate([pvpq, [slack]]))) != n:
            raise ValueError("slack/pv/pq must partition all buses "
                             "(PSERCMain step 11 assertion)")

        s_spec = np.asarray(snet_mva, dtype=np.complex128) / self.s_base
        vset = (np.ones(n) if v_setpoints is None
                else np.asarray(v_setpoints, dtype=float))
        v = (vset.astype(np.complex128) if v0 is None
             else np.asarray(v0, dtype=np.complex128).copy())
        # PV/slack magnitudes pinned to setpoints (pfsolve contract)
        v[pv] = vset[pv] * v[pv] / np.abs(v[pv])
        v[slack] = vset[slack] * v[slack] / np.abs(v[slack])

        converged, it = False, 0
        for it in range(1, self.max_iter + 1):
            mism = ACPowerFlow._mismatch(y, v, s_spec, pvpq, pq)
            if np.max(np.abs(mism)) < self.tol:
                converged = True
                break
            j = ACPowerFlow._jacobian(y, v, pvpq, pq)
            try:
                dx = spla.spsolve(j.tocsc(), -mism)
            except RuntimeError as exc:
                raise SingularNetworkError(f"AC Jacobian singular: {exc}") from exc
            n_a = len(pvpq)
            va = np.angle(v)
            vm = np.abs(v)
            va[pvpq] += dx[:n_a]
            vm[pq] += dx[n_a:]
            v = vm * np.exp(1j * va)

        br = model.branches
        from_bus = br["from_bus"].to_numpy(np.int64)
        to_bus = br["to_bus"].to_numpy(np.int64)
        vdrop = a.T @ v
        i_line = ys.y_prim @ vdrop
        s_send = self.s_base * v[from_bus] * np.conj(i_line)
        s_rec = self.s_base * v[to_bus] * np.conj(i_line)
        va_deg = np.degrees(np.angle(v))
        snet_calc = self.s_base * v * np.conj(y @ v)   # SNetCalc [§6 step 5]
        return PowerFlowResult(
            converged=converged,
            v_mag_pu=np.abs(v),
            v_angle_deg=va_deg,
            flow_mva=np.maximum(np.abs(s_send), np.abs(s_rec)),
            angle_diff_deg=np.abs(a.T @ va_deg),
            snet_calc_mva=snet_calc,
            losses_mw=float(np.sum(np.real(s_send - s_rec))),
            iterations=it,
        )


def solve_ac(model: WeldModel, snet_mva: np.ndarray, slack: int,
             pv: np.ndarray, pq: np.ndarray, *,
             v_setpoints: np.ndarray | None = None, s_base: float = 100.0,
             long_line: bool = True) -> PowerFlowResult:
    """One-shot ACPowerFlowSimple.m equivalent (fresh Y-bus each call —
    use :class:`ACEngine` inside loops)."""
    eng = ACEngine(model, s_base=s_base, long_line=long_line)
    return eng.solve(snet_mva, slack, pv, pq, v_setpoints=v_setpoints)


def check_connected(model: WeldModel) -> None:
    """Exact connectivity screen (DV-I): raises SingularNetworkError listing
    component sizes when the bus-branch graph is disconnected."""
    from scipy.sparse.csgraph import connected_components
    br = model.branches
    n = model.n_buses
    adj = sp.csr_matrix(
        (np.ones(len(br)), (br["from_bus"].to_numpy(np.int64),
                            br["to_bus"].to_numpy(np.int64))), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp > 1:
        sizes = sorted(np.bincount(labels), reverse=True)
        raise SingularNetworkError(
            f"network has {n_comp} components (sizes {sizes[:5]}...) — "
            "legacy screens (condest/sprank) replaced by exact check (DV-I)")
