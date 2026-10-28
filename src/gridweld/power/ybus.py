"""Incidence + admittance assembly (thin gridforge re-export).

Origin: ``BuildANet.m``, ``YBusCalc.m``, ``YBusCalcDC.m`` (algorithm doc 04
§1-§2, both REUSE-AS-IS). gridforge ``network.impedance.ybus`` reproduces
the primitive assembly, contingency handling, and line-charging shunts, and
its explicit ``shunt_b_pu`` argument replaces the ShuntMVAR.mat side channel
(DV-A — the legacy step-6 diagonal was a latent dimension-mismatch crash,
dead only because UseReactiveSupport=0).

Conventions: 0-based ids; EXPLICIT n_buses (the legacy max-id inference —
quirk Q-a — is not reproduced); the DC primitive is REAL +1/X everywhere
(DV-B — YBusCalcDC's convention; DCPowerFlowFull's −1/X flip is not
reproduced).

Phase: P2b. Validated against s4_anet_triplets.csv / s4_bbus_triplets.csv.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from gridforge.network.impedance import YBusSet
from gridforge.network.impedance import ybus as _gridforge_ybus


def incidence(from_bus: np.ndarray, to_bus: np.ndarray,
              n_buses: int) -> sp.csc_matrix:
    """Signed incidence (+1 from, −1 to), n_buses × n_branches (BuildANet.m)."""
    from_bus = np.asarray(from_bus, dtype=np.int64)
    to_bus = np.asarray(to_bus, dtype=np.int64)
    m = len(from_bus)
    rows = np.concatenate([from_bus, to_bus])
    cols = np.tile(np.arange(m, dtype=np.int64), 2)
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    return sp.csc_matrix((vals, (rows, cols)), shape=(n_buses, m))


def ybus(from_bus: np.ndarray, to_bus: np.ndarray, r_pu: np.ndarray,
         x_pu: np.ndarray, b_pu: np.ndarray, n_buses: int, *,
         shunt_b_pu: np.ndarray | None = None,
         contingency: int | None = None) -> YBusSet:
    """YBusCalc.m equivalent → gridforge ``YBusSet`` (y_short/y_long/
    b_short/b_long/y_prim/b_prim/a_con)."""
    a = incidence(from_bus, to_bus, n_buses)
    return _gridforge_ybus(a, np.asarray(from_bus, dtype=np.int64),
                           np.asarray(to_bus, dtype=np.int64),
                           r_pu, x_pu, b_pu,
                           shunt_b_pu=shunt_b_pu, contingency=contingency)


def b_matrix(from_bus: np.ndarray, to_bus: np.ndarray, x_pu: np.ndarray,
             n_buses: int, contingency: int | None = None) -> sp.csc_matrix:
    """YBusCalcDC.m: ``B = A·diag(1/X)·Aᵀ`` with the REAL +1/X primitive."""
    b_prim = 1.0 / np.asarray(x_pu, dtype=np.float64)
    if contingency is not None:
        b_prim = b_prim.copy()
        b_prim[contingency] = 0.0
    a = incidence(from_bus, to_bus, n_buses)
    return (a @ sp.diags(b_prim) @ a.T).tocsc()
