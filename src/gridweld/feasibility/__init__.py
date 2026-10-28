"""gridweld.feasibility — S5 DCOPF feasibility upgrade (algorithm doc 05).

Aditya Rangarajan's stage: replace the plain DC power flow with a DCOPF
that re-dispatches generation (L1-closest to UCED setpoints), so the final
case is OPF-feasible. DV1: YALMIP+Mosek → gurobipy.

Phase: P4.
"""
