"""gridweld.power — network core: incidence/Y-bus, DC and AC power flow.

Thin wrappers over gridforge ``scale.power_flow`` + ``network.impedance``
(D5, doc 04 reuse verdicts) with GridWeld-specific extensions: exact-degree
angles (Q9), single +1/X sign convention (DV-B), explicit bus-type lists and
PV→PQ switching for the reactive stage (DV-C), stats tables.

Phase: P2b.
"""
