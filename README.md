# pumpy
A digital heart that never skips a beat

## Installation

## Activation
'conda activate fenicsx'





How this gives you a full (left) heart cycle

LA receives blood from pulmonary veins (constant ~12 mmHg source through a small resistance), LV ejects into an aortic Windkessel that shapes afterload.

Valves open/close automatically via the resistance switch (open vs closed) based on pressure differences (ΔP).

Coupling: each time step, the 0D model proposes pressures; 3D solves and returns volumes; 0D updates its state, and you iterate once or twice for consistency.

Outputs: time series of P_LA, P_LV, P_AO, V_LA, V_LV, Q_mitral, Q_aortic → you can plot PV-loops and flow curves immediately.

What you’ll need to set for your meshes

la_bc, la_endo, la_pfacets and lv_bc, lv_endo, lv_pfacets must match your facet tags:

bc_tags: fixed boundary facets,

endocardial_tags: inner surface of chamber for volume integral,

pressure_tags: where pressure traction is applied (usually same as endocardium).

If your LA model uses two inlet patches (e.g., 20 and 50), keep both in *_pfacets and *_endo.

Tuning knobs (and what they do)

Valve resistances (Rm_open/Rm_closed, Ra_open/Ra_closed) → regulates when/how much each valve flows/leaks.

Aortic C, R in AorticWindkessel2 → shapes aortic pressure and dicrotic decay.

Elastance ranges (already in pressures.py) → control peak pressures and timing.

picard_iters (usually 2–3) → stronger 0D–3D consistency; raise if you see drift between mechanical and 0D volumes.

Nice immediate add-ons (still “code-first”)

Log and plot PV-loop for LV: (V_lv, P_lv).

Export VTX/XDMF each step for ParaView; keep GIF optional.

If you want the right heart later: duplicate the pattern (RA↔RV) with pulmonary Windkessel and tricuspid/pulmonic valves; then connect the two 0D circuits (systemic ↔ pulmonary).

If you want, I can fold these files into your current tree and add a tiny PV-loop plotting script, but this is already everything you need to run a full left-heart cycle end-to-end with your existing mechanics.





BCS:
LV (Left Ventricle)

Cell tag (info only): 1 = myocardium

Outer surface: 10 = epicardium → free (no pressure, no Dirichlet)

Inner surface: 20 = endocardium → pressure & volume integral

Base / annulus: 50 = basal plane → Dirichlet anchor (fixed)

Use:

lv_bc = (50,)

lv_endo = (20,)

lv_pfacets = (20,)

LA (Left Atrium)

Cell tag (info only): 1 = myocardium

Inner wall: 10 = endocardium → pressure & volume integral

Pulmonary veins: 20 = PV Right, 50 = PV Left → Dirichlet anchors (rings)

Outer surface: 30 = epicardium → free

Mitral annulus: 40 = mitral_valve_opening → Dirichlet anchor (ring)

Use:

la_bc = (40) (pin PV rings + MV ring; this is common and avoids over-constraining the wall)

la_endo = (10,)

la_pfacets = (10,)