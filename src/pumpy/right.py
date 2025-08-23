from __future__ import annotations
import numpy as np

MMHG_TO_PA = 133.322368
PA_TO_MMHG = 1.0 / MMHG_TO_PA

class Valve:
    """Soft valve with opening fraction s in [0,1] driven by ΔP."""
    def __init__(self, R: float, tau_open: float=0.02, tau_close: float=0.04, slope_mmHg: float=1.0):
        self.R = R
        self.s = 0.0
        self.tau_open = tau_open
        self.tau_close = tau_close
        self.k = slope_mmHg * MMHG_TO_PA  # steepness vs ΔP (Pa)

    def step(self, dt: float, dp: float) -> float:
        target = 1.0 / (1.0 + np.exp(-dp / self.k))  # sigmoid in ΔP
        tau = self.tau_open if target > self.s else self.tau_close
        self.s += (target - self.s) * (dt / max(tau, 1e-6))
        self.s = float(np.clip(self.s, 0.0, 1.0))
        # flow through valve
        return self.s * max(dp, 0.0) / self.R

class WK2:
    """Windkessel (R, C) with upstream inflow Qin and distal resistance Rout to venous reservoir Pv."""
    def __init__(self, R: float, C: float, P0_mmHg: float, Rout: float, Pv_mmHg: float):
        self.R = R
        self.C = C
        self.P = P0_mmHg * MMHG_TO_PA     # arterial pressure (Pa)
        self.Rout = Rout
        self.Pv = Pv_mmHg * MMHG_TO_PA    # venous/return pressure (Pa)

    def step(self, dt: float, Qin: float):
        # C * dP/dt = Qin - (P - Pv)/Rout - P/R (out to "ground" can be included in Rout lump)
        # Here keep outflow purely via Rout to Pv (systemic/pulmonary veins).
        dP = (Qin - (self.P - self.Pv) / self.Rout) / self.C * dt
        self.P = max(self.P + dP, 0.0)

class ChamberLogger:
    """Tiny volume/pressure logger for RA/RV mirrored from your LA/LV loggers."""
    def __init__(self, V0_mL: float, R_in: float, R_out: float):
        self.V_m3 = V0_mL * 1e-6
        self.R_in = R_in
        self.R_out = R_out
        self.log = {"t": [], "P_Pa": [], "V_m3": []}

    def step(self, t, P_in, P_self, P_out, valve_in_open, valve_out_open, dt):
        Q_in  = ((P_in - P_self) / self.R_in)   if valve_in_open  else 0.0
        Q_out = ((P_self - P_out) / self.R_out) if valve_out_open else 0.0
        self.V_m3 = max(self.V_m3 + (Q_in - Q_out) * dt, 0.0)
        self.log["t"].append(t); self.log["P_Pa"].append(P_self); self.log["V_m3"].append(self.V_m3)
