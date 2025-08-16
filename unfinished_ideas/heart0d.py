# heart0d.py
import numpy as np
from pressures import MMHG_TO_PA, atrial_elastance_SI, ventricular_elastance_SI

class Valves:
    def __init__(self, Rm_open=5e6, Rm_closed=5e9, Ra_open=5e6, Ra_closed=5e9, Rpv=2e6):
        self.Rm_o, self.Rm_c = Rm_open, Rm_closed
        self.Ra_o, self.Ra_c = Ra_open, Ra_closed
        self.Rpv = Rpv  # pulmonary venous inflow resistance

    def flow_mitral(self, P_la, P_lv):
        dP = P_la - P_lv
        R = self.Rm_o if dP >= 0 else self.Rm_c
        return dP / R

    def flow_aortic(self, P_lv, P_ao):
        dP = P_lv - P_ao
        R = self.Ra_o if dP >= 0 else self.Ra_c
        return dP / R

    def flow_pv(self, P_pv, P_la):
        return (P_pv - P_la) / self.Rpv

class AorticWindkessel2:
    def __init__(self, C=1.5e-3/MMHG_TO_PA, R_sys=1.0e7):
        # C in m^3/Pa (converted from ~1.5 ml/mmHg), R_sys in Pa·s/m^3
        self.C = C
        self.R = R_sys
        self.P = 80.0 * MMHG_TO_PA  # start ~80 mmHg

    def step(self, Q_in, dt):
        # Q_out = P/R ; C dP/dt = Q_in - Q_out
        Q_out = self.P / self.R
        dP = (Q_in - Q_out) / self.C
        self.P += dP * dt
        return Q_out

class LeftHeart0D:
    def __init__(self, V0_la=40e-6, V0_lv=120e-6,
                 P_pv_mmHg=12.0, valves=None, wk=None):
        self.V0_la, self.V0_lv = V0_la, V0_lv
        self.V_la = V0_la
        self.V_lv = V0_lv
        self.P_pv = P_pv_mmHg * MMHG_TO_PA
        self.valves = valves or Valves()
        self.wk = wk or AorticWindkessel2()

    def pressures_from_elastance(self, t, T):
        E_la = atrial_elastance_SI(t, T)
        E_lv = ventricular_elastance_SI(t, T)
        P_la = E_la * (self.V_la - self.V0_la)
        P_lv = E_lv * (self.V_lv - self.V0_lv)
        return P_la, P_lv

    def step(self, t, dt, T):
        # 1) Get chamber pressures from elastance
        P_la, P_lv = self.pressures_from_elastance(t, T)

        # 2) Valve flows
        Q_pv = self.valves.flow_pv(self.P_pv, P_la)      # pulmonary veins -> LA
        Q_m  = self.valves.flow_mitral(P_la, P_lv)       # LA -> LV
        Q_a  = self.valves.flow_aortic(P_lv, self.wk.P)  # LV -> Aorta

        # 3) Update volumes
        self.V_la += (Q_pv - Q_m) * dt
        self.V_lv += (Q_m  - Q_a) * dt

        # 4) Windkessel update (aortic pressure)
        _Qsys = self.wk.step(Q_a, dt)

        # Return the *new* chamber pressures (after vol update) for coupling
        P_la, P_lv = self.pressures_from_elastance(t+dt, T)
        return P_la, P_lv, {"Q_pv":Q_pv, "Q_m":Q_m, "Q_a":Q_a, "P_ao":self.wk.P}
