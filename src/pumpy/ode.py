from __future__ import annotations
import math
import numpy as np
from .utils import MMHG_TO_PA, PA_TO_MMHG, mmHg_per_mL, en_double_hill, smooth_cos_window, atrial_pressure_pa
try:
    from .mesh import volume_from_mesh  # [m^3]
except Exception:
    volume_from_mesh = None 


class EPDriver:
    """Minimal activator; once-per-beat square pulse -> [0,1] activation."""
    def __init__(self, dt: float, hr_bpm: float):
        self.dt = float(dt)
        self.period = 60.0 / max(hr_bpm, 1e-6)
        self.t = 0.0
        self.v = 0.0
        self.v_gate = 0.13
        self.tau_open = 120.0
        self.tau_close = 150.0
        self.T_out = 6.0
        self.a = 0.0

    def _stim(self, t: float) -> float:
        # 2 ms pulse each beat
        return 1.0 if (t % self.period) < 0.002 else 0.0

    def step(self) -> float:
        I = self._stim(self.t)
        v = self.v
        dv = (I - v * (v - self.v_gate) * (1 - v) - v / self.T_out) * self.dt
        self.v = float(np.clip(v + dv, 0.0, 1.0))
        self.a = 0.0 if self.v < self.v_gate else (self.v - self.v_gate) / (1 - self.v_gate)
        self.a = float(np.clip(self.a, 0.0, 1.0))
        self.t += self.dt
        return self.a


class SoftValve:
    """
    Smooth valve opening fraction s∈[0,1] with different open/close time constants.
    Flow model: Q = s * max(ΔP, 0) / R.
    """
    def __init__(
        self,
        R: float,
        tau_open: float = 0.02,
        tau_close: float = 0.04,
        slope_mmHg: float = 1.0,
    ):
        self.R = float(R)
        self.s = 0.0
        self.tau_open = float(tau_open)
        self.tau_close = float(tau_close)
        self.k = float(slope_mmHg) * MMHG_TO_PA  # Pa

    def step(self, dt: float, dp: float) -> tuple[float, float]:
        target = 1.0 / (1.0 + np.exp(-dp / max(self.k, 1e-12)))
        tau = self.tau_open if target > self.s else self.tau_close
        self.s += (target - self.s) * (dt / max(tau, 1e-12))
        self.s = float(np.clip(self.s, 0.0, 1.0))
        Q = self.s * max(dp, 0.0) / self.R
        return self.s, Q


class AorticWK2:
    """Classic 2-element Windkessel (R, C) with Qin as inflow and leak to ground is implicit in Rout via the system."""
    def __init__(self, R: float = 1.5e8, C: float = 1.3e-8, P0_mmHg: float = 80.0,Pv_mmHg: float = 5.0):
        self.R = float(R)
        self.C = float(C)
        self.P = P0_mmHg * MMHG_TO_PA  # Pa
        self.Pv = Pv_mmHg * MMHG_TO_PA

    def step(self, dt: float, Qin: float):
        # C dP/dt = Qin - (P - Pv)/R
        dP = (Qin - (self.P - self.Pv) / self.R) / self.C * dt
        self.P = max(self.P + dP, 0.0)


class PulmVenousSource:
    """Pulmonary venous pressure baseline, optionally modulated by respiration."""
    def __init__(self, P0_mmHg: float = 12.0, resp_mmHg: float = 0.0, resp_hz: float = 0.25):
        self.P0 = float(P0_mmHg) * MMHG_TO_PA
        self.resp_A = float(resp_mmHg) * MMHG_TO_PA
        self.resp_om = 2.0 * math.pi * float(resp_hz)

    def pressure(self, t: float) -> float:
        return self.P0 + (self.resp_A * math.sin(self.resp_om * t) if self.resp_A != 0.0 else 0.0)


class VentricularVolumeLogger:
    def __init__(self, V0_mL: float = 120.0, R_mitral: float = 2.0e8, R_aortic: float = 2.0e8):
        self.V_m3 = float(V0_mL) * 1e-6
        self.R_mitral = float(R_mitral)
        self.R_aortic = float(R_aortic)
        self.log = {"t": [], "P_Pa": [], "V_m3": []}

    def step(self, t, P_LA, P_LV, P_AO, mitral_open, aortic_open, dt):
        Q_in = ((P_LA - P_LV) / self.R_mitral) if mitral_open else 0.0
        Q_out = ((P_LV - P_AO) / self.R_aortic) if aortic_open else 0.0
        self.V_m3 = max(self.V_m3 + (Q_in - Q_out) * dt, 0.0)
        self.log["t"].append(t)
        self.log["P_Pa"].append(P_LV)
        self.log["V_m3"].append(self.V_m3)

    def export_csv(self, path="pv_lv.csv"):
        import pandas as pd
        pd.DataFrame(self.log).to_csv(path, index=False)
        return path


class AtrialVolumeLogger:
    def __init__(self, V0_mL: float = 60.0, R_pv: float = 2.0e8, R_mitral: float = 2.0e8):
        self.V_m3 = float(V0_mL) * 1e-6
        self.R_pv = float(R_pv)
        self.R_mitral = float(R_mitral)
        self.log = {"t": [], "P_Pa": [], "V_m3": []}

    def step(self, t, P_PV, P_LA, P_LV, mitral_open, dt):
        Q_pv_in = max((P_PV - P_LA) / self.R_pv, 0.0)
        Q_mitral_out = ((P_LA - P_LV) / self.R_mitral) if mitral_open else 0.0
        self.V_m3 = max(self.V_m3 + (Q_pv_in - Q_mitral_out) * dt, 0.0)
        self.log["t"].append(t)
        self.log["P_Pa"].append(P_LA)
        self.log["V_m3"].append(self.V_m3)

    def export_csv(self, path="pv_la.csv"):
        import pandas as pd
        pd.DataFrame(self.log).to_csv(path, index=False)
        return path



# simulator 
def simulate(
    beats: int = 3,
    hr_bpm: float = 65.0,
    dt: float = 1e-3,
    log_dir: str = "outputs",
    mode: str = "la_lv",  # 'la' | 'lv' | 'la_lv' | 'full'
    la_mesh: str = "mesh/idealized_LA.msh",
    lv_mesh: str = "mesh/idealized_LV.msh",
    la_mesh_units: str = "auto",
    lv_mesh_units: str = "auto",
    make_plots: bool = True,
    valve_tau_open: float = 0.02,
    valve_tau_close: float = 0.05,
    valve_slope_mmHg: float = 1.0,
    lv_fill_mmHg: float = 12.0,
    resp_mmHg: float = 2.0,
    E_la_min: float = 0.08,  # minimum LA elastance [mmHg/mL]
    R_sys: float = 1.5e8,     # Pa·s/m^3  (≈ MAP/CO for 120/80 & ~5 L/min)
    C_sys: float = 1.3e-8,    # m^3/Pa    (≈ SV/PP ≈ 1.75 mL/mmHg)
    starling_gain: float = 0.3,
):
    """
    Physiologic heart simulator (LA/LV elastance + Windkessel). Returns dict with written file paths.
    """
    import os, time
    import numpy as np
    import pandas as pd

    os.makedirs(log_dir, exist_ok=True)
    
    # --- timing
    T_cyc = 60.0 / max(hr_bpm, 1e-6)
    T_end = beats * T_cyc

    # --- afterload & sources
    ao = AorticWK2(R=R_sys, C=C_sys, P0_mmHg=80.0, Pv_mmHg=5.0)
    pv_src = PulmVenousSource(P0_mmHg=12.0, resp_mmHg=2.0, resp_hz=0.25)

    # --- valves (left side)
    mitral = SoftValve(R=1.5e8, tau_open=valve_tau_open, tau_close=valve_tau_close, slope_mmHg=valve_slope_mmHg)
    aortic = SoftValve(R=2.0e8, tau_open=valve_tau_open, tau_close=valve_tau_close, slope_mmHg=valve_slope_mmHg)

    # --- initial volumes
    def init_vol_m3(default_mL: float, mesh_path: str | None, mesh_units: str) -> float:
        if mesh_path and volume_from_mesh is not None:
            try:
                v = float(volume_from_mesh(mesh_path, units=mesh_units))
                
                if np.isfinite(v) and v > 0.0:
                    if v > 1e-3:
                        v *= 1e-9  # if mashes are huge, assuming units were wrong and downscale as mm³ seems to help
                    return v
            except Exception:
                pass
        return default_mL * 1e-6
    if not la_mesh:
        V_la = 30e-6
    if not lv_mesh:
        V_lv = 120e-6
    V_la = init_vol_m3(30.0,  la_mesh, la_mesh_units) if mode in ("la","la_lv","full") else 0.0
    V_lv = init_vol_m3(130.0, lv_mesh, lv_mesh_units) if mode in ("lv","la_lv","full") else 0.0

    # --- elastance parameters
    E_lv_min = mmHg_per_mL(0.08)   # ~0.08 mmHg/mL
    E_lv_max = mmHg_per_mL(3.5)
    V0_lv    = 10e-6               # ~10 mL
    E_la_min = mmHg_per_mL(E_la_min)
    E_la_max = mmHg_per_mL(0.5)
    V0_la    = 5e-6

    # For Starling scaling
    Vref_LV  = max(V_lv, 120e-6)

    # --- logs
    ts, P_LA_arr, P_LV_arr, P_AO_arr = [], [], [], []
    s_mitral_arr, s_aortic_arr = [], []
    Q_mitral_arr, Q_aortic_arr = [], []

    pv_lv = {"t": [], "P_Pa": [], "V_m3": []} if mode in ("lv", "la_lv", "full") else None
    pv_la = {"t": [], "P_Pa": [], "V_m3": []} if mode in ("la", "la_lv", "full") else None

    # --- main loop
    t = 0.0
    while t <= T_end + 1e-12:
        phi = (t % T_cyc) / T_cyc
        e_n = en_double_hill(phi)
        E_lv = E_lv_min + (E_lv_max - E_lv_min) * e_n
        if starling_gain != 0.0 and Vref_LV > 0:
            E_lv *= float(np.clip(1.0 + starling_gain * ((V_lv - Vref_LV) / Vref_LV), 0.3, 3.0))
        # LA elastance (a-wave late diastole; lead ≈ -0.15 in phase)
        e_na = en_double_hill((phi + 0.85) % 1.0)
        E_la = E_la_min + (E_la_max - E_la_min) * e_na
        P_LA = E_la * max(V_la - V0_la, 0.0) if mode in ("la", "la_lv", "full") else lv_fill_mmHg * MMHG_TO_PA
        P_LV = E_lv * max(V_lv - V0_lv, 0.0) if mode in ("lv", "la_lv", "full") else 0.0
        P_AO = ao.P

        # Inflows/outflows (soft valves)
        s_mitral, Q_mitral = mitral.step(dt, dp=(P_LA - P_LV))              # LA -> LV
        s_aortic,  Q_aortic = aortic.step(dt, dp=(P_LV - P_AO))              # LV -> AO

        # Pulmonary venous inflow to LA
        P_PV = pv_src.pressure(t)
        R_pv = 2.0e7
        Q_pv_in = max((P_PV - P_LA) / R_pv, 0.0)

        # Volume updates (mass balance)
        if mode in ("la", "la_lv", "full"):
            V_la = max(V_la + (Q_pv_in - Q_mitral) * dt, 0.0)
        if mode in ("lv", "la_lv", "full"):
            V_lv = max(V_lv + (Q_mitral - Q_aortic) * dt, 0.0)

        # Afterload update (systemic arteries)
        ao.step(dt, Qin=Q_aortic if mode in ("lv", "la_lv", "full") else 0.0)

        # --- logging
        ts.append(t)
        P_LA_arr.append(P_LA if mode in ("la", "la_lv", "full") else 0.0)
        P_LV_arr.append(P_LV if mode in ("lv", "la_lv", "full") else 0.0)
        P_AO_arr.append(ao.P)
        s_mitral_arr.append(s_mitral); s_aortic_arr.append(s_aortic)
        Q_mitral_arr.append(Q_mitral); Q_aortic_arr.append(Q_aortic)

        if pv_la is not None:
            pv_la["t"].append(t); pv_la["P_Pa"].append(P_LA); pv_la["V_m3"].append(V_la)
        if pv_lv is not None:
            pv_lv["t"].append(t); pv_lv["P_Pa"].append(P_LV); pv_lv["V_m3"].append(V_lv)

        t += dt

    main_csv = os.path.join(log_dir, "simlog.csv")
    df = pd.DataFrame({
        "t": ts,
        "P_LA(Pa)": P_LA_arr,
        "P_LV(Pa)": P_LV_arr,
        "P_AO(Pa)": P_AO_arr,
        "s_mitral": s_mitral_arr,
        "s_aortic": s_aortic_arr,
        "Q_mitral(m3/s)": Q_mitral_arr,
        "Q_aortic(m3/s)": Q_aortic_arr,
        "mode": mode,
    })
    df.to_csv(main_csv, index=False)

    pv_lv_csv = pv_la_csv = None
    if pv_lv is not None:
        pv_lv_csv = os.path.join(log_dir, "pv_lv.csv")
        pd.DataFrame(pv_lv).to_csv(pv_lv_csv, index=False)
    if pv_la is not None:
        pv_la_csv = os.path.join(log_dir, "pv_la.csv")
        pd.DataFrame(pv_la).to_csv(pv_la_csv, index=False)

    # --- plots
    pngs = []
    if make_plots:
        try:
            import matplotlib.pyplot as plt
            t_arr = np.array(ts)
            if len(t_arr) >= 2:
                T_cyc = 60.0 / max(hr_bpm, 1e-6)
                t_end = t_arr[-1]
                mask_last = (t_arr >= (t_end - T_cyc - 1e-9))
            else:
                mask_last = slice(None)
            plt.figure(figsize=(9, 4.5))
            if mode in ("la", "la_lv", "full"):
                plt.plot(t_arr, np.array(P_LA_arr) * PA_TO_MMHG, label="P_LA")
            if mode in ("lv", "la_lv", "full"):
                plt.plot(t_arr, np.array(P_LV_arr) * PA_TO_MMHG, label="P_LV")
            plt.plot(t_arr, np.array(P_AO_arr) * PA_TO_MMHG, label="P_AO")
            plt.step(t_arr, np.array(s_mitral_arr) * 40, where="post", label="mitral s×40")
            plt.step(t_arr, np.array(s_aortic_arr) * 40, where="post", label="aortic s×40")
            plt.xlabel("t [s]"); plt.ylabel("mmHg / scaled s"); plt.legend(); plt.tight_layout()
            #p0 = os.path.join(log_dir, "heart_states.png"); plt.savefig(p0, dpi=150); pngs.append(p0)

            if pv_lv is not None:
                V = np.array(pv_lv["V_m3"])[mask_last] * 1e6
                P = np.array(pv_lv["P_Pa"])[mask_last] * PA_TO_MMHG
                plt.figure(figsize=(5.0, 5.0))
                plt.plot(V, P, "-")
                plt.xlabel("LV Volume [mL]"); plt.ylabel("LV Pressure [mmHg]")
                plt.title("LV PV loop")
                plt.tight_layout(); p1 = os.path.join(log_dir, "pv_lv.png"); plt.savefig(p1, dpi=150); pngs.append(p1)
                

            if pv_la is not None:
                V = np.array(pv_la["V_m3"])[mask_last] * 1e6
                P = np.array(pv_la["P_Pa"])[mask_last] * PA_TO_MMHG
                plt.figure(figsize=(5.0, 5.0))
                plt.plot(V, P, "-")
                plt.xlabel("LA Volume [mL]"); plt.ylabel("LA Pressure [mmHg]")
                plt.title("LA PV loop")
                plt.tight_layout(); p2 = os.path.join(log_dir, "pv_la.png"); plt.savefig(p2, dpi=150); pngs.append(p2)

        except Exception:
            pass

    return {"main_csv": main_csv, "pv_lv_csv": pv_lv_csv, "pv_la_csv": pv_la_csv, "plots": pngs}