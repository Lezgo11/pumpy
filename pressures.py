# pressures.py
import numpy as np

MMHG_TO_PA = 133.322368
ML_TO_M3 = 1e-6

def mmHg_per_ml_to_Pa_per_m3(val_mmHg_per_ml: float) -> float:
    return val_mmHg_per_ml * MMHG_TO_PA / ML_TO_M3

def smooth_cos_window(t, T):
    """0→1→0 periodic, C1-smooth. Peak at t=T/2."""
    return 0.5 * (1.0 - np.cos(2.0*np.pi*(t % T)/T))

# ---- Elastance (E(t)) ----
def atrial_elastance_SI(t, T, Emin_mmHg_ml=0.15, Emax_mmHg_ml=0.5):
    Emin = mmHg_per_ml_to_Pa_per_m3(Emin_mmHg_ml)
    Emax = mmHg_per_ml_to_Pa_per_m3(Emax_mmHg_ml)
    act = smooth_cos_window(t, T)  # atrium: broad activation across cycle
    return Emin + (Emax - Emin) * act

def ventricular_elastance_SI(t, T, Emin_mmHg_ml=0.06, Emax_mmHg_ml=2.0):
    Emin = mmHg_per_ml_to_Pa_per_m3(Emin_mmHg_ml)
    Emax = mmHg_per_ml_to_Pa_per_m3(Emax_mmHg_ml)
    tau = (t % T) / T
    act = smooth_cos_window(t, T) if tau < 0.5 else 0.0  # ventricle: systole in first half
    return Emin + (Emax - Emin) * act

# ---- Pressure waveforms p(t) ----
def atrial_pressure(t, T, p_min_mmHg=2.0, p_max_mmHg=12.0):
    """Simple periodic LA pressure (mmHg) converted to Pa."""
    amp = (p_max_mmHg - p_min_mmHg)
    p_mmHg = p_min_mmHg + amp * smooth_cos_window(t + 0.25*T, T)  # peak late
    return p_mmHg * MMHG_TO_PA

def ventricular_pressure(t, T, p_min_mmHg=5.0, p_max_mmHg=130.0):
    """Simple periodic LV pressure (mmHg) converted to Pa (systolic bump in first half)."""
    tau = (t % T) / T
    base = p_min_mmHg
    bump = (p_max_mmHg - p_min_mmHg) * (smooth_cos_window(t, T) if tau < 0.5 else 0.0)
    return (base + bump) * MMHG_TO_PA
