from __future__ import annotations
import math
import os
from dataclasses import dataclass
import numpy as np

# ---- Units
MMHG_TO_PA = 133.322368
PA_TO_MMHG = 1.0 / MMHG_TO_PA


def edpvr_pressure(V_m3, V0_m3, A_mmHg=0.15, B_per_mL=0.03):
    """
    Exponential EDPVR: P_passive = A*(exp(B*(V-V0)) - 1)
    A in mmHg, B in 1/mL; convert to Pa.
    """
    A = A_mmHg * MMHG_TO_PA
    B = B_per_mL / 1e-6  # (1/mL) -> (1/m^3)
    x = max(V_m3 - V0_m3, 0.0)
    return A * (np.exp(B * x) - 1.0)

def smooth_cos_window(phase: float) -> float:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(phase, 0.0, 1.0)))

def atrial_pressure_pa(
    t: float,
    T: float,
    p_min_mmHg: float = 10.0,
    p_max_mmHg: float = 15.0,
) -> float:
    """Simple atrial pressure waveform (standalone LA mode or as LA driver)."""
    tau = ((t + 0.25 * T) % T) / T
    p_mmHg = p_min_mmHg + (p_max_mmHg - p_min_mmHg) * smooth_cos_window(tau)
    return p_mmHg * MMHG_TO_PA

def mmHg_per_mL(x): 
    return x * MMHG_TO_PA / 1e-6

def en_double_hill(phi, ta=0.3, tb=0.5, m1=1.9, m2=1.9):
    """Normalized elastance function from Stergiopulos et al. (1999)"""
    x1 = (phi/ta)**m1
    x2 = (phi/tb)**m2
    return (x1/(1.0+x1)) * (1.0/(1.0+x2))

