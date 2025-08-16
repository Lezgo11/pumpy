# tests/test_pressures.py
import numpy as np
from pressures import (
    atrial_pressure, ventricular_pressure,
    atrial_elastance_SI, ventricular_elastance_SI,
    MMHG_TO_PA,
)

def test_atrial_pressure_range_and_period():
    T = 1.0
    ts = np.linspace(0, T, 501)
    ps = np.array([atrial_pressure(t, T) for t in ts])
    assert np.all(ps >= 0.0)
    # within physiological range (2–12 mmHg ≈ 266–1600 Pa), allow slack
    assert ps.min() > 200.0 and ps.max() < 2_000.0

    # periodicity (value at 0 and T close)
    assert abs(atrial_pressure(0.0, T) - atrial_pressure(T, T)) < 1e-9

def test_ventricular_pressure_has_systolic_bump():
    T = 1.0
    ts = np.linspace(0, T, 501)
    ps = np.array([ventricular_pressure(t, T) for t in ts])
    assert np.all(ps >= 0.0)
    # Systolic peak substantially above base
    assert ps.max() > 10_000.0  # > ~75 mmHg
    # Diastolic base ~5 mmHg
    assert abs(ventricular_pressure(0.75*T, T)/MMHG_TO_PA - 5.0) < 3.0  # ±3 mmHg band

def test_elastance_bounds_and_activation():
    T = 1.0
    # Atrial elastance should be between Emin and Emax across cycle
    E_as = [atrial_elastance_SI(t, T) for t in np.linspace(0, T, 101)]
    assert min(E_as) >= atrial_elastance_SI(0, T, Emin_mmHg_ml=0.15, Emax_mmHg_ml=0.15)
    assert max(E_as) <= atrial_elastance_SI(0, T, Emin_mmHg_ml=0.5, Emax_mmHg_ml=0.5)

    # Ventricular elastance active in first half only
    t1, t2 = 0.25*T, 0.75*T
    assert ventricular_elastance_SI(t1, T) > ventricular_elastance_SI(t2, T)
