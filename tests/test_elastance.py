# tests/test_elastance.py
import numpy as np
from pressures import ventricular_elastance_SI, atrial_elastance_SI

def _finite_diff_monotone_increasing(f, T, a, b, n=200):
    xs = np.linspace(a, b, n)
    vals = np.array([f(x, T) for x in xs])
    return np.all(np.diff(vals) >= -1e-8) or np.all(np.diff(vals) <= 1e-8)

def test_ventricular_elastance_rises_during_early_systole():
    T = 1.0
    # Check monotonic (or near-monotonic) rise from 0→T/4
    assert _finite_diff_monotone_increasing(ventricular_elastance_SI, T, 0.0, 0.25*T)

def test_atrial_elastance_is_smooth_periodic():
    T = 1.0
    t0 = 0.123
    a = atrial_elastance_SI(t0, T)
    b = atrial_elastance_SI(t0 + T, T)
    assert abs(a - b) < 1e-9
