"""
Physical constants and unit conversions for cardiac simulation.
"""

# Unit conversions
MMHG_TO_PA = 133.322368  # Convert mmHg to Pascals
PA_TO_MMHG = 1.0 / MMHG_TO_PA  # Convert Pascals to mmHg

# Default physiological parameters
DEFAULT_PHYS_TARGETS = {
    "SBP_mmHg": 120.0,      # Systolic blood pressure
    "DBP_mmHg": 80.0,       # Diastolic blood pressure  
    "LV_EDV_mL": 130.0,     # Left ventricular end-diastolic volume
    "LA_EDV_mL": 60.0,      # Left atrial end-diastolic volume
    "LV_ESV_mL": 50.0,      # Left ventricular end-systolic volume
    "HR_bpm": 70.0,         # Heart rate in beats per minute
}

DEFAULT_GAINS = {
    "active_stress_max_kPa": 10.0,  # Maximum active stress
    "pressure_gain": 0.15,          # Pressure tuning gain
    "volume_gain": 0.10,            # Volume tuning gain
}

# Material properties (default values)
DEFAULT_MATERIAL = {
    "E": 10e3,              # Young's modulus (Pa)
    "nu": 0.49,             # Poisson's ratio (nearly incompressible)
}

# Simulation defaults
DEFAULT_SIMULATION = {
    "dt": 0.1,              # Time step (s)
    "cycles": 3,            # Number of cardiac cycles to simulate
    "newton_rtol": 1e-6,    # Newton solver relative tolerance
    "newton_max_it": 60,    # Newton solver maximum iterations
}