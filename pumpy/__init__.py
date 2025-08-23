"""
PumPy

A package for cardiac electrophysiology and mechanics simulations
using FEniCS/DOLFINx for finite element analysis.

Main modules:
- core: Core simulation components (electrophysiology, mechanics, physiology)
- models: Physiological models (circulation, chambers, valves)
- logging: Data logging and visualization
- simulation: Main simulation runner and configuration
- utils: Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "Lez"
__email__ = "leslyperlaza@gmail.com"

# Import main classes for easy access
from .core.constants import MMHG_TO_PA, PA_TO_MMHG
from .core.electrophysiology import EPDriver
from .core.physiology import PhysioTuner
from .models.circulation import AorticWK2, PulmVenousSource
from .logging.volume_loggers import VentricularVolumeLogger, AtrialVolumeLogger
from .simulation.runner import CardiacSimulation
from .simulation.config import SimulationConfig

__all__ = [
    "MMHG_TO_PA",
    "PA_TO_MMHG",
    "EPDriver",
    "PhysioTuner", 
    "AorticWK2",
    "PulmVenousSource",
    "VentricularVolumeLogger",
    "AtrialVolumeLogger",
    "CardiacSimulation",
    "SimulationConfig",
]