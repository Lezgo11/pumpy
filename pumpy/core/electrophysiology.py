"""
Electrophysiology models for cardiac simulation.

This module contains simplified electrophysiology models for generating
activation patterns and active stress in cardiac tissue.
"""

import numpy as np
from typing import Dict, Union
from .constants import DEFAULT_GAINS


class EPDriver:
    """
    Simple electrophysiology driver based on a modified FitzHugh-Nagumo model.
    
    Generates activation patterns for cardiac tissue that can be used to
    drive active stress in mechanical simulations.
    
    Parameters
    ----------
    dt : float
        Time step for integration
    hr_bpm : float
        Heart rate in beats per minute
    gains : dict
        Dictionary containing gain parameters, must include 'active_stress_max_kPa'
    """
    
    def __init__(self, dt: float, hr_bpm: float, gains: Dict[str, float] = None):
        self.dt = float(dt)
        self.period = 60.0 / max(hr_bpm, 1e-6)  # Cardiac cycle period
        self.t = 0.0
        
        # State variables
        self.v = 0.0      # Voltage-like variable
        self.w = 1.0      # Recovery variable
        self.a = 0.0      # Activation (normalized 0-1)
        
        # Model parameters
        self.v_gate = 0.13      # Gating threshold
        self.tau_open = 120.0   # Opening time constant
        self.tau_close = 150.0  # Closing time constant  
        self.T_out = 6.0        # Outward current time constant
        
        # Gains
        self.gains = gains if gains is not None else DEFAULT_GAINS.copy()
        
    def _stimulus(self, t: float) -> float:
        """
        Generate stimulus current - short square pulse per cardiac cycle.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        float
            Stimulus current (1.0 during stimulus, 0.0 otherwise)
        """
        return 1.0 if (t % self.period) < 0.002 else 0.0
    
    def step(self) -> float:
        """
        Advance the electrophysiology model by one time step.
        
        Returns
        -------
        float
            Current activation level (0-1)
        """
        # Stimulus current
        I = self._stimulus(self.t)
        
        # Current state
        v, w = self.v, self.w
        
        # State derivatives
        dv = (I - v * (v - self.v_gate) * (1 - v) - v * w / self.T_out) * self.dt
        dw = ((1 - w) / self.tau_close if v < self.v_gate else -w / self.tau_open) * self.dt
        
        # Update state (with clipping for stability)
        self.v = float(np.clip(v + dv, 0.0, 1.0))
        self.w = float(np.clip(w + dw, 0.0, 1.0))
        
        # Update time
        self.t += self.dt
        
        # Calculate activation
        if self.v < self.v_gate:
            self.a = 0.0
        else:
            self.a = (self.v - self.v_gate) / (1 - self.v_gate)
        
        self.a = float(np.clip(self.a, 0.0, 1.0))
        
        return self.a
    
    def active_stress_kPa(self) -> float:
        """
        Calculate active stress from current activation level.
        
        Returns
        -------
        float
            Active stress in kPa
        """
        return self.gains["active_stress_max_kPa"] * self.a
    
    def reset(self):
        """Reset the EP model to initial conditions."""
        self.t = 0.0
        self.v = 0.0
        self.w = 1.0
        self.a = 0.0
        
    def get_state(self) -> Dict[str, float]:
        """
        Get current state of the EP model.
        
        Returns
        -------
        dict
            Dictionary containing current state variables
        """
        return {
            "t": self.t,
            "v": self.v,
            "w": self.w,
            "activation": self.a,
            "active_stress_kPa": self.active_stress_kPa()
        }