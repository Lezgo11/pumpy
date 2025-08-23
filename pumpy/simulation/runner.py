"""
Main simulation runner for cardiac simulations.

This module provides a high-level interface for running cardiac simulations
with configurable parameters and automatic result logging.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from mpi4py import MPI

from ..core.constants import MMHG_TO_PA, PA_TO_MMHG
from ..core.electrophysiology import EPDriver
from ..core.physiology import PhysioTuner
from ..core.mechanics import make_solid, newton_solve
from ..models.circulation import AorticWK2, PulmVenousSource
from ..logging.volume_loggers import VentricularVolumeLogger, AtrialVolumeLogger
from ..utils.mesh_io import load_mesh
from ..utils.helpers import smooth_cos_window, atrial_pressure_pa
from .config import SimulationConfig


class CardiacSimulation:
    """
    Main cardiac simulation class.
    
    This class orchestrates the cardiac simulation, managing all components
    including electrophysiology, mechanics, circulation, and logging.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration object containing all simulation parameters
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.t = 0.0
        self.T_cyc = 60.0 / config.physiology.hr_bpm
        self.T_end = config.simulation.cycles * self.T_cyc
        
        # Initialize components
        self._init_chambers()
        self._init_ep_model()
        self._init_physiology_tuner()
        self._init_circulation_models()
        self._init_volume_loggers()
        self._init_logging()
        
        # Status tracking
        self._start_time = None
        self._converged_steps = 0
        self._total_steps = 0
        
    def _init_chambers(self):
        """Initialize cardiac chambers based on configuration."""
        self.chambers = {}
        
        if "LA" in self.config.chambers.active_chambers:
            mesh_data = load_mesh(self.config.chambers.la_mesh_path)
            self.chambers["LA"] = make_solid(
                mesh_data["domain"], 
                mesh_data["facet_tags"],
                load_tags=self.config.chambers.la_load_tags,
                bc_tags=self.config.chambers.la_bc_tags,
                **self.config.materials.to_dict()
            )
            
        if "LV" in self.config.chambers.active_chambers:
            mesh_data = load_mesh(self.config.chambers.lv_mesh_path)
            self.chambers["LV"] = make_solid(
                mesh_data["domain"],
                mesh_data["facet_tags"], 
                load_tags=self.config.chambers.lv_load_tags,
                bc_tags=self.config.chambers.lv_bc_tags,
                **self.config.materials.to_dict()
            )
    
    def _init_ep_model(self):
        """Initialize electrophysiology model if enabled."""
        if self.config.enable_electro:
            self.ep_model = EPDriver(
                dt=self.config.simulation.dt,
                hr_bpm=self.config.physiology.hr_bpm,
                gains=self.config.gains.to_dict()
            )
        else:
            self.ep_model = None
    
    def _init_physiology_tuner(self):
        """Initialize physiology tuner if enabled."""
        if self.config.enable_physio_tuner:
            self.physio_tuner = PhysioTuner(
                targets=self.config.physiology.to_dict(),
                gains=self.config.gains.to_dict()
            )
        else:
            self.physio_tuner = None
    
    def _init_circulation_models(self):
        """Initialize circulation models."""
        self.aortic_wk = None
        self.pv_source = None
        
        if "LV" in self.config.chambers.active_chambers:
            self.aortic_wk = AorticWK2(
                R=self.config.circulation.aortic_resistance,
                C=self.config.circulation.aortic_compliance,
                P0_mmHg=self.config.physiology.dbp_mmhg
            )
            
        if "LA" in self.config.chambers.active_chambers:
            self.pv_source = PulmVenousSource(
                P0_mmHg=self.config.circulation.pv_pressure_mmhg
            )
    
    def _init_volume_loggers(self):
        """Initialize volume loggers if enabled."""
        self.volume_loggers = {}
        
        if (self.config.enable_volume and 
            "LV" in self.config.chambers.active_chambers):
            self.volume_loggers["LV"] = VentricularVolumeLogger(
                V0_mL=self.config.physiology.lv_edv_ml,
                R_mitral=self.config.circulation.mitral_resistance,
                R_aortic=self.config.circulation.aortic_valve_resistance
            )
            
        if (self.config.enable_volume and 
            "LA" in self.config.chambers.active_chambers):
            self.volume_loggers["LA"] = AtrialVolumeLogger(
                V0_mL=self.config.physiology.la_edv_ml,
                R_pv=self.config.circulation.pv_resistance,
                R_mitral=self.config.circulation.mitral_resistance
            )
    
    def _init_logging(self):
        """Initialize simulation logging."""
        self.log = {
            "t": [],
            "P_LA_Pa": [],
            "P_LV_raw_Pa": [],
            "P_LV_eff_Pa": [],
            "P_AO_Pa": [],
            "mitral_open": [],
            "aortic_open": [],
            "active_stress_kPa": []
        }
    
    def _elastance(self, t: float) -> float:
        """
        Time-varying elastance function for LV pressure generation.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        float
            Normalized elastance (0-1)
        """
        phi = (t % self.T_cyc) / self.T_cyc
        return 0.05 + 0.95 * smooth_cos_window((phi + 0.1) % 1.0)
    
    def _compute_pressures(self, t: float) -> Dict[str, float]:
        """
        Compute chamber and vessel pressures at current time.
        
        Parameters
        ----------
        t : float
            Current time
            
        Returns
        -------
        dict
            Dictionary of pressures in Pascals
        """
        pressures = {}
        
        # Left atrial pressure
        if "LA" in self.chambers:
            pressures["P_LA"] = atrial_pressure_pa(t, self.T_cyc)
            if self.physio_tuner:
                pressures["P_LA"] = self.physio_tuner.scale_P(
                    pressures["P_LA"], systole=False
                )
        else:
            pressures["P_LA"] = 10.0 * MMHG_TO_PA
            
        # Pulmonary venous pressure
        if self.pv_source:
            pressures["P_PV"] = self.pv_source.pressure(t)
        else:
            pressures["P_PV"] = 12.0 * MMHG_TO_PA
            
        # Left ventricular pressures
        if "LV" in self.chambers:
            P_LV_passive = 8.0 * MMHG_TO_PA
            pressures["P_LV_raw"] = P_LV_passive + 60e3 * self._elastance(t)
            
            # Aortic pressure
            pressures["P_AO"] = (self.aortic_wk.P if self.aortic_wk 
                               else 80.0 * MMHG_TO_PA)
            
            # Valve states
            mitral_open = pressures["P_LA"] > pressures["P_LV_raw"]
            aortic_open = pressures["P_LV_raw"] > pressures["P_AO"]
            
            # Effective LV pressure
            if aortic_open:
                pressures["P_LV_eff"] = pressures["P_LV_raw"]
            elif mitral_open:
                pressures["P_LV_eff"] = pressures["P_LA"]
            else:
                pressures["P_LV_eff"] = pressures["P_LV_raw"]
                
            if self.physio_tuner:
                pressures["P_LV_eff"] = self.physio_tuner.scale_P(
                    pressures["P_LV_eff"], systole=aortic_open
                )
                
            pressures["mitral_open"] = mitral_open
            pressures["aortic_open"] = aortic_open
        else:
            pressures.update({
                "P_LV_raw": np.nan,
                "P_LV_eff": np.nan, 
                "P_AO": np.nan,
                "mitral_open": False,
                "aortic_open": False
            })
            
        return pressures
    
    def _solve_mechanics(self, pressures: Dict[str, float]) -> Dict[str, Tuple[bool, int]]:
        """
        Solve mechanical equilibrium for all active chambers.
        
        Parameters
        ----------
        pressures : dict
            Dictionary of chamber pressures
            
        Returns
        -------
        dict
            Dictionary of convergence results (converged, iterations)
        """
        results = {}
        
        if "LA" in self.chambers:
            converged, iterations = self.chambers["LA"]["apply_and_solve"](
                pressures["P_LA"]
            )
            results["LA"] = (converged, iterations)
            
        if "LV" in self.chambers:
            converged, iterations = self.chambers["LV"]["apply_and_solve"](
                pressures["P_LV_eff"]
            )
            results["LV"] = (converged, iterations)
            
        return results
    
    def _update_volume_loggers(self, t: float, pressures: Dict[str, float]):
        """Update volume loggers with current state."""
        dt = self.config.simulation.dt
        
        if "LV" in self.volume_loggers:
            self.volume_loggers["LV"].step(
                t, pressures["P_LA"], pressures["P_LV_eff"],
                pressures["P_AO"], pressures["mitral_open"],
                pressures["aortic_open"], dt
            )
            
        if "LA" in self.volume_loggers:
            self.volume_loggers["LA"].step(
                t, pressures["P_PV"], pressures["P_LA"],
                pressures.get("P_LV_raw", pressures["P_LA"]),
                pressures["mitral_open"], dt
            )
    
    def _update_circulation(self, pressures: Dict[str, float]):
        """Update circulation models."""
        dt = self.config.simulation.dt
        
        if self.aortic_wk and "LV" in self.chambers:
            if pressures["aortic_open"]:
                Q_out = (pressures["P_LV_eff"] - pressures["P_AO"]) / 2.0e8
            else:
                Q_out = 0.0
            self.aortic_wk.step(dt, Q_out)
    
    def _log_step(self, t: float, pressures: Dict[str, float], 
                  active_stress: float):
        """Log current simulation step."""
        self.log["t"].append(t)
        self.log["P_LA_Pa"].append(pressures["P_LA"])
        self.log["P_LV_raw_Pa"].append(pressures.get("P_LV_raw", np.nan))
        self.log["P_LV_eff_Pa"].append(pressures.get("P_LV_eff", np.nan))
        self.log["P_AO_Pa"].append(pressures.get("P_AO", np.nan))
        self.log["mitral_open"].append(int(pressures.get("mitral_open", 0)))
        self.log["aortic_open"].append(int(pressures.get("aortic_open", 0)))
        self.log["active_stress_kPa"].append(active_stress)
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run the cardiac simulation.
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        dict
            Dictionary containing simulation results and statistics
        """