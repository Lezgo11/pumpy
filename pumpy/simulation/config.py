"""
Configuration management for cardiac simulations.

This module provides configuration classes and utilities for managing
simulation parameters, making it easy to set up and modify simulations.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import yaml

from ..core.constants import (
    DEFAULT_PHYS_TARGETS, DEFAULT_GAINS, DEFAULT_MATERIAL, DEFAULT_SIMULATION
)


@dataclass
class PhysiologyConfig:
    """Physiological parameters configuration."""
    sbp_mmhg: float = DEFAULT_PHYS_TARGETS["SBP_mmHg"]
    dbp_mmhg: float = DEFAULT_PHYS_TARGETS["DBP_mmHg"] 
    lv_edv_ml: float = DEFAULT_PHYS_TARGETS["LV_EDV_mL"]
    la_edv_ml: float = DEFAULT_PHYS_TARGETS["LA_EDV_mL"]
    lv_esv_ml: float = DEFAULT_PHYS_TARGETS["LV_ESV_mL"]
    hr_bpm: float = DEFAULT_PHYS_TARGETS["HR_bpm"]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format expected by PhysioTuner."""
        return {
            "SBP_mmHg": self.sbp_mmhg,
            "DBP_mmHg": self.dbp_mmhg,
            "LV_EDV_mL": self.lv_edv_ml,
            "LA_EDV_mL": self.la_edv_ml,
            "LV_ESV_mL": self.lv_esv_ml,
            "HR_bpm": self.hr_bpm,
        }


@dataclass 
class GainsConfig:
    """Control gains configuration."""
    active_stress_max_kPa: float = DEFAULT_GAINS["active_stress_max_kPa"]
    pressure_gain: float = DEFAULT_GAINS["pressure_gain"]
    volume_gain: float = DEFAULT_GAINS["volume_gain"]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class MaterialConfig:
    """Material properties configuration."""
    E: float = DEFAULT_MATERIAL["E"]              # Young's modulus (Pa)
    nu: float = DEFAULT_MATERIAL["nu"]            # Poisson's ratio
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class ChambersConfig:
    """Cardiac chambers configuration."""
    active_chambers: List[str] = field(default_factory=lambda: ["LV", "LA"])
    
    # Mesh file paths
    la_mesh_path: str = "mesh/hollow_sphere_LA.msh"
    lv_mesh_path: str = "mesh/idealized_LV.msh"
    la_adv_mesh_path: str = "mesh/LA_adv.msh"
    
    # Boundary condition tags
    la_load_tags: Tuple[int, ...] = (30,)
    la_bc_tags: int = 10
    lv_load_tags: Tuple[int, ...] = (11,)
    lv_bc_tags: int = None
    la_adv_bc_tags: int = 50


@dataclass
class CirculationConfig:
    """Circulation model parameters."""
    # Aortic Windkessel
    aortic_resistance: float = 1.1e7      # Systemic resistance (Pa·s/m³)
    aortic_compliance: float = 1.5e-3     # Aortic compliance (m³/Pa)
    aortic_valve_resistance: float = 2.0e8 # Aortic valve resistance (Pa·s/m³)
    
    # Mitral valve
    mitral_resistance: float = 2.0e8      # Mitral valve resistance (Pa·s/m³)
    
    # Pulmonary venous
    pv_pressure_mmhg: float = 12.0        # Pulmonary venous pressure (mmHg)
    pv_resistance: float = 2.0e8          # Pulmonary venous resistance (Pa·s/m³)


@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    dt: float = DEFAULT_SIMULATION["dt"]
    cycles: int = DEFAULT_SIMULATION["cycles"]
    newton_rtol: float = DEFAULT_SIMULATION["newton_rtol"]
    newton_max_it: int = DEFAULT_SIMULATION["newton_max_it"]


@dataclass
class Config:
    """Complete simulation configuration."""
    # Feature flags
    enable_volume: bool = True
    enable_electro: bool = False
    enable_physio_tuner: bool = False
    
    # Configuration sections
    physiology: PhysiologyConfig = field(default_factory=PhysiologyConfig)
    gains: GainsConfig = field(default_factory=GainsConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    chambers: ChambersConfig = field(default_factory=ChambersConfig)
    circulation: CirculationConfig = field(default_factory=CirculationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary
            
        Returns
        -------
        Config
            Configuration object
        """
        # Extract main flags
        enable_volume = config_dict.get("enable_volume", True)
        enable_electro = config_dict.get("enable_electro", False) 
        enable_physio_tuner = config_dict.get("enable_physio_tuner", False)
        
        # Extract and create sub-configurations
        physiology = PhysiologyConfig(**config_dict.get("physiology", {}))
        gains = GainsConfig(**config_dict.get("gains", {}))
        materials = MaterialConfig(**config_dict.get("materials", {}))
        chambers = ChambersConfig(**config_dict.get("chambers", {}))
        circulation = CirculationConfig(**config_dict.get("circulation", {}))
        simulation = SimulationConfig(**config_dict.get("simulation", {}))
        
        return cls(
            enable_volume=enable_volume,
            enable_electro=enable_electro,
            enable_physio_tuner=enable_physio_tuner,
            physiology=physiology,
            gains=gains,
            materials=materials,
            chambers=chambers,
            circulation=circulation,
            simulation=simulation
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        yaml_path : str or Path
            Path to YAML configuration file
            
        Returns
        -------
        Config
            Configuration object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            "enable_volume": self.enable_volume,
            "enable_electro": self.enable_electro,
            "enable_physio_tuner": self.enable_physio_tuner,
            "physiology": asdict(self.physiology),
            "gains": asdict(self.gains),
            "materials": asdict(self.materials),
            "chambers": asdict(self.chambers),
            "circulation": asdict(self.circulation),
            "simulation": asdict(self.simulation),
        }
    
    def save(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Parameters
        ----------
        yaml_path : str or Path
            Path to save YAML configuration file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns
        -------
        list of str
            List of validation messages
        """
        warnings = []
        
        # Check mesh file paths exist
        for chamber in self.chambers.active_chambers:
            if chamber == "LA":
                mesh_path = Path(self.chambers.la_mesh_path)
                if not mesh_path.exists():
                    warnings.append(f"LA mesh file not found: {mesh_path}")
            elif chamber == "LV":
                mesh_path = Path(self.chambers.lv_mesh_path)
                if not mesh_path.exists():
                    warnings.append(f"LV mesh file not found: {mesh_path}")
            elif chamber == "LA_adv":
                mesh_path = Path(self.chambers.la_adv_mesh_path)
                if not mesh_path.exists():
                    warnings.append(f"LA_adv mesh file not found: {mesh_path}")
        
        # Check physiological parameters
        if self.physiology.hr_bpm <= 0:
            warnings.append("Heart rate must be positive")
        
        if self.physiology.sbp_mmhg <= self.physiology.dbp_mmhg:
            warnings.append("Systolic pressure should be greater than diastolic")
        
        # Check simulation parameters
        if self.simulation.dt <= 0:
            warnings.append("Time step must be positive")
        
        if self.simulation.cycles <= 0:
            warnings.append("Number of cycles must be positive")
        
        # Check material properties
        if self.materials.nu >= 0.5:
            warnings.append("Poisson's ratio should be < 0.5 for stability")
        
        return warnings


# Alias for backward compatibility
SimulationConfig = Config