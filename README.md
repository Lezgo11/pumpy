# PumPy

A comprehensive Python package for cardiac electrophysiology to never skip a beat in our research!

Using FEniCS/DOLFINx, PumPy allows your to set up your fem mechanics simulations for heart simulations.

## Features

- **Cardiac Mechanics**: Finite element modeling of cardiac tissue deformation
- **Electrophysiology**: Simplified EP models for activation patterns
- **Circulation Models**: Windkessel models for afterload and preload
- **Volume Tracking**: Pressure-volume loop generation and analysis
- **Multi-Chamber Support**: Left ventricle (LV) and left atrium (LA) modeling
- **Configurable Parameters**: Easy setup via YAML configuration files
- **Visualization**: Automatic generation of pressure-volume loops and time series plots

## Installation
### Prerequisites

This package requires several dependencies including FEniCS/DOLFINx, which can be challenging to install. We recommend using conda:

```bash
# Create a new conda environment
conda create -n pumpy python=3.10
conda activate pumpy

# Install FEniCS/DOLFINx
conda install -c conda-forge fenics-dolfinx

# Install other scientific computing dependencies
conda install -c conda-forge numpy scipy matplotlib pandas pyyaml
```

### Install the Package

#### From source (development):
```bash
git clone TODO
cd pumpy
pip install -e .
```

#### From PyPI (when available):
```bash
pip install pumpy
```

## Quick Start

### Basic Usage

```python
from pumpy import CardiacSimulation, Config

# Create default configuration
config = Config()

# Customize parameters
config.simulation.cycles = 3
config.physiology.hr_bpm = 75.0
config.chambers.active_chambers = ["LV", "LA"]

# Run simulation
sim = CardiacSimulation(config)
results = sim.run()

# Save results and plots
sim.save_results(results, "my_simulation")
sim.plot_results(results, "my_simulation")
```

### Configuration File

Create a YAML configuration file:

```yaml
# config.yaml
enable_volume: true
enable_electro: false
enable_physio_tuner: false

physiology:
  hr_bpm: 70.0
  sbp_mmhg: 120.0
  dbp_mmhg: 80.0
  lv_edv_ml: 130.0
  la_edv_ml: 60.0

chambers:
  active_chambers: ["LV", "LA"]
  la_mesh_path: "mesh/hollow_sphere_LA.msh"
  lv_mesh_path: "mesh/idealized_LV.msh"

simulation:
  dt: 0.1
  cycles: 3
```

Then load and use it:

```python
from pumpy import CardiacSimulation, Config

# Load configuration
config = Config.from_yaml("config.yaml")

# Run simulation
sim = CardiacSimulation(config)
results = sim.run()
```

## Package Structure

```
pumpy/
├── core/                  # Core simulation components
│   ├── constants.py       # Physical constants and defaults
│   ├── electrophysiology.py  # EP models
│   ├── mechanics.py       # Solid mechanics
│   └── physiology.py      # Physiological models
├── models/                # Circulation and chamber models
│   ├── circulation.py     # Windkessel, valve models
│   └── chambers.py        # Chamber-specific models
├── logging/               # Data logging and visualization
│   ├── volume_loggers.py  # PV loop logging
│   └── visualization.py   # Plotting utilities  
├── simulation/            # Main simulation runner
│   ├── runner.py          # CardiacSimulation class
│   └── config.py          # Configuration management
└── utils/                 # Utilities
│   ├── mesh_io.py         # Mesh loading
│   └── helpers.py         # Helper functions
└── mesh/
    ├── idealized_LV.msh
    ├── idealized_LA.msh
    ├── realistic_LV.msh
    └── realistic_LA.msh
```

## Examples

See the `examples/` directory for complete example scripts:

- `basic_simulation.py` - Basic LV+LA simulation
- `advanced_simulation.py` - Simulation with electrophysiology
- `parameter_study.py` - Parameter sensitivity study

## Mesh Files
You'll need mesh files for your cardiac geometries. The package expects:

- `mesh/idealized_LV.msh` - Left ventricle mesh
- `mesh/hollow_sphere_LA.msh` - Left atrium mesh  
- `mesh/LA_adv.msh` - Realistic LA mesh (optional)
- `mesh/LV_adv.msh`- Realistic LV mesh (optional)

Mesh files should be in Gmsh format (.msh) with appropriate boundary tags.

## Configuration
The package uses a hierarchical configuration system:

### Main Sections

- **Physiology**: Heart rate, blood pressures, chamber volumes
- **Chambers**: Active chambers and mesh file paths
- **Materials**: Tissue material properties (Young's modulus, Poisson's ratio)
- **Circulation**: Windkessel parameters, valve resistances
- **Simulation**: Time stepping, solver parameters
- **Gains**: Control gains for physiology tuner

### Feature Flags

- `enable_volume`: Enable volume tracking and PV loops
- `enable_electro`: Enable electrophysiology model
- `enable_physio_tuner`: Enable physiological parameter tuning

## Results

Simulations generate several outputs:

- `simulation_log.csv` - Time series of pressures, valve states
- `pv_lv.csv`, `pv_la.csv` - Pressure-volume data for each chamber
- `pressure_vs_time.png` - Pressure and valve state plots
- `pv_loop_lv.png`, `pv_loop_la.png` - Pressure-volume loop plots
- `simulation_config.yaml` - Complete configuration used
- `statistics.yaml` - Simulation performance statistics

## API Reference

### Main Classes

- `CardiacSimulation` - Main simulation runner
- `Config` - Configuration management
- `EPDriver` - Electrophysiology model
- `VentricularVolumeLogger` - LV volume tracking
- `AtrialVolumeLogger` - LA volume tracking
- `AorticWK2` - Aortic Windkessel model

### Key Methods

- `CardiacSimulation.run()` - Execute simulation
- `CardiacSimulation.save_results()` - Save all outputs
- `CardiacSimulation.plot_results()` - Generate plots
- `Config.from_yaml()` - Load configuration from file
- `Config.validate()` - Check configuration validity

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cardiac-sim.git
cd cardiac-sim

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{PumPy,
  title = {Cardiac Simulation Package :D},
  author = {Lez},
  year = {2025},
  url = {https://github.com/yourusername/cardiac-sim}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built on FEniCS/DOLFINx for finite element computations
- Inspired by cardiac modeling research community
- Thanks to contributors and beta testers

## Support

- **Documentation**: https://cardiac-sim.readthedocs.io/
- **Issues**: https://github.com/yourusername/cardiac-sim/issues
- **Discussions**: https://github.com/yourusername/cardiac-sim/discussions

## Changelog

### v0.1.0 (2025-08-22)
- Initial release
- Basic LV/LA mechanics simulation
- Volume tracking and PV loops
- Configuration system
- Electrophysiology models
- Circulation models









How this gives you a full (left) heart cycle

LA receives blood from pulmonary veins (constant ~12 mmHg source through a small resistance), LV ejects into an aortic Windkessel that shapes afterload.

Valves open/close automatically via the resistance switch (open vs closed) based on pressure differences (ΔP).

Coupling: each time step, the 0D model proposes pressures; 3D solves and returns volumes; 0D updates its state, and you iterate once or twice for consistency.

Outputs: time series of P_LA, P_LV, P_AO, V_LA, V_LV, Q_mitral, Q_aortic → you can plot PV-loops and flow curves immediately.

What you’ll need to set for your meshes

la_bc, la_endo, la_pfacets and lv_bc, lv_endo, lv_pfacets must match your facet tags:

bc_tags: fixed boundary facets,

endocardial_tags: inner surface of chamber for volume integral,

pressure_tags: where pressure traction is applied (usually same as endocardium).

If your LA model uses two inlet patches (e.g., 20 and 50), keep both in *_pfacets and *_endo.

Tuning knobs (and what they do)

Valve resistances (Rm_open/Rm_closed, Ra_open/Ra_closed) → regulates when/how much each valve flows/leaks.

Aortic C, R in AorticWindkessel2 → shapes aortic pressure and dicrotic decay.

Elastance ranges (already in pressures.py) → control peak pressures and timing.

picard_iters (usually 2–3) → stronger 0D–3D consistency; raise if you see drift between mechanical and 0D volumes.

Nice immediate add-ons (still “code-first”)

Log and plot PV-loop for LV: (V_lv, P_lv).

Export VTX/XDMF each step for ParaView; keep GIF optional.

If you want the right heart later: duplicate the pattern (RA↔RV) with pulmonary Windkessel and tricuspid/pulmonic valves; then connect the two 0D circuits (systemic ↔ pulmonary).

If you want, I can fold these files into your current tree and add a tiny PV-loop plotting script, but this is already everything you need to run a full left-heart cycle end-to-end with your existing mechanics.





BCS:
LV (Left Ventricle)

Cell tag (info only): 1 = myocardium

Outer surface: 10 = epicardium → free (no pressure, no Dirichlet)

Inner surface: 20 = endocardium → pressure & volume integral

Base / annulus: 50 = basal plane → Dirichlet anchor (fixed)

Use:

lv_bc = (50,)

lv_endo = (20,)

lv_pfacets = (20,)



LA (Left Atrium)

Cell tag (info only): 1 = myocardium

Inner wall: 10 = endocardium → pressure & volume integral

Pulmonary veins: 20 = PV Right, 50 = PV Left → Dirichlet anchors (rings)

Outer surface: 30 = epicardium → free

Mitral annulus: 40 = mitral_valve_opening → Dirichlet anchor (ring)

Use:

la_bc = (40) (pin PV rings + MV ring; this is common and avoids over-constraining the wall)

la_endo = (10,)

la_pfacets = (10,)



