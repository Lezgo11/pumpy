"""
Basic cardiac simulation example.

This example demonstrates how to set up and run a basic cardiac simulation
using the cardiac_sim package.
"""

import sys
from pathlib import Path

# Add the package to the path (if not installed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cardiac_sim import CardiacSimulation, Config
from cardiac_sim.simulation.config import (
    PhysiologyConfig, ChambersConfig, SimulationConfig
)


def main():
    """Run a basic cardiac simulation."""
    
    print("Setting up basic cardiac simulation...")
    
    # Create configuration
    config = Config()
    
    # Customize simulation parameters
    config.simulation.dt = 0.1
    config.simulation.cycles = 2
    config.enable_volume = True
    config.enable_electro = False
    config.enable_physio_tuner = False
    
    # Set active chambers
    config.chambers.active_chambers = ["LV", "LA"]
    
    # Customize physiology
    config.physiology.hr_bpm = 75.0
    config.physiology.sbp_mmhg = 120.0
    config.physiology.dbp_mmhg = 80.0
    
    # Validate configuration
    warnings = config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Create and run simulation
    sim = CardiacSimulation(config)
    
    print("\nRunning simulation...")
    results = sim.run(verbose=True)
    
    # Save results
    print("\nSaving results...")
    sim.save_results(results, output_dir="basic_simulation_output")
    
    # Generate plots
    print("Generating plots...")
    sim.plot_results(results, output_dir="basic_simulation_output")
    
    print("\nSimulation complete!")
    print(f"Results saved to: basic_simulation_output/")
    
    # Print some statistics
    stats = results["statistics"]
    print(f"\nStatistics:")
    print(f"  Total time: {stats['elapsed_time_s']:.2f} s")
    print(f"  Steps: {stats['total_steps']}")
    print(f"  Convergence rate: {stats['convergence_rate']*100:.1f}%")


if __name__ == "__main__":
    main()