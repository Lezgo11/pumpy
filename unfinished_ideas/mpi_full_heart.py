# run_full_heart_mpi.py
import os, tempfile
import numpy as np
from mpi4py import MPI
from mechanics import MechanicsChamber
from heart0d import LeftHeart0D
from pressures import MMHG_TO_PA

def make_jit_cache():
    return {"timeout": 600, "cache_dir": os.path.join(tempfile.gettempdir(), f"pumpy_ffcx_fullheart")}

def run_full_left_heart(
    la_mesh="mesh/hollow_sphere_LA.msh",
    lv_mesh="mesh/idealized_LV.msh",
    T=1.0, steps=100,
    la_bc=(40,), la_endo=(10,), la_pfacets=(10,),
    lv_bc=(50,), lv_endo=(20,), lv_pfacets=(20,),
    material="neo_hookean",
    picard_iters=3, vol_tol=1e-8,
    relax=0.3,
    max_pressure=20000.0,
    output_file="heart_results.csv"
):
    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 prints startup info
    if rank == 0:
        print(f"Starting full left heart simulation on {size} MPI processes...")
    
    dt = T/steps
    jit = make_jit_cache()

    # 3D mechanics for LA & LV - these will be distributed automatically
    try:
        LA = MechanicsChamber(
            la_mesh, material=material,
            bc_tags=la_bc, endocardial_tags=la_endo, pressure_tags=la_pfacets,
            scale_geometry="auto",
            traction_mode="pressure",
            jit_options=jit,
        )

        LV = MechanicsChamber(
            lv_mesh, material=material,
            bc_tags=lv_bc, endocardial_tags=lv_endo, pressure_tags=lv_pfacets,
            scale_geometry="auto",
            traction_mode="pressure",
            jit_options=jit,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error initializing chambers: {e}")
        raise

    # 0D model - only rank 0 handles this to avoid conflicts
    if rank == 0:
        H0D = LeftHeart0D(V0_la=LA.V0, V0_lv=LV.V0)
        series = {"t":[], "P_la":[], "P_lv":[], "P_ao":[], "V_la":[], "V_lv":[], "Q_m":[], "Q_a":[]}
    else:
        H0D = None
        series = None

    # Initialize time stepping for chambers
    LA.begin_time_step(dt)
    LV.begin_time_step(dt)

    # Time loop
    t = 0.0
    failed_steps = 0
    max_failed_steps = 5
    
    for k in range(steps):
        if rank == 0:
            print(f"Step {k+1}/{steps}, t={t+dt:.4f}s", end="")
        
        try:
            # Only rank 0 does 0D computation and broadcasts results
            if rank == 0:
                P_la, P_lv, flows = H0D.step(t, dt, T)
                P_la = min(P_la, max_pressure)
                P_lv = min(P_lv, max_pressure)
                pressures = np.array([P_la, P_lv], dtype=np.float64)
                print(f", P_la={P_la/MMHG_TO_PA:.1f}, P_lv={P_lv/MMHG_TO_PA:.1f} mmHg", end="")
            else:
                pressures = np.zeros(2, dtype=np.float64)
            
            # Broadcast pressures to all processes
            comm.Bcast(pressures, root=0)
            P_la, P_lv = pressures[0], pressures[1]
            
            # Fixed-point iteration (all processes participate in 3D solve)
            if rank == 0:
                V_la_prev, V_lv_prev = H0D.V_la, H0D.V_lv
            
            converged_picard = False
            for picard_it in range(picard_iters):
                try:
                    # All processes solve their part of the 3D problem
                    LA.solve_with_pressure(P_la, load_splits=4, max_it=10)
                    LV.solve_with_pressure(P_lv, load_splits=4, max_it=10)
                    
                    # Measure volumes (automatically handles MPI reduction)
                    V_la_mech = LA.cavity_volume()
                    V_lv_mech = LV.cavity_volume()
                    
                    if rank == 0:
                        print(f", V_la={V_la_mech*1e6:.1f}mL, V_lv={V_lv_mech*1e6:.1f}mL", end="")

                        # Update 0D volumes
                        H0D.V_la = (1 - relax) * H0D.V_la + relax * V_la_mech
                        H0D.V_lv = (1 - relax) * H0D.V_lv + relax * V_lv_mech

                        # Recompute pressures
                        P_la, P_lv = H0D.pressures_from_elastance(t+dt, T)
                        P_la = min(P_la, max_pressure)
                        P_lv = min(P_lv, max_pressure)

                        # Check convergence
                        vol_error_la = abs(V_la_mech - V_la_prev)
                        vol_error_lv = abs(V_lv_mech - V_lv_prev)
                        
                        if vol_error_la < vol_tol and vol_error_lv < vol_tol:
                            converged_picard = True
                        else:
                            V_la_prev, V_lv_prev = V_la_mech, V_lv_mech
                            
                        # Prepare pressures for next iteration
                        pressures = np.array([P_la, P_lv], dtype=np.float64)
                    
                    # Broadcast convergence status and updated pressures
                    converged_picard = comm.bcast(converged_picard, root=0)
                    if not converged_picard and picard_it < picard_iters - 1:
                        comm.Bcast(pressures, root=0)
                        P_la, P_lv = pressures[0], pressures[1]
                    else:
                        break
                        
                except RuntimeError as e:
                    if rank == 0:
                        print(f" - Picard iter {picard_it} failed: {e}")
                    if picard_it == picard_iters - 1:
                        raise
                    # Try with reduced pressures
                    P_la *= 0.7
                    P_lv *= 0.7
                    continue

            # Only rank 0 records results
            if rank == 0:
                if not converged_picard:
                    print(f" - Picard not converged after {picard_iters} iterations")

                # Record results
                series["t"].append(t+dt)
                series["P_la"].append(P_la)
                series["P_lv"].append(P_lv)
                series["P_ao"].append(H0D.wk.P)
                series["V_la"].append(H0D.V_la)
                series["V_lv"].append(H0D.V_lv)
                series["Q_m"].append(flows["Q_m"])
                series["Q_a"].append(flows["Q_a"])
            
            # All processes update time stepping
            LA.begin_time_step(dt)
            LV.begin_time_step(dt)
            
            t += dt
            failed_steps = 0
            if rank == 0:
                print(" - OK")
            
        except Exception as e:
            failed_steps += 1
            if rank == 0:
                print(f" - FAILED: {e}")
            
            if failed_steps >= max_failed_steps:
                if rank == 0:
                    print(f"Too many failed steps ({failed_steps}), terminating simulation")
                break
            
            # Recovery logic (similar to before, but coordinated across processes)
            t += dt

    # Save results (only rank 0)
    if rank == 0 and series and len(series["t"]) > 0:
        save_results_to_csv(series, output_file)
        print(f"Results saved to {output_file}")

    return series if rank == 0 else None

def save_results_to_csv(series, filename="heart_results.csv"):
    """Save simulation results to CSV file."""
    import pandas as pd
    
    # Convert pressures from Pa to mmHg for easier interpretation
    df = pd.DataFrame({
        'time_s': series["t"],
        'P_la_mmHg': np.array(series["P_la"]) / MMHG_TO_PA,
        'P_lv_mmHg': np.array(series["P_lv"]) / MMHG_TO_PA,
        'P_ao_mmHg': np.array(series["P_ao"]) / MMHG_TO_PA,
        'V_la_mL': np.array(series["V_la"]) * 1e6,
        'V_lv_mL': np.array(series["V_lv"]) * 1e6,
        'Q_mitral_mL_s': np.array(series["Q_m"]) * 1e6,
        'Q_aortic_mL_s': np.array(series["Q_a"]) * 1e6,
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} data points to {filename}")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        print("Starting full left heart simulation...")
        data = run_full_left_heart(T=1.0, steps=3, output_file="heart_simulation.csv")
        if rank == 0:
            print(f"Simulation completed successfully!")
            if data:
                print(f"Final LV pressure: {data['P_lv'][-1]/MMHG_TO_PA:.1f} mmHg")
                print(f"Final Aortic pressure: {data['P_ao'][-1]/MMHG_TO_PA:.1f} mmHg")
                print(f"Final LV volume: {data['V_lv'][-1]*1e6:.1f} mL")
    except Exception as e:
        if rank == 0:
            print(f"Simulation failed: {e}")
            import traceback
            traceback.print_exc()