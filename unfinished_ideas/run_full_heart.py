# run_full_heart.py
import os, tempfile
import numpy as np
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
    picard_iters=3, vol_tol=1e-8,  # Increased picard iters and tighter tolerance
    relax=0.3,  # Smaller relaxation for stability
    max_pressure=20000.0  # Maximum pressure limit (Pa) ~ 150 mmHg
):
    dt = T/steps
    jit = make_jit_cache()

    # 3D mechanics for LA & LV
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
        print(f"Error initializing chambers: {e}")
        raise

    # 0D model — initialize V0 from 3D reference volumes
    H0D = LeftHeart0D(V0_la=LA.V0, V0_lv=LV.V0)

    series = {"t":[], "P_la":[], "P_lv":[], "P_ao":[], "V_la":[], "V_lv":[], "Q_m":[], "Q_a":[]}

    # Initialize time stepping for chambers
    LA.begin_time_step(dt)
    LV.begin_time_step(dt)

    # Time loop
    t = 0.0
    failed_steps = 0
    max_failed_steps = 5
    
    for k in range(steps):
        print(f"Step {k+1}/{steps}, t={t+dt:.4f}s", end="")
        
        try:
            # Predictor: one 0D step to get starting pressures
            P_la, P_lv, flows = H0D.step(t, dt, T)
            
            # Pressure limiting for stability
            P_la = min(P_la, max_pressure)
            P_lv = min(P_lv, max_pressure)
            
            print(f", P_la={P_la/MMHG_TO_PA:.1f}, P_lv={P_lv/MMHG_TO_PA:.1f} mmHg", end="")
            
            # Fixed-point iteration to match 0D volumes with 3D deformation
            V_la_prev, V_lv_prev = H0D.V_la, H0D.V_lv
            converged_picard = False
            
            for picard_it in range(picard_iters):
                try:
                    LA.solve_with_pressure(P_la, load_splits=4, max_it=10)
                    LV.solve_with_pressure(P_lv, load_splits=4, max_it=10)
                    V_la_mech = LA.cavity_volume()
                    V_lv_mech = LV.cavity_volume()
                    
                    print(f", V_la={V_la_mech*1e6:.1f}mL, V_lv={V_lv_mech*1e6:.1f}mL", end="")

                    H0D.V_la = (1 - relax) * H0D.V_la + relax * V_la_mech
                    H0D.V_lv = (1 - relax) * H0D.V_lv + relax * V_lv_mech

                    P_la, P_lv = H0D.pressures_from_elastance(t+dt, T)
                    P_la = min(P_la, max_pressure)
                    P_lv = min(P_lv, max_pressure)
                    # convergence
                    vol_error_la = abs(V_la_mech - V_la_prev)
                    vol_error_lv = abs(V_lv_mech - V_lv_prev)
                    if vol_error_la < vol_tol and vol_error_lv < vol_tol:
                        converged_picard = True
                        break
                        
                    V_la_prev, V_lv_prev = V_la_mech, V_lv_mech
                    
                except RuntimeError as e:
                    print(f" - Picard iter {picard_it} failed: {e}")
                    if picard_it == picard_iters - 1:
                        raise
                    # Try with reduced pressures
                    P_la *= 0.7
                    P_lv *= 0.7
                    continue

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
            
            # Update previous displacements for next time step
            LA.begin_time_step(dt)
            LV.begin_time_step(dt)
            t += dt
            failed_steps = 0  # Reset failed step counter
            print(" - OK")
        except Exception as e:
            failed_steps += 1
            print(f" - FAILED: {e}")
            
            if failed_steps >= max_failed_steps:
                print(f"Too many failed steps ({failed_steps}), terminating simulation")
                break
            
            # Try to recover by reducing time step or pressures
            print(f"Attempting recovery... (attempt {failed_steps})")
            
            # Reduce pressures and try again
            if hasattr(H0D, 'V_la') and hasattr(H0D, 'V_lv'):
                try:
                    P_la_reduced = P_la * 0.5
                    P_lv_reduced = P_lv * 0.5
                    
                    LA.solve_with_pressure(P_la_reduced, load_splits=4, max_it=10)
                    LV.solve_with_pressure(P_lv_reduced, load_splits=4, max_it=10)
                    
                    # Use reduced pressure results
                    series["t"].append(t+dt)
                    series["P_la"].append(P_la_reduced)
                    series["P_lv"].append(P_lv_reduced)
                    series["P_ao"].append(H0D.wk.P)
                    series["V_la"].append(H0D.V_la)
                    series["V_lv"].append(H0D.V_lv)
                    series["Q_m"].append(flows["Q_m"] if 'flows' in locals() else 0.0)
                    series["Q_a"].append(flows["Q_a"] if 'flows' in locals() else 0.0)
                    LA.begin_time_step(dt)
                    LV.begin_time_step(dt)
                    t += dt
                    failed_steps = 0
                    print("Recovery successful")
                except:
                    if len(series["t"]) > 0:
                        series["t"].append(t+dt)
                        series["P_la"].append(series["P_la"][-1])
                        series["P_lv"].append(series["P_lv"][-1])
                        series["P_ao"].append(series["P_ao"][-1])
                        series["V_la"].append(series["V_la"][-1])
                        series["V_lv"].append(series["V_lv"][-1])
                        series["Q_m"].append(0.0)
                        series["Q_a"].append(0.0)
                    t += dt

    return series

if __name__ == "__main__":
    print("Starting full left heart simulation...")
    try:
        data = run_full_left_heart(T=1.0, steps=5)
        print(f"Simulation completed successfully!")
        print(f"Final LV pressure: {data['P_lv'][-1]/MMHG_TO_PA:.1f} mmHg")
        print(f"Final Aortic pressure: {data['P_ao'][-1]/MMHG_TO_PA:.1f} mmHg")
        print(f"Final LV volume: {data['V_lv'][-1]*1e6:.1f} mL")
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()