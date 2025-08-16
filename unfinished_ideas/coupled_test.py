# leftheart.py
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio
from dolfinx import fem, mesh as dmesh
from dolfinx.plot import vtk_mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd


# utils functions you already have
from utils import setup_function_space, define_neo_hookean

# -------------------------
# Helper classes & fns
# -------------------------
class Windkessel3:
    """Simple 3-element Windkessel: R1 - (C || R2)"""
    def __init__(self, R1=1e7, C=1e-6, R2=1e7, Pa0=1000.0):
        # user may tune these; units consistent with Pa and m^3/s
        self.R1 = R1
        self.C = C
        self.R2 = R2
        self.Pa = Pa0  # arterial pressure (Pa)

    def step(self, dt, Q_in):
        """
        Explicit Euler update:
        C dPa/dt = Q_in - Pa/R2
        where Q_in = (Plv - Pa) / R1 (but we compute from provided Q_in)
        """
        dPa = (Q_in - self.Pa / self.R2) / self.C
        self.Pa += dt * dPa
        return self.Pa


class Chamber:
    """
    Encapsulates a solid chamber (LA or LV) solved as nonlinear elasticity.
    - mesh_path: .msh file path
    - bc_tags: list of facet tags to apply Dirichlet 0 BC (e.g., mitral ring / basal)
    - pressure_tags: list of facet tags where Neumann (endocardial) pressure is applied
    - endo_tag: the facet tag corresponding to endocardium surface used to compute cavity volume
    """
    def __init__(self, mesh_path, bc_tags, pressure_tags, endo_tag, name="chamber"):
        self.current_pressure = 0.0  # Pa; will be ramped toward the target each step
        self.name = name
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, gdim=3
        )
        self.V = setup_function_space(self.domain)
        self.bcs = self._make_dirichlet_bcs(bc_tags)
        # material params (tune)
        E_base = 1e5
        nu = 0.3
        self.mu = fem.Constant(self.domain, PETSc.ScalarType(E_base / (2*(1+nu))))
        self.lmbda = fem.Constant(self.domain, PETSc.ScalarType(E_base*nu / ((1+nu)*(1-2*nu))))
        # neo-hookean setup
        self.u, self.v, self.Pi, self.F_res, self.Jac = define_neo_hookean(self.mu, self.lmbda, self.domain, self.V)
        self.solver = None
        self.dt = None
        self.facet_tags = self.facet_tags
        self.endo_tag = endo_tag
        self.pressure_tags = pressure_tags
        self.mt = self.facet_tags
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.mt)

        # compute V0 (initial cavity volume) via surface integral: V0 = 1/3 ∫ X·n dS
        X = ufl.SpatialCoordinate(self.domain)
        n = ufl.FacetNormal(self.domain)
        one_third = PETSc.ScalarType(1.0/3.0)
        form_V0 = one_third * ufl.dot(X, n) * ufl.ds(self.endo_tag, subdomain_data=self.mt)
        self.V0 = float(fem.assemble_scalar(fem.form(form_V0)))
        # storage for last assembled u (to compute volume change)
        self.last_volume = self.V0

    def _make_dirichlet_bcs(self, bc_tags):
        # collect facet indices and build BC
        if not bc_tags:
            return []
        fdim = self.domain.topology.dim - 1
        indices = np.concatenate([self.facet_tags.indices[self.facet_tags.values == t] for t in bc_tags])
        dofs = fem.locate_dofs_topological(self.V, fdim, indices)
        zero = np.zeros(3, dtype=PETSc.ScalarType)
        return [fem.dirichletbc(zero, dofs, self.V)]

    def assemble_solver(self):
        problem = NonlinearProblem(self.F_res, self.u, self.bcs, J=self.Jac)
        self.solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-6
        self.solver.max_it = 50


    def apply_pressure_and_solve(self, P_target, max_inc=2.0e3, min_inc=1.0e2, max_halves=6):
        """
        Robust nonlinear solve with adaptive load increments and PETSc line-search.
        - P_target: desired endocardial pressure for this (global) time step [Pa]
        - max_inc:  maximum pressure increment per local substep [Pa]
        - min_inc:  minimum increment before giving up [Pa]
        - max_halves: how many times we’re willing to halve the increment upon failure

        Returns (converged: bool, total_newton_its: int)
        """
        # Helper to build the traction form for a given pressure value
        def traction_form(P):
            traction_vec = fem.Constant(self.domain, PETSc.ScalarType((0.0, 0.0, P)))
            return sum(ufl.dot(traction_vec, self.v) * self.ds(tag) for tag in self.pressure_tags) if self.pressure_tags else 0

        total_its = 0
        P_applied = float(self.current_pressure)
        P_target = float(P_target)

        # Early exit if nothing to do
        if abs(P_target - P_applied) < 1e-12:
            return True, 0

        direction = 1.0 if P_target > P_applied else -1.0
        remaining = abs(P_target - P_applied)
        inc = min(max_inc, remaining)

        while remaining > 1e-12:
            # Propose an increment
            inc = min(inc, remaining)
            P_trial = P_applied + direction * inc

            # Build problem at this substep
            L = traction_form(P_trial)
            problem = NonlinearProblem(self.F_res - L, self.u, self.bcs, J=self.Jac)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "residual"
            solver.rtol = 1e-6
            solver.atol = 1e-8
            solver.max_it = 50

            # PETSc SNES + KSP/PC options with a prefix
            solver.set_options_prefix("ns_")
            opts = PETSc.Options()
            # Backtracking line search for robustness
            #opts["ns_snes_linesearch_type"] = "bt"
            # Direct solve per Newton step (robust; change to ilu/gamg later for speed)
            opts["ns_ksp_type"] = "gmres"
            opts["ns_pc_type"] = "gamg"
            opts["ns_mg_levels_ksp_type"] = "chebyshev"
            opts["ns_mg_levels_pc_type"] = "jacobi"
            opts["ns_ksp_rtol"] = 1e-6
            #opts["ns_ksp_max_it"] = 200
            
            
            #opts["ns_pc_factor_mat_solver_type"] = "mumps"
            # Optional: monitor (comment out to silence)
            # opts["ns_snes_monitor"] = None
            # opts["ns_ksp_monitor"] = None

            # Try to solve; if it fails, halve the increment and retry
            success = False
            halves = 0
            while halves <= max_halves:
                try:
                    nits, converged = solver.solve(self.u)
                    total_its += nits
                    if converged:
                        success = True
                        break
                except RuntimeError:
                    converged = False

                # Didn’t converge: back off the increment
                inc *= 0.5
                halves += 1
                if inc < min_inc:
                    break

                # Rebuild problem with smaller inc
                P_trial = P_applied + direction * inc
                L = traction_form(P_trial)
                problem = NonlinearProblem(self.F_res - L, self.u, self.bcs, J=self.Jac)
                solver = NewtonSolver(MPI.COMM_WORLD, problem)
                solver.convergence_criterion = "residual"
                solver.rtol = 1e-6
                solver.atol = 1e-8
                solver.max_it = 30
                solver.set_options_prefix("ns_")
                # reapply PETSc options
                opts = PETSc.Options()
                opts["ns_snes_linesearch_type"] = "bt"
                opts["ns_ksp_type"] = "preonly"
                opts["ns_pc_type"] = "lu"
                opts["ns_pc_factor_mat_solver_type"] = "mumps"

            if not success:
                # Give up this global step
                return False, total_its

            # Accept the substep and move on
            P_applied = P_trial
            remaining = abs(P_target - P_applied)
            # Optionally increase increment again (mild aggressiveness)
            inc = min(max_inc, inc * 1.5)

        # Store the new applied pressure for next global step
        self.current_pressure = P_applied
        return True, total_its

        

    def compute_cavity_volume(self):
        """
        Use V = V0 + ∫_{endo} u·n dS  (first-order accurate for moderate deformations)
        """
        n = ufl.FacetNormal(self.domain)
        form = ufl.dot(self.u, n) * ufl.ds(self.endo_tag, subdomain_data=self.mt)
        deltaV = float(fem.assemble_scalar(fem.form(form)))
        Vnow = self.V0 + deltaV
        self.last_volume = Vnow
        return Vnow

    def save_vtk(self):
        # quick export for visualization
        topology, cell_types, geometry = vtk_mesh(self.domain)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        u_vals = self.u.x.array.reshape(-1, 3)
        grid.points += u_vals
        grid.point_data["Displacement"] = u_vals
        grid.point_data["Magnitude"] = np.linalg.norm(u_vals, axis=1)
        return grid

# -------------------------
# Elastance functions
# -------------------------
def atrial_elastance(t, T, E_min=1e3, E_max=5e3):
    # simple sinusoidal activation (user tune)
    tau = (t % T) / T
    e = np.sin(2*np.pi*t/T)**2  # simple shape
    return E_min + (E_max - E_min) * e

def ventricular_elastance(t, T, E_min=1e4, E_max=2e5):
    tau = (t % T) / T
    # peak during systole
    e = 0.5*(1 - np.cos(2*np.pi*t/T)) if tau < 0.5 else 0.0
    return E_min + (E_max - E_min) * e

# -------------------------
# Coupling loop
# -------------------------
def simulate_left_heart(
    la_mesh="mesh/hollow_sphere_LA.msh", la_bc=[40], la_inlets=[20,50], la_endo=10,
    lv_mesh="mesh/idealized_LV.msh", lv_bc=[50], lv_endo=20, lv_endo_pressure_tag=10,
    T=1.0, num_steps=20, dt=None, R_mitral_open=1e6, R_mitral_closed=1e12
):
    if dt is None:
        dt = T / num_steps

    # instantiate chambers
    LA = Chamber(la_mesh, bc_tags=la_bc, pressure_tags=[], endo_tag=la_endo, name="LA")
    LV = Chamber(lv_mesh, bc_tags=lv_bc, pressure_tags=[lv_endo_pressure_tag], endo_tag=lv_endo, name="LV")

    # assemble solvers
    LA.assemble_solver()
    LV.assemble_solver()

    # windkessel for aorta
    wk = Windkessel3(R1=1e7, C=1e-6, R2=1e7, Pa0=1e3)

    # initial volumes
    Vla = LA.compute_cavity_volume()
    Vlv = LV.compute_cavity_volume()

    times = []
    data = {"t": [], "V_LA": [], "V_LV": [], "P_LA": [], "P_LV": [], "P_aorta": [], "Q_mitral": []}

    # initial pressures via elastance
    P_la = atrial_elastance(0.0, T) * (Vla - LA.V0)
    P_lv = ventricular_elastance(0.0, T) * (Vlv - LV.V0)

    # valve state
    mitral_open = True

    # PyVista gif setup (shared)
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif("figs/left_heart_coupled.gif", fps=5)

    for n in range(num_steps):
        t = (n+1)*dt

        # ---------------- LA step ----------------
        # For now, LA pressure computed from elastance and LA volume (drives inlet flow)
        P_la = atrial_elastance(t, T) * (Vla - LA.V0)

        # solve LA mechanics with pulmonary vein traction (optional)
        # Here we skip external traction and keep BCs fixed; advanced: add PV inflow traction
        converged_la, its_la = LA.apply_pressure_and_solve(0.0)

        # compute new LA volume
        Vla = LA.compute_cavity_volume()

        # ---------------- valve logic ----------------
        # Mitral valve opens if P_la > P_lv (simple)
        P_lv = ventricular_elastance(t, T) * (Vlv - LV.V0)
        if P_la > P_lv:
            mitral_open = True
            R_mit = R_mitral_open
        else:
            mitral_open = False
            R_mit = R_mitral_closed

        # mitral flow (positive into LV)
        Q_m = (P_la - P_lv) / R_mit

        # update LV volume explicitly from mitral flow
        Vlv += Q_m * dt

        # ---------------- LV step ----------------
        # compute LV pressure from elastance given updated volume
        P_lv = ventricular_elastance(t, T) * (Vlv - LV.V0)

        # Apply LV endocardial pressure (Neumann) and solve LV mechanics
        converged_lv, its_lv = LV.apply_pressure_and_solve(P_lv)

        # compute new LV cavity volume (after mechanical deformation)
        Vlv = LV.compute_cavity_volume()

        # ---------------- Aortic / Windkessel ----------------
        # Simple logic: if P_lv > Pa (aortic pressure), AV opens and flow to aorta occurs
        if P_lv > wk.Pa:
            # flow through valve approximated by (P_lv - Pa)/R1
            Q_a = (P_lv - wk.Pa) / wk.R1
        else:
            Q_a = 0.0
        Pa = wk.step(dt, Q_a)

        # ---------------- Record data & visualize ----------------
        times.append(t)
        data["t"].append(t)
        data["V_LA"].append(Vla)
        data["V_LV"].append(Vlv)
        data["P_LA"].append(P_la)
        data["P_LV"].append(P_lv)
        data["P_aorta"].append(Pa)
        data["Q_mitral"].append(Q_m)

        # Visualization: combine LA+LV grids into plotter frame
        grid_LA = LA.save_vtk()
        grid_LV = LV.save_vtk()
        plotter.clear()
        plotter.add_mesh(grid_LA, scalars="Magnitude", show_edges=False, opacity=0.6)
        plotter.add_mesh(grid_LV, scalars="Magnitude", show_edges=False, opacity=0.6)
        plotter.add_text(f"t={t:.3f}s P_LA={P_la:.1f} Pa P_LV={P_lv:.1f} Pa", position="upper_left")
        plotter.write_frame()

        print(f"t={t:.3f}s Vla={Vla:.6e} Vlv={Vlv:.6e} P_la={P_la:.1f} P_lv={P_lv:.1f} Q_m={Q_m:.3e} Pa={Pa:.1f}")

    #plotter.close()

    # ---------------- Postprocess plots ----------------
    if MPI.COMM_WORLD.rank == 0:
        plt.figure(figsize=(10,6))
        plt.plot(data["t"], np.array(data["P_LV"])/133.322, label="P_LV (mmHg)")
        plt.plot(data["t"], np.array(data["P_LA"])/133.322, label="P_LA (mmHg)")
        plt.plot(data["t"], np.array(data["P_aorta"])/133.322, label="Paorta (mmHg)")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Pressure [mmHg]")
        plt.title("Left heart pressures")
        plt.grid(True)
        plt.savefig("figs/pressures_left_heart.png")
        plt.show()

    return data

if __name__ == "__main__":

    results=simulate_left_heart()
    df = pd.DataFrame(results) # convert to dataframe
    df.to_csv("simulation_results.csv", index=False) # save
