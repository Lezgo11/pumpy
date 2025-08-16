# leftheart.py
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem, mesh as dmesh
from dolfinx.plot import vtk_mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
import pyvista as pv
import matplotlib.pyplot as plt
from petsc4py import PETSc



import pandas as pd
from materials import define_neo_hookean, define_holzapfel_ogden
from utils import setup_function_space
from fibers import solve_transmural_phi, build_fiber_field
from pressures import atrial_elastance_SI, ventricular_elastance_SI, mmHg_per_ml_to_Pa_per_m3

# -------------------------
# Helper classes & fns
# -------------------------
class Windkessel3:
    """
    -- Windkessel with realistic-ish SI params --
    3-element Windkessel with SI units:
      R1, R2 in Pa·s/m^3, C in m^3/Pa, Pa in Pa
    Recommended ballpark:
      R1 ~ 5e7 Pa·s/m^3   (~0.375 mmHg·s/ml)
      R2 ~ 1.5e8 Pa·s/m^3 (~1.125 mmHg·s/ml)
      C  ~ 1e-8 m^3/Pa    (~0.75 ml/mmHg total arterial compliance)
    """
    def __init__(self, R1=5e7, C=1e-8, R2=1.5e8, Pa0_mmHg=80.0):
        """Simple 3-element Windkessel: R1 - (C || R2)"""
        self.R1 = R1
        self.C = C
        self.R2 = R2
        self.Pa = Pa0_mmHg * 133.322  # initial aortic pressure

    def step(self, dt, Q_in):
        """
        Explicit Euler:
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
    def __init__(self, mesh_path, bc_tags, pressure_tags, endo_tag, name="chamber",
                 use_holzapfel=False, epi_tag=None, fiber_angles=(-60.0, 60.0)):
        self.name = name
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, gdim=3
        )
        self.V = setup_function_space(self.domain)
        self.bcs = self._make_dirichlet_bcs(bc_tags)

        # Material: either Neo-Hookean or Holzapfel–Ogden with fibers
        if use_holzapfel:
            assert epi_tag is not None, "epi_tag required when use_holzapfel=True"
            # transmural scalar and fiber field
            phi = solve_transmural_phi(self.domain, self.facet_tags, epi_tag=epi_tag, endo_tag=endo_tag)
            F_fiber = build_fiber_field(self.domain, phi,
                                        alpha_epi_deg=fiber_angles[0], alpha_endo_deg=fiber_angles[1])
            # HO params (tune as needed)
            mu_iso     = fem.Constant(self.domain, PETSc.ScalarType(3e4))    # Pa
            kappa_vol  = fem.Constant(self.domain, PETSc.ScalarType(1e7))    # Pa (bulk penalty)
            k1         = fem.Constant(self.domain, PETSc.ScalarType(4e5))    # Pa
            k2         = fem.Constant(self.domain, PETSc.ScalarType(8.0))    # -
            self.u, self.v, self.Pi, self.F_res, self.Jac = define_holzapfel_ogden(
                self.domain, self.V, mu_iso, kappa_vol, k1, k2, F_fiber
            )
        else:
            # baseline Neo-Hookean (as you had)
            E_base = 1e5
            nu = 0.3
            mu = fem.Constant(self.domain, PETSc.ScalarType(E_base / (2*(1+nu))))
            lmbda = fem.Constant(self.domain, PETSc.ScalarType(E_base*nu / ((1+nu)*(1-2*nu))))
            self.u, self.v, self.Pi, self.F_res, self.Jac = define_neo_hookean(mu, lmbda, self.domain, self.V)

        self.solver = None
        self.endo_tag = endo_tag
        self.pressure_tags = pressure_tags
        self.mt = self.facet_tags
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.mt)

        # Initial cavity volume
        X = ufl.SpatialCoordinate(self.domain)
        n = ufl.FacetNormal(self.domain)
        self.V0 = float(fem.assemble_scalar(fem.form((1.0/3.0) * ufl.dot(X, n) *
                                                     ufl.ds(self.endo_tag, subdomain_data=self.mt))))
        self.last_volume = self.V0

    def _make_dirichlet_bcs(self, bc_tags):
        if not bc_tags:
            return []
        fdim = self.domain.topology.dim - 1
        indices = np.concatenate([self.facet_tags.indices[self.facet_tags.values == t] for t in bc_tags])
        dofs = fem.locate_dofs_topological(self.V, fdim, indices)
        zero = np.zeros(3, dtype=PETSc.ScalarType)
        return [fem.dirichletbc(zero, dofs, self.V)]

    def apply_pressure_and_solve(self, pressure_value):
        traction = fem.Constant(self.domain, PETSc.ScalarType((0.0, 0.0, pressure_value)))
        L = sum(ufl.dot(traction, self.v) * self.ds(i) for i in self.pressure_tags) if self.pressure_tags else 0
        problem = NonlinearProblem(self.F_res - L, self.u, self.bcs, J=self.Jac)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.max_it = 40
        its, conv = solver.solve(self.u)
        return bool(conv), its

    def assemble_solver(self):
        problem = NonlinearProblem(self.F_res, self.u, self.bcs, J=self.Jac)
        self.solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-6
        self.solver.max_it = 30

    def compute_cavity_volume(self):
        n = ufl.FacetNormal(self.domain)
        form = ufl.dot(self.u, n) * ufl.ds(self.endo_tag, subdomain_data=self.mt)
        Vnow = self.V0 + float(fem.assemble_scalar(fem.form(form)))
        self.last_volume = Vnow
        return Vnow

    def save_vtk(self):
        topo, cell_types, geom = vtk_mesh(self.domain)
        grid = pv.UnstructuredGrid(topo, cell_types, geom)
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
# ---------------- Coupled simulation (use HO for LV) ----------------
def simulate_left_heart(
    la_mesh="mesh/hollow_sphere_LA.msh", la_bc=[40], la_inlets=[20,50], la_endo=10, la_epi=30,
    lv_mesh="mesh/idealized_LV.msh",  lv_bc=[50], lv_endo=20, lv_endo_pressure_tag=10, lv_epi=30,
    T=1.0, num_steps=60, dt=None,
    R_mitral_open_mmHg_s_ml=0.4, R_mitral_closed_mmHg_s_ml=1e6
):
    if dt is None:
        dt = T / num_steps

    # Convert valve resistances to SI
    def R_mmHg_s_ml_to_SI(R):  # (mmHg·s/ml) -> Pa·s/m^3
        return R * 133.322 / 1e-6
    Rm_open = R_mmHg_s_ml_to_SI(R_mitral_open_mmHg_s_ml)
    Rm_closed = R_mmHg_s_ml_to_SI(R_mitral_closed_mmHg_s_ml)

    # Chambers
    LA = Chamber(la_mesh, bc_tags=la_bc, pressure_tags=[], endo_tag=la_endo, name="LA",
                 use_holzapfel=False, epi_tag=la_epi)
    LV = Chamber(lv_mesh, bc_tags=lv_bc, pressure_tags=[lv_endo_pressure_tag], endo_tag=lv_endo, name="LV",
                 use_holzapfel=True, epi_tag=lv_epi, fiber_angles=(-60.0, 60.0))  # <-- HO + fibers ON

    # Windkessel
    wk = Windkessel3(R1=5e7, C=1.0e-8, R2=1.5e8, Pa0_mmHg=80.0)

    # Initial volumes
    Vla = LA.compute_cavity_volume()
    Vlv = LV.compute_cavity_volume()

    times = []
    P_la_hist, P_lv_hist, Pa_hist = [], [], []
    V_la_hist, V_lv_hist, Qm_hist = [], [], []

    # GIF
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif("figs/left_heart_coupled_HO.gif", fps=6)

    for n in range(num_steps):
        t = (n + 1) * dt

        # Elastances (SI)
        Ela = atrial_elastance_SI(t, T)
        Elv = ventricular_elastance_SI(t, T)

        # Pressures from elastance and volumes
        P_la = Ela * (Vla - LA.V0)
        P_lv = Elv * (Vlv - LV.V0)

        # Mitral valve flow
        Rm = Rm_open if P_la > P_lv else Rm_closed
        Qm = (P_la - P_lv) / Rm
        Vlv += Qm * dt
        Vla -= Qm * dt

        # Solve mechanics
        # LA (no explicit traction yet)
        LA.apply_pressure_and_solve(0.0)
        Vla = LA.compute_cavity_volume()

        # LV traction = P_lv on endocardium
        LV.apply_pressure_and_solve(P_lv)
        Vlv = LV.compute_cavity_volume()

        # Aortic valve + Windkessel
        if P_lv > wk.Pa:
            Qa = (P_lv - wk.Pa) / wk.R1
        else:
            Qa = 0.0
        Pa = wk.step(dt, Qa)

        # Record
        times.append(t)
        P_la_hist.append(P_la); P_lv_hist.append(P_lv); Pa_hist.append(Pa)
        V_la_hist.append(Vla);  V_lv_hist.append(Vlv); Qm_hist.append(Qm)

        # Frame
        grid_LA = LA.save_vtk()
        grid_LV = LV.save_vtk()
        plotter.clear()
        plotter.add_mesh(grid_LA, scalars="Magnitude", show_edges=False, opacity=0.6)
        plotter.add_mesh(grid_LV, scalars="Magnitude", show_edges=False, opacity=0.8)
        plotter.add_text(f"t={t:.2f}s  P_LA={P_la/133.322:.1f}mmHg  P_LV={P_lv/133.322:.1f}mmHg", font_size=10)
        plotter.write_frame()

        print(f"[{n+1}/{num_steps}] t={t:.3f}  P_la={P_la/133.322:.1f} mmHg  "
              f"P_lv={P_lv/133.322:.1f} mmHg  Pa={Pa/133.322:.1f} mmHg  Qm={Qm:.3e} m^3/s")

    plotter.close()  # Close GIF writer
    pv.close_all()  # Close any open PyVista plots
    # Plots
    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(times, np.array(P_lv_hist)/133.322, label="LV")
        plt.plot(times, np.array(P_la_hist)/133.322, label="LA")
        plt.plot(times, np.array(Pa_hist)/133.322,  label="Aorta (Windkessel)")
        plt.ylabel("Pressure [mmHg]"); plt.xlabel("Time [s]"); plt.legend(); plt.grid(True)
        plt.title("Left heart pressures (HO fibers in LV)")
        plt.savefig("figs/pressures_left_heart_HO.png"); plt.show()
        plt.figure(figsize=(10,5))
        plt.plot(times, V_lv_hist, label="LV volume")
        plt.plot(times, V_la_hist, label="LA volume")
        plt.ylabel("Volume [m^3]"); plt.xlabel("Time [s]"); plt.legend(); plt.grid(True)
        plt.title("Volumes")
        plt.savefig("figs/volumes_left_heart_HO.png"); plt.show()
        
        

    return dict(
        t=np.array(times),
        P_LA=np.array(P_la_hist), P_LV=np.array(P_lv_hist), P_Ao=np.array(Pa_hist),
        V_LA=np.array(V_la_hist), V_LV=np.array(V_lv_hist), Q_MV=np.array(Qm_hist)
    )

if __name__ == "__main__":
    results=simulate_left_heart()
    df = pd.DataFrame(results) # convert to dataframe
    df.to_csv("simulation_results.csv", index=False) # save
    