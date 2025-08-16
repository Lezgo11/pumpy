# leftheart.py
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import pyvista as pv
from dolfinx.plot import vtk_mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem


from utils import (
    compute_fiber_angle,
    setup_function_space,
)

from materials import (
    define_holzapfel_ogden,
    define_neo_hookean,
)
from fibers import (
    solve_transmural_phi,
    build_fiber_field,
    define_holzapfel_ogden,
)
from pressures import (
    mmHg_per_ml_to_Pa_per_m3,
    smooth_cos_window,
    atrial_elastance_SI,
    ventricular_elastance_SI,
    atrial_pressure,
    ventricular_pressure,
)

def simulate_chamber(
    mesh_path: str,
    bc_tags: list[int],
    pressure_tags: list[int],
    pressure_function,
    material_model: str = "neo_hookean",   # "neo_hookean" | "holzapfel_ogden"
    fiber_epi_tag: int | None = None,      # needed if you want fibers
    fiber_endo_tag: int | None = None,
    T: float = 1.0,
    num_steps: int = 20,
    E_base: float = 1e5,
    nu: float = 0.3,
    mu_iso: float = 5e4,
    kappa_vol: float = 5e6,
    k1: float = 2.0e5,
    k2: float = 5.0,
):
    # ---- chamber name from path ----
    if mesh_path.endswith("hollow_sphere_LA.msh"):
        chamber_name = "LA"
        default_pressure = atrial_pressure
    elif mesh_path.endswith("idealized_LV.msh"):
        chamber_name = "LV"
        default_pressure = ventricular_pressure
    else:
        raise ValueError("Unsupported mesh. Use LA or LV example meshes.")

    # Allow caller override but keep a sensible default per chamber
    if pressure_function is None:
        pressure_function = default_pressure

    # ---- Mesh & space ----
    comm = MPI.COMM_WORLD
    domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, comm, gdim=3)
    V = setup_function_space(domain)

    # ---- Dirichlet BCs (fixed boundary) ----
    fdim = domain.topology.dim - 1
    if len(bc_tags) == 0:
        raise ValueError("bc_tags must contain at least one facet tag.")
    bc_facets = np.concatenate([facet_tags.indices[facet_tags.values == tag] for tag in bc_tags])
    bc_dofs = fem.locate_dofs_topological(V, fdim, bc_facets)
    bc = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), bc_dofs, V)

    # ---- Material model ----
    if material_model == "neo_hookean":
        mu = fem.Constant(domain, PETSc.ScalarType(E_base / (2.0 * (1.0 + nu))))
        lmbda = fem.Constant(domain, PETSc.ScalarType(E_base * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
        u, v, _, F_res, J = define_neo_hookean(mu, lmbda, domain, V)
    elif material_model == "holzapfel_ogden":
        if fiber_epi_tag is None or fiber_endo_tag is None:
            raise ValueError("Provide fiber epi/endo facet tags for HO material.")
        # transmural coordinate and fiber field
        phi = solve_transmural_phi(domain, facet_tags, fiber_epi_tag, fiber_endo_tag)
        f_expr = build_fiber_field(domain, phi, -60.0, 60.0, ref_axis=(0.0, 0.0, 1.0))
        u, v, _, F_res, J = define_holzapfel_ogden(domain, V, mu_iso, kappa_vol, k1, k2, f_expr)
    else:
        raise ValueError("Unknown material_model.")

    # ---- Measures ----
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)

    # ---- Time loop params ----
    dt = T / num_steps
    times, disp_norms = [], []

    # ---- PyVista (rank 0 only) ----
    pv.OFF_SCREEN = True
    do_viz = (comm.rank == 0)
    if do_viz:
        plotter = pv.Plotter(off_screen=True)
        plotter.open_gif(f"figs/deformation_{chamber_name}.gif", fps=5)
        topology, cell_types, geometry = vtk_mesh(domain)

    # ---- Time stepping ----
    for k in range(num_steps):
        t = (k + 1) * dt
        p = float(pressure_function(t, T))  # from utils
        # Neumann traction: -p * n
        L = sum((-p) * ufl.inner(v, n) * ds(tag) for tag in pressure_tags)

        problem = NonlinearProblem(F_res - L, u, [bc], J=J)
        solver = NewtonSolver(comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        its, converged = solver.solve(u)
        if comm.rank == 0:
            print(f"[{chamber_name}] t={t:.3f}: converged={converged} in {its} iters, p={p:.2e} Pa")

        # monitor ‖u‖_L2
        disp_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)),
            op=MPI.SUM))
        times.append(t)
        disp_norms.append(disp_norm)

        # ---- Visualization (rank 0) ----
        if do_viz and converged:
            grid = pv.UnstructuredGrid(topology, cell_types, geometry.copy())

            # safer component-wise collapse to vertex order
            u_components = []
            for i in range(3):
                Vi, dofs = V.sub(i).collapse()          # space, then mapping
                ui = fem.Function(Vi)                   # make a function on the collapsed space
                ui.x.array[:] = u.x.array[dofs]         # pull component dofs from the parent vector
                u_components.append(ui.x.array.copy())

            u_vals = np.vstack(u_components).T  # shape (n_verts, 3) for P1


            grid.points += u_vals
            grid.point_data["Displacement"] = u_vals
            grid.point_data["Magnitude"] = np.linalg.norm(u_vals, axis=1)
            plotter.add_mesh(grid, scalars="Magnitude", clim=[0, 0.1], cmap="viridis", show_edges=True)
            plotter.add_text(f"Time: {t:.2f}s  |  p={p:.2e} Pa", position="upper_edge")
            plotter.write_frame()
            plotter.clear()

    if do_viz:
        plotter.close()
        # Time series plot
        plt.figure(figsize=(9, 4))
        plt.plot(times, disp_norms, lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("‖u‖$_{L2}$")
        plt.title(f"Displacement over time – {chamber_name}")
        plt.grid(True)
        out_png = f"figs/displacement_{chamber_name}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()

if __name__ == "__main__":
    # Example: LA with atrial pressure curve on pulmonary vein tags (20, 50)
    simulate_chamber(
        mesh_path="mesh/hollow_sphere_LA.msh",
        bc_tags=[40],
        pressure_tags=[20, 50],
        pressure_function=atrial_pressure,
        material_model="neo_hookean",
    )
