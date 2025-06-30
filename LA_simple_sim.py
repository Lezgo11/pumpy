import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io,nls
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import pyvista as pv
from dolfinx.plot import vtk_mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from utils import compute_fiber_angle, setup_function_space, define_neo_hookean, atrial_pressure

#Remember to always activate fenicsx first: conda activate fenicsx

#------ Mesh and Function Space Setup ------
# Create unit cube mesh with hexahedral elements
domain, cell_tags, facet_tags = gmshio.read_from_msh("mesh/hollow_sphere_LA.msh", MPI.COMM_WORLD, gdim=3)
# Define the element type and function space
V = setup_function_space(domain)

#------ Boundary Conditions ------
fdim = domain.topology.dim - 1
boundary_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.indices[facet_tags.values == 40]) # Mitral Valve opening(Physical Group tag = 40)
zero_vector = np.zeros(3, dtype=PETSc.ScalarType)
bc = fem.dirichletbc(zero_vector, boundary_dofs, V)# Apply zero displacement BC on the up face

#------ Material parameters ------
E_base = 1e5
nu = 0.3
# Enhanced Material Parameters (nonlinear and anisotropic)
mu = fem.Constant(domain, PETSc.ScalarType(E_base / (2*(1+nu)))) #mu = E / (2*(1 + nu))
lmbda = fem.Constant(domain, PETSc.ScalarType(E_base*nu / ((1+nu)*(1-2*nu))))#lmbda = E*nu / ((1 + nu)*(1 - 2*nu))
# Fiber orientation for anisotropy
fiber_direction = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0, 0.0)))

# ------ Define functions ------ 
# Neo-Hookean model
u,v,Pi,F_res,J = define_neo_hookean(mu, lmbda,domain,V)
right_facets = facet_tags.indices[facet_tags.values == 20] # Over pulmonary valves R and L
mt = facet_tags
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

# ------ PyVista setup ------
pv.OFF_SCREEN = True  # For headless rendering if needed
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("deformation_idealizedLA.gif", fps=5)

#------ Simulation parameters ------
# Time parameters
T = 1.0             # final time
num_steps = 20      # number of time steps
dt = T / num_steps  # time step size
# Time stepping
displacement_norms = []
times = []

# ------ Time stepping loop ------
for n in range(num_steps):
    t = (n+1)*dt
    p_ext = atrial_pressure(t,T)
    print("Pressure value at t =", t, "is", atrial_pressure(t,T))
    traction = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, p_ext)))
    L = sum(ufl.dot(traction, v) * ds(i) for i in [20, 50]) # Pulmonary veins R+L

    # Create new problem with updated residual (F_res - L)
    problem = NonlinearProblem(F_res - L, u, [bc], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    num_its, converged = solver.solve(u)
    print(f"Converged: {converged}, iterations: {num_its}")
    
    if converged:
        # Compute displacement norm
        disp_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)), 
            op=MPI.SUM))
        
        displacement_norms.append(disp_norm)
        times.append(t)
        
        # Visualization
        topology, cell_types, geometry = vtk_mesh(domain)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        
        # Warp by displacement
        u_vals = u.x.array.reshape(-1, 3)
        grid.points += u_vals
        
        # Add fields
        grid.point_data["Displacement"] = u_vals
        grid.point_data["Magnitude"] = np.linalg.norm(u_vals, axis=1)
        
        plotter.add_mesh(grid, scalars="Magnitude", clim=[0, 0.1], 
                        cmap="viridis", show_edges=True)
        plotter.add_text(f"Time: {t:.2f}s", position="upper_edge")
        plotter.write_frame()
        plotter.clear()
        
        print(f"Step {n+1}, Time {t:.2f}, Displacement Norm = {disp_norm:.4e}")
plotter.close()

# Plot results
if domain.comm.rank == 0:
    plt.figure(figsize=(10, 5))
    plt.plot(times, displacement_norms)
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement Norm [L2]")
    plt.title("Displacement Response Over Time")
    plt.grid(True)
    plt.savefig("displacement_IdealizedLV.png")
    plt.show()