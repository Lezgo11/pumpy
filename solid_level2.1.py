import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io,nls
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import pyvista as pv
from dolfinx.plot import vtk_mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem

# Time parameters
T = 1.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size

# Create unit cube mesh with hexahedral elements
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, mesh.CellType.hexahedron)
# Define the element type and function space
element = basix.ufl.element(
    family="Lagrange",  # or "CG"
    cell="hexahedron",  # since your mesh is hexahedral
    degree=1,
    shape=(3,),  # Vector element for 3D displacement
    dtype=PETSc.ScalarType,
)
V = fem.functionspace(domain, element)


# Material parameters
E_base = 1e5
nu = 0.3
# Enhanced Material Parameters (nonlinear and anisotropic)
mu = fem.Constant(domain, PETSc.ScalarType(E_base / (2*(1+nu)))) #mu = E / (2*(1 + nu))
lmbda = fem.Constant(domain, PETSc.ScalarType(E_base*nu / ((1+nu)*(1-2*nu))))#lmbda = E*nu / ((1 + nu)*(1 - 2*nu))


# Fiber orientation for anisotropy
fiber_direction = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0, 0.0)))

# Boundary condition: fix left face (x=0)
def left_boundary(x):
    return np.isclose(x[0], 0.0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
zero_vector = np.zeros(3, dtype=PETSc.ScalarType)
bc = fem.dirichletbc(zero_vector, boundary_dofs, V)# Apply zero displacement BC on the left face

# Define functions
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# Kinematics
I = ufl.Identity(domain.geometry.dim)
F = I + ufl.grad(u)
C = F.T * F
J = ufl.det(F)

# Neo-Hookean model
psi = (mu/2)*(ufl.tr(C) - 3) - mu*ufl.ln(J) + (lmbda/2)*(ufl.ln(J))**2

# Variational form
Pi = psi * ufl.dx
F_res = ufl.derivative(Pi, u, v)
J = ufl.derivative(F_res, u, du)
# Mark right boundary (x=1)
class Right():
    def mark(self, x):
        return np.isclose(x[0], 1.0)

right = Right()
boundary_facets_right = mesh.locate_entities_boundary(domain, fdim, right.mark)
marked_facets = boundary_facets_right
marked_values = np.full_like(marked_facets, 1)
mt = mesh.meshtags(domain, fdim, marked_facets, marked_values)
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

# Pressure loading function
def get_pressure(t):
    return 10e3 * np.sin(2*np.pi*t)

# PyVista setup
pv.OFF_SCREEN = True  # For headless rendering if needed
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("def_cube.gif", fps=5)

# Time stepping
displacement_norms = []
times = []

for n in range(num_steps):
    t = (n+1)*dt
    p_ext = get_pressure(t)
    traction = fem.Constant(domain, PETSc.ScalarType((p_ext, 0.0, 0.0)))
    L = ufl.dot(traction, v) * ds(1)
    
    # Create new problem with updated residual (F_res - L)
    problem = NonlinearProblem(F_res - L, u, [bc], J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    
    num_its, converged = solver.solve(u)
    
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
    plt.savefig("displacement_level2solid.png")
    plt.show()