
""" 
Simplified solid mechanics + 0D flow coupling example
Mesh: unit cube
Solver: FEniCS (solid) + placeholder 0D model
"""
import h5py as h5
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


# read H5 file
#pc = h5.File('UKBRVLV.h5', 'r')
# note that H5PY matrices in python are transposed
# generate the first principal mode
# with 1.5 times the standard deviation
#S = np.transpose(pc['MU']) + (1.5 * np.sqrt(pc['LATENT'][0,0]) * pc['COEFF'][0,:])
# get ED & ES points, & convert to 3 columns matrix [x, y, z]
#N = S.shape[1] // 2
#ed = np.reshape(S[0,:N], (-1,3))
#es = np.reshape(S[0,N:], (-1,3))



# Time parameters
T = 1.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size

# Material parameters
E = 1e5            # Young's modulus
nu = 0.3           # Poisson's ratio
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))

# Create unit cube mesh
mesh = UnitCubeMesh(8, 8, 8)

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Boundary condition: fix left face
def left_boundary(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), left_boundary)

# Define functions
u = Function(V)  # displacement
v = TestFunction(V)
du = TrialFunction(V)

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Strain energy density (Neo-Hookean)
psi = (mu/2)*(tr(C) - 3) - mu*ln(det(F)) + (lmbda/2)*(ln(det(F)))**2

# Total potential energy
Pi = psi*dx

# Compute first variation of Pi (residual)
F_res = derivative(Pi, u, v)

# Compute Jacobian
J = derivative(F_res, u, du)

# Placeholder for 0D model interaction (simple pressure pulse)
def get_pressure(t):
    return 10e3 * np.sin(2*np.pi*t)  # simple sinusoidal pressure [Pa]

# External load on right boundary
ds = Measure('ds', domain=mesh)

pv.OFF_SCREEN = True  # For headless rendering if needed
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("deformation_from_0dsolid.gif", fps=5)
# Define right boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

right = Right()
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
right.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Time-stepping
displacement_norms = []
times = []

for n in range(num_steps):
    t = (n+1)*dt
    p_ext = get_pressure(t)
    traction = Constant((p_ext, 0.0, 0.0))
    L = dot(traction, v)*ds(1)

    solve(F_res - L == 0, u, bc, J=J)

    disp_norm = norm(u, 'L2')
    displacement_norms.append(disp_norm)
    times.append(t)
    print(f"Step {n+1}, Time {t:.2f}, Displacement Norm = {disp_norm:.4e}")



# Plot displacement norm over time

plt.plot(times, displacement_norms)
plt.xlabel("Time [s]")
plt.ylabel("Displacement Norm [L2]")
plt.title("Displacement Response Over Time")
plt.grid(True)
plt.show()
