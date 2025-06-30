import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem, io,nls
from petsc4py import PETSc



def compute_fiber_angle(wall_depth):
    # wall_depth ∈ [0, 1], 0 at epicardium, 1 at endocardium
    return np.radians(-60 + 120 * wall_depth)

# Assign fiber direction as a function, or use dolfinx.Function with interpolation.


def setup_function_space(domain):
    # Create vector function space on domain
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=PETSc.ScalarType)
    return fem.functionspace(domain, element)

def define_neo_hookean(mu, lmbda, domain,V):
    u = fem.Function(V, name="Displacement")
    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    C = F.T * F
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    J = ufl.det(F)
    psi = (mu / 2) * (ufl.tr(C) - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    Pi = psi * ufl.dx
    F_res = ufl.derivative(Pi, u, v)
    J = ufl.derivative(F_res, u, du)
    return u, v, Pi, F_res, J


def atrial_pressure(t, T):
    tau = (t % T) / T
    return np.where(tau > 0.5, -100 * np.sin(np.pi * t / T) ** 2, 30 * np.sin(np.pi * t / T) ** 2)

def ventricular_pressure(t, T):
    tau = (t % T) / T
    return 100 * np.sin(np.pi * tau) ** 2 if tau < 0.5 else 0.0