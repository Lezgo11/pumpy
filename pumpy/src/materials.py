# materials.py
import ufl
from dolfinx import fem
from petsc4py import PETSc

def define_neo_hookean(mu, lmbda, domain, V):
    """
    Simple compressible Neo-Hookean material.
    Returns (u, v, du, F_res, J) for dolfinx NonlinearProblem.
    """
    u = fem.Function(V, name="Displacement")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    C = F.T*F
    Jdet = ufl.det(F)

    mu_, lam_ = mu, lmbda
    psi_iso = (mu_/2.0)*(ufl.tr(C) - 3.0)
    psi_vol = (lam_/2.0)*(ufl.ln(Jdet))**2  # compressible penalty
    psi = psi_iso + psi_vol

    F_res = ufl.derivative(ufl.inner(psi, 1.0)*ufl.dx(domain), u, v)
    J = ufl.derivative(F_res, u, du)
    return u, v, du, F_res, J

def define_holzapfel_ogden(domain, V, mu_iso, kappa_vol, k1, k2, f_expr):
    """
    Minimal anisotropic Holzapfel–Ogden-like strain energy with a single fiber family.
    f_expr: ufl vector field (unit fibers).
    Returns (u, v, du, F_res, J).
    """
    u = fem.Function(V, name="Displacement")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    C = F.T*F
    Jdet = ufl.det(F)

    # invariants
    I1 = ufl.tr(C)
    f0 = ufl.as_vector(f_expr)
    I4f = ufl.dot(f0, C*f0)

    # isotropic + anisotropic + volumetric
    psi_iso = (mu_iso/2.0)*(I1 - 3.0)
    # Only tension in fiber (I4f > 1)
    E_f = ufl.max_value(I4f - 1.0, 0.0)
    psi_f = (k1/(2.0*k2))*(ufl.exp(k2*(E_f**2)) - 1.0)
    psi_vol = (kappa_vol/2.0)*(ufl.ln(Jdet))**2

    psi = psi_iso + psi_f + psi_vol
    F_res = ufl.derivative(ufl.inner(psi, 1.0)*ufl.dx(domain), u, v)
    J = ufl.derivative(F_res, u, du)
    return u, v, du, F_res, J
