# fibers.py
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


def solve_transmural_phi(domain, facet_tags, epi_tag, endo_tag):
    """
    Solve -Δφ=0 in Ω, φ=0 on epi_tag, φ=1 on endo_tag. Returns fem.Function(W).
    """
    W = fem.functionspace(domain, ("Lagrange", 1))
    phi = fem.Function(W, name="phi")
    v = ufl.TestFunction(W)
    u = ufl.TrialFunction(W)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Boundary dofs (Dirichlet)
    fdim = domain.topology.dim - 1
    epi_facets = facet_tags.indices[facet_tags.values == epi_tag]
    endo_facets = facet_tags.indices[facet_tags.values == endo_tag]
    epi_dofs = fem.locate_dofs_topological(W, fdim, epi_facets)
    endo_dofs = fem.locate_dofs_topological(W, fdim, endo_facets)

    bc_epi = fem.dirichletbc(PETSc.ScalarType(0.0), epi_dofs, W)
    bc_endo = fem.dirichletbc(PETSc.ScalarType(1.0), endo_dofs, W)

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx(domain)
    L = ufl.Constant(domain, PETSc.ScalarType(0.0))*v*ufl.dx(domain)

    A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc_epi, bc_endo])
    A.assemble()
    b = fem.petsc.assemble_vector(fem.form(L))
    fem.petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc_epi, bc_endo]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc_epi, bc_endo])

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.solve(b, phi.vector)

    return phi

def build_fiber_field(domain, phi_function, alpha_epi_deg=-60.0, alpha_endo_deg=60.0, ref_axis=(0.0, 0.0, 1.0)):
    """
    Returns a UFL vector giving the local fiber direction.
    We rotate a reference axis by a transmural angle α(φ) = α_epi + (α_endo-α_epi)*φ.
    """
    # Continuous angle field
    alpha = alpha_epi_deg + (alpha_endo_deg - alpha_epi_deg) * phi_function
    alpha_rad = alpha * (np.pi/180.0)

    # Rotate ref_axis in a fixed plane (e1-e3) by alpha_rad; you can extend as needed
    e1 = ufl.as_vector((1.0, 0.0, 0.0))
    e3 = ufl.as_vector((0.0, 0.0, 1.0))
    f = ufl.cos(alpha_rad) * e3 + ufl.sin(alpha_rad) * e1
    # Normalize for safety
    f = f / ufl.sqrt(ufl.dot(f, f) + 1e-12)
    return f
