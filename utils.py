import numpy as np
import ufl
import basix.ufl
from dolfinx import fem
from petsc4py import PETSc



def compute_fiber_angle(wall_depth):
    # wall_depth ∈ [0, 1], 0 at epicardium, 1 at endocardium
    return np.radians(-60 + 120 * wall_depth)

# Assign fiber direction as a function, or use dolfinx.Function with interpolation.


def setup_function_space(domain):
    # Create vector function space on domain
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=PETSc.ScalarType)
    return fem.functionspace(domain, element)

