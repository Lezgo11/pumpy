from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

domain, cell_tags, facet_tags = gmshio.read_from_msh("mesh/hollow_sphere_LA.msh", MPI.COMM_WORLD, gdim=3)
X = domain.geometry.x
mins = X.min(axis=0); maxs = X.max(axis=0)
print("BBox min:", mins, "max:", maxs, "extent:", maxs - mins)
