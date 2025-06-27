import numpy as np
def compute_fiber_angle(wall_depth):
    # wall_depth ∈ [0, 1], 0 at epicardium, 1 at endocardium
    return np.radians(-60 + 120 * wall_depth)

# Assign fiber direction as a function, or use dolfinx.Function with interpolation.
