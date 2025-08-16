# mechanics.py
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from materials import define_neo_hookean, define_holzapfel_ogden

def infer_scale_to_meters(domain, threshold_mm=2.0):
    """
    If the largest bbox extent > threshold_mm, we assume geometry is in mm
    and return 1e-3. Otherwise return 1.0 (already meters).
    """
    X = domain.geometry.x
    L = float(np.max(np.max(X, axis=0) - np.min(X, axis=0)))
    return 1e-3 if L > threshold_mm else 1.0
 
class MechanicsChamber:
    def __init__(self, mesh_path, material="neo_hookean",
                 bc_tags=(), endocardial_tags=(), pressure_tags=(),
                 E_base=4e5, nu=0.3,
                 mu_iso=5e4, kappa_vol=5e6, k1=2e5, k2=5.0,
                 fiber_epi_tag=None, fiber_endo_tag=None,
                 jit_options=None,
                 scale_geometry="auto",
                 traction_mode="pressure",
                 traction_dir=(0.0, 0.0, 1.0)):
        
        self.comm = MPI.COMM_WORLD
        self.domain, self.cell_tags, self.facet_tags = gmshio.read_from_msh(mesh_path, self.comm, gdim=3)
        
        # ---- Auto-scale to meters ----
        if scale_geometry == "auto":
            scale = infer_scale_to_meters(self.domain)
        else:
            scale = float(scale_geometry)
        if abs(scale - 1.0) > 1e-12:
            self.domain.geometry.x[:] *= scale
        
        # Small rank-0 report
        X = self.domain.geometry.x
        mins = X.min(axis=0); maxs = X.max(axis=0); ext = maxs - mins
        if self.comm.rank == 0:
            unit = "m" if scale == 1.0 else "mm→m (×1e-3)"
            print(f"[{mesh_path}] scaled: {unit}, bbox extent ≈ {ext}, max={ext.max():.4f} m")
        
        # ---- Function space and measures ----
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (3,)))
        self.fdim = self.domain.topology.dim - 1
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        self.n = ufl.FacetNormal(self.domain)
        self.jit_options = jit_options or {}
        self.traction_mode = traction_mode
        self.traction_dir = traction_dir
        
        # ---- Time stepping and damping ----
        self.u_prev = fem.Function(self.V)
        self.dt = 1.0
        self.eta = 1e3   # Small viscous damping for stability (Pa·s)
        
        # ---- Dirichlet BCs ----
        if not bc_tags:
            raise ValueError("bc_tags must be provided")
        
        bc_facets = np.concatenate([self.facet_tags.indices[self.facet_tags.values == t] for t in bc_tags])
        bc_dofs = fem.locate_dofs_topological(self.V, self.fdim, bc_facets)
        zero = np.array((0.0, 0.0, 0.0), dtype=PETSc.ScalarType)
        self.bc = fem.dirichletbc(zero, bc_dofs, self.V)
        
        # ---- Material model ----
        if material == "neo_hookean":
            mu = fem.Constant(self.domain, PETSc.ScalarType(E_base / (2.0 * (1.0 + nu))))
            lam = fem.Constant(self.domain, PETSc.ScalarType(E_base * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
            self.u, self.v, self.du, self.F_res, self.Jform = define_neo_hookean(mu, lam, self.domain, self.V)
        elif material == "holzapfel_ogden":
            from fibers import solve_transmural_phi, build_fiber_field
            if fiber_epi_tag is None or fiber_endo_tag is None:
                raise ValueError("Provide fiber epi/endo tags for HO.")
            phi = solve_transmural_phi(self.domain, self.facet_tags, fiber_epi_tag, fiber_endo_tag)
            f_expr = build_fiber_field(self.domain, phi, -60.0, 60.0, ref_axis=(0, 0, 1))
            self.u, self.v, self.du, self.F_res, self.Jform = define_holzapfel_ogden(
                self.domain, self.V, mu_iso, kappa_vol, k1, k2, f_expr
            )
        else:
            raise ValueError("Unknown material")
        
        # ---- Endocardium/pressure facets ----
        if not endocardial_tags or not pressure_tags:
            raise ValueError("endocardial_tags and pressure_tags must be provided")
        self.endo_tags = list(endocardial_tags)
        self.pressure_tags = list(pressure_tags)
        
        # ---- Reference cavity volume ----
        self.V0 = self._reference_cavity_volume()
        if self.comm.rank == 0:
            print(f"[{mesh_path}] reference cavity V0 ≈ {1e6*self.V0:.1f} mL")
    
    def begin_time_step(self, dt):
        """Update time step and store previous displacement."""
        self.dt = float(dt)
        self.u_prev.x.array[:] = self.u.x.array
    
    def _damped_forms(self):
        """Return residual and Jacobian with optional viscous damping."""
        if self.eta <= 0.0:
            return self.F_res, self.Jform
        
        # Add viscous damping term
        Kv = (self.eta/self.dt) * ufl.inner(ufl.grad(self.u - self.u_prev), ufl.grad(self.v)) * ufl.dx
        Fd = self.F_res + Kv
        Jd = ufl.derivative(Fd, self.u, self.du)
        return Fd, Jd
    
    def _reference_cavity_volume(self):
        """Compute reference volume of the domain."""
        topo = self.domain.topology
        topo.create_connectivity(topo.dim, 0)
        x = self.domain.geometry.x
        cells = topo.connectivity(topo.dim, 0).array.reshape(-1, 4)
        v = 0.0
        for c in cells:
            x0, x1, x2, x3 = x[c]
            v += abs(np.linalg.det(np.c_[x1-x0, x2-x0, x3-x0]))/6.0
        return v
    
    def cavity_volume(self):
        """Approximate current cavity volume via ΔV ≈ ∫_endo u·n dS + V0."""
        form = sum(ufl.inner(self.u, self.n) * self.ds(tag) for tag in self.endo_tags)
        dV = fem.assemble_scalar(fem.form(form, jit_options=self.jit_options))
        dV = self.domain.comm.allreduce(dV, op=MPI.SUM)
        return float(self.V0 + dV)
    
    def _L_total(self, p_now):
        """Compute total load vector for given pressure."""
        if self.traction_mode == "pressure":
            return sum((-p_now) * ufl.inner(self.v, self.n) * self.ds(tag)
                      for tag in self.pressure_tags)
        elif self.traction_mode == "vector":
            t = ufl.as_vector(self.traction_dir) * p_now
            return sum(ufl.dot(t, self.v) * self.ds(tag) for tag in self.pressure_tags)
        else:
            raise ValueError("traction_mode must be 'pressure' or 'vector'")
    
    def solve_with_pressure(self, p_scalar, load_splits: int = 16, max_it: int = 80, tikhonov: float = 0.0):
        """
        Apply traction -p n on pressure_tags with adaptive continuation.
        Improved convergence with better parameters and error handling.
        """
        p_total = float(p_scalar)
        
        # Use damped forms (more stable than Tikhonov for this problem)
        F_base, J_base = self._damped_forms()
        
        # Adaptive continuation
        alpha = 0.0
        dalpha = 1.0 / float(load_splits)
        
        while alpha < 1.0 - 1e-12:
            dalpha = min(dalpha, 1.0 - alpha)
            alpha_next = alpha + dalpha
            p_now = alpha_next * p_total
            
            # Create nonlinear problem
            L_total = self._L_total(p_now)
            problem = fem.petsc.NonlinearProblem(
                F_base - L_total, self.u, bcs=[self.bc], J=J_base,
                jit_options=self.jit_options,
                form_compiler_options={"optimize": True},
            )
            
            # Configure Newton solver with better parameters
            solver = NewtonSolver(self.comm, problem)
            solver.convergence_criterion = "incremental"
            solver.rtol = 1e-8  # Tighter relative tolerance
            solver.atol = 1e-10  # Tighter absolute tolerance
            solver.max_it = max_it
            
            # Configure line search
            solver.line_search = "bt"  # backtracking line search
            
            # Solve
            try:
                its, converged = solver.solve(self.u)
            except RuntimeError as e:
                if "Newton solver did not converge" in str(e):
                    converged = False
                    its = max_it
                else:
                    raise
            
            if converged:
                alpha = alpha_next
                # Increase step size if convergence was easy
                if its < 5:
                    dalpha *= 1.5
                elif its > 15:
                    dalpha *= 0.8
            else:
                # Reduce step size and retry
                dalpha *= 0.4
                if dalpha < 1e-5:
                    if self.comm.rank == 0:
                        print(f"Warning: Continuation stalled at alpha={alpha:.3f}, p_now={p_now:.3e} Pa")
                        print(f"Trying with smaller tolerances...")
                    
                    # Try with relaxed tolerances as last resort
                    solver.rtol = 1e-6
                    solver.atol = 1e-8
                    solver.max_it = max_it * 2
                    
                    try:
                        its, converged = solver.solve(self.u)
                        if converged:
                            alpha = alpha_next
                            dalpha = 1e-3  # Very small steps from here
                        else:
                            raise RuntimeError(f"Continuation failed completely at alpha={alpha:.3f}, p_now={p_now:.3e} Pa")
                    except RuntimeError:
                        raise RuntimeError(f"Continuation failed completely at alpha={alpha:.3f}, p_now={p_now:.3e} Pa")
        
        return True, its