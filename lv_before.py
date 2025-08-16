# left_heart_simple_digital_valves.py
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.nls.petsc import NewtonSolver

# ------------------------
# Helpers
# ------------------------
MMHG_TO_PA = 133.322368

class AorticWK2:
    def __init__(self, C_ml_per_mmHg=1.5, R_mmHg_s_per_l=1.0):
        # Convert to SI: C [m^3/Pa], R [Pa·s/m^3]
        self.C = (C_ml_per_mmHg * 1e-6) / 133.322368
        self.R = (R_mmHg_s_per_l * 133.322368) / (1e-3)
        self.P = 80.0 * 133.322368  # start ~80 mmHg

    def step(self, Q_in, dt):
        # Here Q_in is *conceptual* since we’re not solving fluids yet.
        Q_out = self.P / self.R
        self.P += (Q_in - Q_out) / self.C * dt
        return self.P


def smooth_cos_window(phase):
    # 0..1 -> 0..1..0, C1 smooth
    return 0.5 * (1.0 - np.cos(2.0*np.pi*phase))

def atrial_pressure_pa(t, T, p_min_mmHg=2.0, p_max_mmHg=12.0):
    # simple LA pressure (physiologic range), shifted a bit late in cycle
    tau = ((t + 0.25*T) % T) / T
    p_mmHg = p_min_mmHg + (p_max_mmHg - p_min_mmHg) * smooth_cos_window(tau)
    return p_mmHg * MMHG_TO_PA

def ventricular_pressure_pa(t, T, p_sys_mmHg=120.0, active_window=0.45):
    # bump in first ~half of the cycle
    tau = (t % T) / T
    active = tau < active_window
    # shape the bump with sin^2 for smoothness
    amp = (p_sys_mmHg - 5.0) * (np.sin(np.pi * (tau/active_window))**2 if active else 0.0)
    return (5.0*MMHG_TO_PA) + amp*MMHG_TO_PA

def newton_solve(problem, u, comm, rtol=1e-6, max_it=40):
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = rtol
    solver.max_it = max_it
    solver.line_search = "bt"  # backtracking helps
    its, ok = solver.solve(u)
    return ok, its

def make_vector_space(domain):
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=PETSc.ScalarType)
    return fem.functionspace(domain, element)

def build_chamber(mesh_path, bc_tag, load_tags, traction_dir=(0.0, 0.0, 1.0),
                  E_base=1e5, nu=0.3):
    """
    Minimal chamber like your simple scripts:
    - vector traction (fixed direction) on load_tags
    - zero displacement BC on bc_tag
    - compressible Neo-Hookean
    """
    domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=3)
    V = make_vector_space(domain)
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Material
    mu = fem.Constant(domain, PETSc.ScalarType(E_base/(2*(1+nu))))
    lmbda = fem.Constant(domain, PETSc.ScalarType(E_base*nu/((1+nu)*(1-2*nu))))
    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    C = F.T*F
    J = ufl.det(F)
    psi = (mu/2)*(ufl.tr(C) - 3) - mu*ufl.ln(J) + (lmbda/2)*(ufl.ln(J))**2
    Pi = psi * ufl.dx
    F_res = ufl.derivative(Pi, u, v)
    Jform = ufl.derivative(F_res, u, du)

    # Measures / tags
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    fdim = domain.topology.dim - 1

    # Dirichlet BC on a ring/plane
    bc_facets = facet_tags.indices[facet_tags.values == bc_tag]
    bc_dofs = fem.locate_dofs_topological(V, fdim, bc_facets)
    zero = np.zeros(3, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, bc_dofs, V)

    # Traction direction
    t_dir = ufl.as_vector(traction_dir)

    def apply_and_solve(p_scalar):
        # vector traction like your scripts: t = p * dir (no surface normal here)
        t_vec = fem.Constant(domain, PETSc.ScalarType(tuple(p_scalar * np.array(traction_dir))))
        # sum over load tags
        L = sum(ufl.dot(t_vec, v) * ds(tag) for tag in load_tags)
        problem = fem.petsc.NonlinearProblem(F_res - L, u, [bc], J=Jform)
        ok, its = newton_solve(problem, u, MPI.COMM_WORLD, rtol=1e-6, max_it=60)
        return ok, its

    return {
        "domain": domain, "V": V, "u": u, "F_res": F_res, "J": Jform,
        "ds": ds, "facet_tags": facet_tags,
        "apply_and_solve": apply_and_solve
    }

# ------------------------
# Main orchestration
# ------------------------
if __name__ == "__main__":
    # Time
    T = 1.0
    steps = 10
    dt = T/steps
    # LV: fix basal plane (50), load on epicardium (10)   
    WK = AorticWK2()
 
    LV = build_chamber(
        "mesh/idealized_LV.msh",
        bc_tag=50,
        load_tags=[10],
        traction_dir=(0.0, 0.0, 1.0),
        E_base=1e5, nu=0.3
    )

    # LA: fix mitral ring (40), load on PV rings (20, 50)
    LA = build_chamber(
        "mesh/hollow_sphere_LA.msh",
        bc_tag=40,
        load_tags=[20, 50],
        traction_dir=(0.0, 0.0, 1.0),
        E_base=1e5, nu=0.3
    )

    # Aortic pressure baseline (very simple): ~80 mmHg in Pa
    P_AO = 80.0 * MMHG_TO_PA

    log = {
        "t": [], "P_LA": [], "P_LV_raw": [], "P_LV_eff": [],
        "mitral_open": [], "aortic_open": []
    }

    for k in range(steps):
        t = (k+1)*dt
        # simple waveforms
        P_LA = atrial_pressure_pa(t, T)                   # ~2..12 mmHg
        P_LV_raw = ventricular_pressure_pa(t, T)          # ~5..120 mmHg


        # digital valves
        mitral_open = P_LA > P_LV_raw
        aortic_open = P_LV_raw > P_AO
        Q_aortic = 1e-6 if aortic_open else 0.0  # m^3/s; tiny placeholder
        P_AO = WK.step(Q_aortic, dt)

        # effective LV pressure to apply (deformation-only logic)
        if aortic_open:
            P_LV_eff = P_LV_raw           # ejection
        elif mitral_open:
            P_LV_eff = P_LA               # filling - equalize
        else:
            P_LV_eff = P_LV_raw           # isovolumic-ish (still deforms in this toy model)

        # --- Apply loads and solve (vector tractions, like your simple codes) ---
        ok_la, its_la = LA["apply_and_solve"](P_LA)
        ok_lv, its_lv = LV["apply_and_solve"](P_LV_eff)

        if MPI.COMM_WORLD.rank == 0:
            print(f"[t={t:.3f}] LA ok={ok_la} its={its_la} | LV ok={ok_lv} its={its_lv} | "
                  f"mitral={mitral_open} aortic={aortic_open} | "
                  f"P_LA={P_LA/MMHG_TO_PA:.1f}mmHg P_LV_raw={P_LV_raw/MMHG_TO_PA:.1f} "
                  f"P_LV_eff={P_LV_eff/MMHG_TO_PA:.1f} P_AO={P_AO/MMHG_TO_PA:.1f}")

        # log
        log["t"].append(t)
        log["P_LA"].append(P_LA)
        log["P_LV_raw"].append(P_LV_raw)
        log["P_LV_eff"].append(P_LV_eff)
        log["mitral_open"].append(int(mitral_open))
        log["aortic_open"].append(int(aortic_open))

    if MPI.COMM_WORLD.rank == 0:
        # quick summary
        import csv
        with open("left_heart_simple_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t","P_LA(Pa)","P_LV_raw(Pa)","P_LV_eff(Pa)","mitral_open","aortic_open"])
            for i in range(len(log["t"])):
                w.writerow([log["t"][i], log["P_LA"][i], log["P_LV_raw"][i], log["P_LV_eff"][i],
                            log["mitral_open"][i], log["aortic_open"][i]])
        print("Wrote left_heart_simple_log.csv")

        import matplotlib.pyplot as plt, numpy as np
        t = np.array(log["t"])
        mmHg = 1.0/133.322368
        plt.figure(figsize=(8,4))
        plt.plot(t, np.array(log["P_LA"])*mmHg, label="P_LA")
        plt.plot(t, np.array(log["P_LV_raw"])*mmHg, label="P_LV_raw")
        plt.plot(t, np.array(log["P_LV_eff"])*mmHg, label="P_LV_eff", ls="--")
        plt.step(t, np.array(log["mitral_open"])*20, where="post", label="mitral (×20)")
        plt.step(t, np.array(log["aortic_open"])*20, where="post", label="aortic (×20)")
        plt.legend(); plt.xlabel("t [s]"); plt.ylabel("mmHg")
        plt.tight_layout(); plt.savefig("left_heart_states.png", dpi=150)

