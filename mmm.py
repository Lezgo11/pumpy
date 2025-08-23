import sys
try:
    import numpy as np
    import ufl
    import basix.ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import fem
    from dolfinx.io import gmshio
    from dolfinx.nls.petsc import NewtonSolver
except ModuleNotFoundError as e:
    missing = e.name
    print(f"Required module '{missing}' is not installed. Please install it and retry.")
    sys.exit(1)
import time

MMHG_TO_PA = 133.322368
PA_TO_MMHG = 1.0 / MMHG_TO_PA

ENABLE_VOLUME = False
ENABLE_ELECTRO = False
ENABLE_PHYSIO_TUNER = False
ACTIVE_CHAMBERS = ("LV",)

PHYS_TARGETS = {
    "SBP_mmHg": 120.0,
    "DBP_mmHg": 80.0,
    "LV_EDV_mL": 130.0,
    "LA_EDV_mL": 60.0,
    "LV_ESV_mL": 50.0,
    "HR_bpm": 120.0,
}

GAINS = {
    "active_stress_max_kPa": 10.0,
    "pressure_gain": 0.15,
    "volume_gain": 0.10,
}


class EPDriver:
    def __init__(self, dt: float, hr_bpm: float, gains):
        self.dt = float(dt)
        self.period = 60.0 / max(hr_bpm, 1e-6)
        self.t = 0.0
        self.v = 0.0
        self.w = 1.0
        self.v_gate = 0.13
        self.tau_open = 120.0
        self.tau_close = 150.0
        self.T_out = 6.0
        self.gains = gains
        self.a = 0.0

    def _stim(self, t):
        return 1.0 if (t % self.period) < 0.002 else 0.0

    def step(self):
        I = self._stim(self.t)
        v, w = self.v, self.w
        dv = (I - v * (v - self.v_gate) * (1 - v) - v * w / self.T_out) * self.dt
        dw = ((1 - w) / self.tau_close if v < self.v_gate else -w / self.tau_open) * self.dt
        self.v = float(np.clip(v + dv, 0.0, 1.0))
        self.w = float(np.clip(w + dw, 0.0, 1.0))
        self.t += self.dt
        self.a = 0.0 if self.v < self.v_gate else (self.v - self.v_gate) / (1 - self.v_gate)
        self.a = float(np.clip(self.a, 0.0, 1.0))
        return self.a

    def active_stress_kPa(self):
        return GAINS["active_stress_max_kPa"] * self.a


class PhysioTuner:
    def __init__(self, targets, gains):
        self.t = targets
        self.g = gains

    def scale_P(self, P_Pa, systole: bool):
        target = self.t["SBP_mmHg"] if systole else self.t["DBP_mmHg"]
        mmHg = P_Pa * PA_TO_MMHG
        mmHg_adj = mmHg + self.g["pressure_gain"] * (target - mmHg)
        return mmHg_adj * MMHG_TO_PA

    def scale_V(self, V_m3, diastole: bool):
        target = self.t["LV_EDV_mL"] if diastole else self.t["LV_ESV_mL"]
        ml = V_m3 * 1e6
        ml_adj = ml + self.g["volume_gain"] * (target - ml)
        return ml_adj * 1e-6


class VentricularVolumeLogger:
    def __init__(self, V0_mL=120.0, R_mitral=2.0e8, R_aortic=2.0e8):
        self.V_m3 = V0_mL * 1e-6
        self.R_mitral = R_mitral
        self.R_aortic = R_aortic
        self.log = {"t": [], "P_Pa": [], "V_m3": []}

    def step(self, t, P_LA, P_LV, P_AO, mitral_open, aortic_open, dt):
        Q_in = ((P_LA - P_LV) / self.R_mitral) if mitral_open else 0.0
        Q_out = ((P_LV - P_AO) / self.R_aortic) if aortic_open else 0.0
        dV = (Q_in - Q_out) * dt
        self.V_m3 = max(self.V_m3 + dV, 0.0)
        self.log["t"].append(t)
        self.log["P_Pa"].append(P_LV)
        self.log["V_m3"].append(self.V_m3)

    def export_csv(self, path="pv_lv.csv"):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            print("pandas not installed; skipping ventricular CSV export")
            return None
        df = pd.DataFrame(self.log)
        df.to_csv(path, index=False)
        return path


class AtrialVolumeLogger:
    def __init__(self, V0_mL=60.0, R_pv=2.0e8, R_mitral=2.0e8):
        self.V_m3 = V0_mL * 1e-6
        self.R_pv = R_pv
        self.R_mitral = R_mitral
        self.log = {"t": [], "P_Pa": [], "V_m3": []}

    def step(self, t, P_PV, P_LA, P_LV, mitral_open, dt):
        Q_pv_in = max((P_PV - P_LA) / self.R_pv, 0.0)
        Q_mitral_out = ((P_LA - P_LV) / self.R_mitral) if mitral_open else 0.0
        dV = (Q_pv_in - Q_mitral_out) * dt
        self.V_m3 = max(self.V_m3 + dV, 0.0)
        self.log["t"].append(t)
        self.log["P_Pa"].append(P_LA)
        self.log["V_m3"].append(self.V_m3)

    def export_csv(self, path="pv_la.csv"):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            print("pandas not installed; skipping atrial CSV export")
            return None
        df = pd.DataFrame(self.log)
        df.to_csv(path, index=False)
        return path


def make_vector_space(domain, LA_adv=False):
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=PETSc.ScalarType)
    if LA_adv:
        element = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,), dtype=PETSc.ScalarType)
    return fem.functionspace(domain, element)


def read_mesh_safe(path, name):
    try:
        return gmshio.read_from_msh(path, MPI.COMM_WORLD, 0, gdim=3)
    except (OSError, RuntimeError) as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"Error reading {name} mesh '{path}': {e}")
        return None, None, None


def newton_solve(problem, u, comm, rtol=1e-6, max_it=10):
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.error_on_nonconvergence = False
    solver.rtol = rtol
    solver.max_it = max_it
    try:
        n, converged = solver.solve(u)
    except RuntimeError as e:
        if comm.rank == 0:
            print(f"Newton solver failed: {e}")
        return False, 0
    if not converged and comm.rank == 0:
        print(f"Warning: Newton solver did not converge after {n} iterations")
    return converged, n


def make_solid(domain, facet_tags, load_tags, bc_tags, E=10e3, nu=0.49, traction_dir=(0.0, 1.0, 0.0)):
    V = make_vector_space(domain, LA_adv)
    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    C = F.T * F
    Ic = ufl.tr(C)
    J = ufl.det(F)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    psi = (mu / 2.0) * (Ic - 3.0) - mu * ufl.ln(J) + (lmbda / 2.0) * (ufl.ln(J)) ** 2
    F_res = ufl.derivative(psi * ufl.dx, u, v)
    Jform = ufl.derivative(F_res, u, du)

    domain.topology.create_connectivity(domain.topology.dim - 2, domain.topology.dim)
    fdim = domain.topology.dim - 1
    facets = facet_tags.find(1) if 1 in np.unique(facet_tags.values) else np.array([], dtype=np.int32)
    bc = None
    if facets.size > 0:
        V0 = fem.FunctionSpace(domain, ("Lagrange", 1))
        zero = fem.Function(V)
        zero.x.array[:] = 0.0
        bc = fem.dirichletbc(zero, fem.locate_dofs_topological(V, domain.topology.dim - 1, facets))
    else:
        bc_facets = facet_tags.indices[facet_tags.values == bc_tags]
        bc_dofs = fem.locate_dofs_topological(V, fdim, bc_facets)
        zero = np.zeros(3, dtype=PETSc.ScalarType)
        bc = fem.dirichletbc(zero, bc_dofs, V)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    def apply_and_solve(p_scalar):
        t_vec = fem.Constant(domain, PETSc.ScalarType(tuple(p_scalar * np.array(traction_dir))))
        L = 0
        for tag in load_tags:
            n = ufl.FacetNormal(domain)
            L += -p_scalar * ufl.dot(v, n) * ds(tag)
        problem = fem.petsc.NonlinearProblem(F_res - L, u, [bc] if bc else [], J=Jform)
        ok, its = newton_solve(problem, u, MPI.COMM_WORLD, rtol=1e-6, max_it=10)
        return ok, its

    return {
        "domain": domain,
        "V": V,
        "u": u,
        "F_res": F_res,
        "J": Jform,
        "ds": ds,
        "facet_tags": facet_tags,
        "apply_and_solve": apply_and_solve,
    }


class AorticWK2:
    def __init__(self, R=1.1e7, C=1.5e-3, P0_mmHg=80.0):
        self.R = R
        self.C = C
        self.P = P0_mmHg * MMHG_TO_PA

    def step(self, dt, Q_in):
        Q_out = self.P / self.R
        self.P += (Q_in - Q_out) / self.C * dt
        return self.P


class PulmVenousSource:
    def __init__(self, P0_mmHg=12.0):
        self.P0 = P0_mmHg * MMHG_TO_PA

    def pressure(self, t):
        return self.P0


def smooth_cos_window(phase):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))


def atrial_pressure_pa(t, T, p_min_mmHg=2.0, p_max_mmHg=12.0):
    tau = ((t + 0.25 * T) % T) / T
    p_mmHg = p_min_mmHg + (p_max_mmHg - p_min_mmHg) * smooth_cos_window(tau)
    return p_mmHg * MMHG_TO_PA


if __name__ == "__main__":
    time_start = time.time()
    ATRIUM_MSH = "mesh/hollow_sphere_LA.msh"
    VENTRICLE_MSH = "mesh/idealized_LV.msh"
    ATRIUM_MSH_ADV = "mesh/LA_adv.msh"

    LA = None
    LV = None
    if "LA" in ACTIVE_CHAMBERS:
        msh_A, ct_A, ft_A = read_mesh_safe(ATRIUM_MSH, "LA")
        if msh_A is not None:
            LA_adv = False
            LA = make_solid(msh_A, ft_A, load_tags=(10,), bc_tags=20)

    if "LV" in ACTIVE_CHAMBERS:
        msh_V, ct_V, ft_V = read_mesh_safe(VENTRICLE_MSH, "LV")
        if msh_V is not None:
            LA_adv = False
            LV = make_solid(msh_V, ft_V, load_tags=(11,), bc_tags=None)

    if "LA_adv" in ACTIVE_CHAMBERS:
        msh_A, ct_A, ft_A = read_mesh_safe(ATRIUM_MSH_ADV, "LA_adv")
        if msh_A is not None:
            LA_adv = True
            LA = make_solid(msh_A, ft_A, load_tags=(30,), bc_tags=50)

    hr = PHYS_TARGETS["HR_bpm"]
    T_cyc = 60.0 / hr
    dt = 0.1
    T_end = 3 * T_cyc
    print(f"Simulation time: {T_end:.2f}s (dt={dt:.3f}s, {hr} bpm), with T_cyc={T_cyc:.3f}s")

    ep = EPDriver(dt, hr, GAINS) if ENABLE_ELECTRO else None
    tuner = PhysioTuner(PHYS_TARGETS, GAINS) if ENABLE_PHYSIO_TUNER else None
    vlogger = VentricularVolumeLogger(V0_mL=PHYS_TARGETS["LV_EDV_mL"]) if (ENABLE_VOLUME and ("LV" in ACTIVE_CHAMBERS)) else None
    alogger = AtrialVolumeLogger(V0_mL=PHYS_TARGETS["LA_EDV_mL"]) if (ENABLE_VOLUME and ("LA" in ACTIVE_CHAMBERS)) else None
    pv_src = PulmVenousSource(P0_mmHg=12.0) if ("LA" in ACTIVE_CHAMBERS) else None
    ao = AorticWK2() if ("LV" in ACTIVE_CHAMBERS) else None

    log = {
        "t": [],
        "P_PV(Pa)": [],
        "P_LA(Pa)": [],
        "P_LV_raw(Pa)": [],
        "P_LV_eff(Pa)": [],
        "mitral_open": [],
        "aortic_open": [],
    }

    def elastance(t):
        phi = (t % T_cyc) / T_cyc
        return 0.05 + 0.95 * smooth_cos_window((phi + 0.1) % 1.0)

    P_LV_passive = 8.0 * MMHG_TO_PA

    t = 0.0
    while t <= T_end + 1e-12:
        active_stress_kPa = 0.0
        if ep:
            a = ep.step()
            active_stress_kPa = ep.active_stress_kPa()

        if "LA_adv" in ACTIVE_CHAMBERS:
            P_LA = atrial_pressure_pa(t, T_cyc)
            P_PV = pv_src.pressure(t) if pv_src else 12.0 * MMHG_TO_PA
            if tuner:
                P_LA = tuner.scale_P(P_LA, systole=False)
        elif "LA" in ACTIVE_CHAMBERS:
            P_LA = atrial_pressure_pa(t, T_cyc)
            P_PV = pv_src.pressure(t) if pv_src else 12.0 * MMHG_TO_PA
            if tuner:
                P_LA = tuner.scale_P(P_LA, systole=False)
        else:
            P_PV = 12.0 * MMHG_TO_PA
            P_LA = 10.0 * MMHG_TO_PA

        if "LV" in ACTIVE_CHAMBERS:
            P_AO = ao.P if ao is not None else 120.0 * MMHG_TO_PA
            P_LV_raw = P_LV_passive + 60e3 * elastance(t)
            mitral_open = P_LA > P_LV_raw
            aortic_open = P_LV_raw > P_AO
            if aortic_open:
                P_LV_eff = P_LV_raw
            elif mitral_open:
                P_LV_eff = P_LA
            else:
                P_LV_eff = P_LV_raw
            if tuner:
                P_LV_eff = tuner.scale_P(P_LV_eff, systole=aortic_open)
        else:
            P_LV_raw = np.nan
            P_LV_eff = np.nan
            mitral_open = False
            aortic_open = False
            P_AO = np.nan

        if LA is not None:
            ok_la, its_la = LA["apply_and_solve"](P_LA)
            if alogger is not None:
                alogger.step(t, P_PV, P_LA, P_LV_raw if np.isfinite(P_LV_raw) else P_LA, mitral_open, dt)
        if LV is not None:
            ok_lv, its_lv = LV["apply_and_solve"](P_LV_eff)
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Simulation completed in {its_la if LA else its_lv}[t={t:.3f}] LA={'on' if LA else 'off'} LV={'on' if LV else 'off'} | "
                f"mitral={mitral_open} aortic={aortic_open} | "
                f"P_LA={P_LA*PA_TO_MMHG:.1f}mmHg P_LV_raw={P_LV_raw*PA_TO_MMHG if np.isfinite(P_LV_raw) else float('nan'):.1f} "
                f"P_LV_eff={P_LV_eff*PA_TO_MMHG if np.isfinite(P_LV_eff) else float('nan'):.1f} "
                f"P_AO={P_AO*PA_TO_MMHG if np.isfinite(P_AO) else float('nan'):.1f} mmHg"
            )

        if LV is not None:
            if vlogger is not None:
                vlogger.step(
                    t,
                    P_LA,
                    P_LV_eff if np.isfinite(P_LV_eff) else P_LA,
                    P_AO if np.isfinite(P_AO) else P_LA,
                    mitral_open,
                    aortic_open,
                    dt,
                )
            Q_lv_out = ((P_LV_eff - P_AO) / ao.R) if (ao and aortic_open) else 0.0
            if ao:
                ao.step(dt, Q_lv_out)

        log["t"].append(t)
        log["P_PV(Pa)"].append(P_PV)
        log["P_LA(Pa)"].append(P_LA)
        log["P_LV_raw(Pa)"].append(P_LV_raw)
        log["P_LV_eff(Pa)"].append(P_LV_eff)
        log["mitral_open"].append(int(bool(mitral_open)))
        log["aortic_open"].append(int(bool(aortic_open)))

        t += dt
    print("Simulation completed in {:.2f}s".format(time.time() - time_start))

    if MPI.COMM_WORLD.rank == 0:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            pd = None
            print("pandas not installed; skipping log CSV")
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            plt = None
            print("matplotlib not installed; skipping plots")

        if pd is not None:
            df = pd.DataFrame(log)
            df.to_csv("left_heart_simple_log.csv", index=False)
            print("Wrote left_heart_simple_log.csv")
        if vlogger:
            path = vlogger.export_csv("pv_lv.csv")
            if path:
                print("Wrote", path)
        if alogger:
            path = alogger.export_csv("pv_la.csv")
            if path:
                print("Wrote", path)
        if plt and vlogger:
            V_ml = np.array(vlogger.log["V_m3"]) * 1e6
            P_mmHg = np.array(vlogger.log["P_Pa"]) * PA_TO_MMHG
            plt.figure(figsize=(4, 4))
            plt.plot(V_ml, P_mmHg)
            plt.xlabel("Volume [mL]"); plt.ylabel("Pressure [mmHg]")
            plt.title("LV PV loop (simple)")
            plt.tight_layout(); plt.savefig("pv_lv.png", dpi=150)
        if plt and alogger:
            V_ml = np.array(alogger.log["V_m3"]) * 1e6
            P_mmHg = np.array(alogger.log["P_Pa"]) * PA_TO_MMHG
            plt.figure(figsize=(4, 4))
            plt.plot(V_ml, P_mmHg)
            plt.xlabel("Volume [mL]"); plt.ylabel("Pressure [mmHg]")
            plt.title("LA PV loop (simple)")
            plt.tight_layout(); plt.savefig("pv_la.png", dpi=150)
        if plt:
            t_arr = np.array(log["t"])
            plt.figure(figsize=(8, 4))
            plt.plot(t_arr, np.array(log["P_LA(Pa)"]) * PA_TO_MMHG, label="P_LA")
            if np.isfinite(np.nanmean(log["P_LV_raw(Pa)"])):
                plt.plot(t_arr, np.array(log["P_LV_raw(Pa)"]) * PA_TO_MMHG, label="P_LV_raw")
                plt.plot(
                    t_arr,
                    np.array(log["P_LV_eff(Pa)"]) * PA_TO_MMHG,
                    label="P_LV_eff",
                    ls="--",
                )
                plt.step(
                    t_arr,
                    np.array(log["mitral_open"]) * 20,
                    where="post",
                    label="mitral (×20)",
                )
                plt.step(
                    t_arr,
                    np.array(log["aortic_open"]) * 20,
                    where="post",
                    label="aortic (×20)",
                )
            plt.legend(); plt.xlabel("t [s]"); plt.ylabel("mmHg")
            plt.tight_layout(); plt.savefig("left_heart_states.png", dpi=150)