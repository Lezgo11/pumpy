import argparse, json
from .ode import simulate

def main():
    p = argparse.ArgumentParser(description="Run a simple left-heart model with selectable modes.")
    p.add_argument("--mode", choices=["la", "lv", "la_lv", "full"], default="la_lv",
                    help=( "la: LA only | lv: LV only | la_lv: coupled left heart (LA+LV) | "
                        "full: full heart (RA+RV+Pulmonary + LA+LV+Systemic)"))
    p.add_argument("--beats", type=int, default=3)
    p.add_argument("--hr", type=float, default=60.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--la-mesh", default="mesh/idealized_LA.msh")
    p.add_argument("--lv-mesh", default="mesh/idealized_LV.msh")
    p.add_argument("--resp-mmHg", type=float, default=0.0)
    p.add_argument("--starling-gain", type=float, default=0.0)
    p.add_argument("--valve-tau-open", type=float, default=0.02)
    p.add_argument("--valve-tau-close", type=float, default=0.04)
    p.add_argument("--valve-slope-mmHg", type=float, default=1.0)
    p.add_argument("--E-la-min",type=float, default=0.08,help="Minimum LA elastance [mmHg/mL]"
)
    args = p.parse_args()
    

    mode = "la_lv" if args.mode == "full" else args.mode

    outputs = simulate(
        beats=args.beats,
        hr_bpm=args.hr,
        dt=args.dt,
        log_dir=args.outdir,
        mode=mode,
        la_mesh=args.la_mesh,
        lv_mesh=args.lv_mesh,
        resp_mmHg=args.resp_mmHg,
        starling_gain=args.starling_gain,
        valve_tau_open=args.valve_tau_open,
        valve_tau_close=args.valve_tau_close,
        valve_slope_mmHg=args.valve_slope_mmHg,
    )
    print(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    main()
