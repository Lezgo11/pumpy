import argparse
from pumpy.ode import simulate

# Demo of presets to tweak physiology freely
PRESETS = {
    "normal": {
        "hr_bpm": 65,
        "starling_gain": 0.3,
        "resp_mmHg": 2.0,
        "valve_tau_open": 0.02,
        "valve_tau_close": 0.04,
        "valve_slope_mmHg": 1.0,
        "lv_fill_mmHg": 12.0,
    },
    "hypertension": {  
        "hr_bpm": 65,
        "starling_gain": 0.25,
        "resp_mmHg": 1.5,
        "valve_tau_open": 0.02,
        "valve_tau_close": 0.05,
        "valve_slope_mmHg": 1.0,
        "lv_fill_mmHg": 12.0,
    },
    "aging_stiff_arteries": { 
        "hr_bpm": 65,
        "starling_gain": 0.25,
        "resp_mmHg": 1.5,
        "valve_tau_open": 0.02,
        "valve_tau_close": 0.05,
        "valve_slope_mmHg": 1.0,
        "lv_fill_mmHg": 12.0,
    },
    "hfpEF": { 
        "hr_bpm": 70,
        "starling_gain": 0.2,
        "resp_mmHg": 1.0,
        "valve_tau_open": 0.02,
        "valve_tau_close": 0.05,
        "valve_slope_mmHg": 1.0,
        "lv_fill_mmHg": 14.0,
    },
    "hfrEF": { 
        "hr_bpm": 70,
        "starling_gain": 0.1,
        "resp_mmHg": 1.0,
        "valve_tau_open": 0.025,
        "valve_tau_close": 0.05,
        "valve_slope_mmHg": 1.0,
        "lv_fill_mmHg": 14.0,
    },
}

def main():
    ap = argparse.ArgumentParser(description="Run full-heart with physiologic presets.")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="normal")
    ap.add_argument("--beats", type=int, default=5)
    ap.add_argument("--outdir", default="outputs/full_heart_example")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]

    print(f"Running FULL heart - custom ")
    simulate(
        mode="full",
        beats=args.beats,
        hr_bpm=cfg["hr_bpm"],
        log_dir=args.outdir,
        make_plots=True,
        starling_gain=cfg["starling_gain"],
        resp_mmHg=cfg["resp_mmHg"],
        valve_tau_open=cfg["valve_tau_open"],
        valve_tau_close=cfg["valve_tau_close"],
        valve_slope_mmHg=cfg["valve_slope_mmHg"],
        lv_fill_mmHg=cfg["lv_fill_mmHg"],
    )

if __name__ == "__main__":
    main()
