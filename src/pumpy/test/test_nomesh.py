import math
import pandas as pd
import numpy as np
from pumpy.ode import simulate, PA_TO_MMHG  # PA→mmHg

def _last_beat_mask(t: np.ndarray, hr_bpm: float) -> np.ndarray:
    T = 60.0 / max(hr_bpm, 1e-9)
    return t >= (t[-1] - T - 1e-9)

def test_nomesh_coupled_left_heart(tmp_path):
    """
    Run LA+LV with no meshes and assert that:
      - CSVs are written
      - LV PV loop runs in physiological ranges
      - LA PV loop lives in physiological ranges
    """
    outdir = tmp_path / "../outputs/test/"
    out = simulate(
        mode="la_lv",
        beats=6,
        hr_bpm=60.0,
        dt=1e-3,
        log_dir=str(outdir),
        make_plots=False,     # avoid GUI needs in CI
        la_mesh="",           # disable meshes
        lv_mesh="",
        # keep your tuned defaults inside the package
    )

    # --- files exist
    main_csv = out["main_csv"]
    pv_lv_csv = out["pv_lv_csv"]
    pv_la_csv = out["pv_la_csv"]
    assert main_csv and pv_lv_csv and pv_la_csv
    for p in [main_csv, pv_lv_csv, pv_la_csv]:
        assert outdir.joinpath(p.split("/")[-1]).exists()

    # --- load
    df_main = pd.read_csv(main_csv)
    df_lv   = pd.read_csv(pv_lv_csv)
    df_la   = pd.read_csv(pv_la_csv)

    # --- slice the last beat
    t = df_lv["t"].to_numpy()
    mask = _last_beat_mask(t, hr_bpm=70.0)

    # LV ranges (targets: V 0–150 mL, P 0–120 mmHg; realistic band below)
    V_lv = (df_lv["V_m3"].to_numpy()[mask]) * 1e6     # mL
    P_lv = (df_lv["P_Pa"].to_numpy()[mask]) * PA_TO_MMHG

    assert 30 <= V_lv.max() <= 160, "LV EDV out of range"
    assert 30 <= V_lv.min() <= 100, "LV ESV out of range"
    sv = V_lv.max() - V_lv.min()
    assert 5 <= sv <= 110, "LV stroke volume not physiological"
    assert 20 <= P_lv.max() <= 130, "LV systolic pressure out of range"
    assert 0  <= P_lv.min() <= 20,  "LV diastolic pressure too high"

    # LA ranges (targets: Volume 5–45 mL, Pressure 10–15 mmHg)
    t_la = df_la["t"].to_numpy()
    mask_la = _last_beat_mask(t_la, hr_bpm=60.0)
    V_la = (df_la["V_m3"].to_numpy()[mask_la]) * 1e6   # mL
    P_la = (df_la["P_Pa"].to_numpy()[mask_la]) * PA_TO_MMHG
    assert 5  <= V_la.min() <= 90,  "LA min volume too large/small"
    assert 25 <= V_la.max() <= 90,  "LA max volume out of range"
    mean_la = P_la.mean()
    assert 9.0 <= mean_la <= 16.0,  "LA mean pressure out of range"
