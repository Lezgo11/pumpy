# PumPy

*A heart simulator to never skip a beat!*
by **lez**

---

## What is PumPy?

**PumPy** is a lightweight, Python-based heart simulator that models the dynamics of the left atrium and left ventricle (and can be extended to the full heart). It’s designed for developers, students, and researchers who want to play with cardiovascular physiology without the overhead of full finite element solvers like FEniCSx.

At its core, PumPy couples **time-varying elastance models**, **valve dynamics**, and a **Windkessel afterload** into a neat ODE system. The result: pressure–volume loops and waveforms that look and feel physiologically real.

---

## Why PumPy?

Because coding a heart from scratch is hard. Tuning it until it “looks right” is harder.
PumPy gives you:

* **Four modes of play**

  * `la` → left atrium only
  * `lv` → left ventricle only
  * `la_lv` → a coupled left heart (atrium + ventricle)
  * `full` → extendable to full heart (right side + pulmonary circulation)

* **Mesh-based initialization** (optional)
  Start your chambers with real geometrical volumes from `.msh` meshes (via meshio). Or, skip meshes for quick tests.

* **Physiological features out of the box**

  * Time-varying elastance (ventricular and atrial)
  * Passive filling curves (EDPVR)
  * Starling effect (preload-dependent contractility)
  * Soft valves with smooth opening fractions
  * Systemic afterload (2-element Windkessel with venous baseline)
  * Pulmonary venous inflow with optional respiration modulation

* **Beautiful outputs**
  CSV logs for all variables, plus auto-generated plots:

  * Pressure–volume loops (LA, LV)
  * Time-series traces (pressures, flows, valve states)

* **Test coverage**
  Comes with a pytest test that runs a no-mesh simulation and asserts physiological ranges. Because reproducibility matters.

---

## Installation

Clone the repo, then install with pip:

```bash
git clone https://github.com/lez/pumpy.git
cd pumpy
pip install -r requirements.txt
```

---

## Quickstart

Run the simulator straight from the command line:

```bash
pumpy-run --mode la_lv --beats 5
```

This produces:

* `outputs/left_heart_simple_log.csv`
* `outputs/pv_lv.csv`
* `outputs/pv_la.csv`
* `outputs/*.png` plots

You can also disable meshes explicitly:

```bash
pumpy-run --mode la_lv --la-mesh "" --lv-mesh "" --beats 6
```

---

## Examples

Example scripts live in `examples/`:

* `run_la.py` → LA only
* `run_lv.py` → LV only
* `run_la_lv.py` → coupled left heart
* `run_full.py` → full heart extension
* `run_nomesh.py` → quick test without meshes (great for CI/tests)

---

## Tuning the physiology

The fun part of pumpy is tweaking parameters and seeing how loops shift.

* **Elastance**:

  ```bash
  pumpy-run --mode la_lv --E-la-min 0.2 --E-la-max 0.6
  ```

  Adjusts LA stiffness, directly changing LA loop width and pressure.

* **Afterload**:
  Tune systemic R and C in `ode.py` to shift MAP and pulse pressure.

* **Filling resistance (R\_pv)**:
  Controls preload (EDV). Lower R\_pv → larger loops.

* **Starling gain**:
  Positive feedback between EDV and contractility. Keep ≤0.3 for stability.

---

## Testing

Run the included pytest:

```bash
pytest -q
```

It will:

* Run a no-mesh LA+LV simulation.
* Check that CSVs are generated.
* Verify LV volumes/pressures and LA volumes/pressures fall in physiological ranges.

This ensures every commit still produces “believable” hearts.

---

## Philosophy

pumpy is not a clinical model. It’s a **sandbox** to learn, experiment, and have fun with physiology. The code is kept **simple, hackable, and modular**, so you can extend it:

* Add new afterload models (3-element Windkessel, inertance).
* Add failure presets (HFpEF, HFrEF, hypertension).
* Connect to your own data assimilation pipelines.
* Or just generate pretty PV loops for your slides.

---

## Contributing

PRs welcome! If you’ve got ideas for new presets, better valve dynamics, or fun visualizations, send them in.
Bug reports and feature requests → issues page.

---

## Credits

* **dev:** lez
* **catch phrase:** *a heart simulator to never skip a beat!*
* **inspiration:** countless physiology diagrams and too many nights debugging pressure units.

---

## License

MIT. Free to use, hack, and share.
