# PumPy

*A heart simulator to never skip a beat!*

## What is PumPy?

**PumPy** is a lightweight, Python-based heart simulator that models the dynamics of the Atrium Ventricle and models a full heart simulation! It’s designed mostly with students and researchers in mind, who want to pla around with cardiovascular physiology without the hassle and complexity of full finite element solvers.

Coding a heart from scratch is hard. Tuning it until it it's right is harder; PumPy unifies a time-varying elastance models, a valve dynamics module, and a Windkessel afterload model into an ODE system to obtain PV physiologically meaningful PV loops and time series data!

## Installation

Clone the repo, then install with pip:

```bash
git clone https://github.com/Lezgo11/pumpy.git
cd pumpy
pip install -e .
pip install -r requirements.txt
```

## Quickstart

*Note*: you will need mesh files for running the simulation. The defaults are found in the `mesh` folder as `idealized_LA.msh` and `idealized_LV.msh`. See [Parameter-tuning](#parameter-tuning)

Run the simulator straight from the command line:

```bash
pumpy-run --mode la_lv --beats 5 --log_dir="outputs/left_heart_example"
```

This produces:

* `outputs/left_heart_example/left_heart_simple_log.csv`
* `outputs/left_heart_example/pv_lv.csv`
* `outputs/left_heart_example/pv_la.csv`
* `outputs/left_heart_example/*.png` plots

### Examples

Example scripts live in `examples/`:

* `run_la_example.py` → LA only
* `run_lv_example.py` → LV only
* `run_left_heart_example.py` → coupled left heart
* `run_full_heart_example.py` → full heart extension (L+R)

### Outputs

CSV logs for all variables, plus auto-generated plots (if activated) such as

* Pressure–volume loops (LA, LV)
* Time-series traces (pressures, flows, valve states)

### Example calls

<details>
  <summary>Click here for more example calls</summary>

##### 1. Run a steady, coupled left heart with plots:

```bash
pumpy-run --mode la_lv --beats 6 --hr 60 \
  --la-mesh "" --lv-mesh "" \
  --E-la-min 0.20 --resp-mmHg 2.0
```

##### 2. Higher afterload (hypertensive feel) and stiffer arteries:

```bash
pumpy-run --mode la_lv --beats 6 \
  --la-mesh "" --lv-mesh "" \
  --E-la-min 0.18 --resp-mmHg 1.0
# Tip: in code or CLI, set R_sys≈2.2e8 and C_sys≈0.9e-8
```

##### 3. Use meshes in millimetres:

```bash
pumpy-run --mode la \
  --la-mesh mesh/your_mesh.msh
  --la-mesh-units mm
```

</details>

## Parameter tuning

The key part of PumpPy is being able to tweak parameters and seeing their impact on the PV loop formation! The function `simulate()` allows for many parameters to be changed simoultaneosly. Here's an overview of the most important ones:

```python
def simulate(
    beats: int,
    hr_bpm: float,
    dt: float,
    log_dir: str,
    mode: str,
    la_mesh: str,
    lv_mesh: str,
    la_mesh_units: str,
    lv_mesh_units: str,
    make_plots: bool,
    valve_tau_open: float,
    valve_tau_close: float,
    valve_slope_mmHg: float,
    lv_fill_mmHg: float,
    resp_mmHg: float,
    E_la_min: float,
    R_sys: float,
    C_sys: float,
    starling_gain: float,
)
```

### Simulation control

<details>
<summary>Simmulation parameters</summary>

* **`beats`** (int, default `3`)
How many cardiac cycles to simulate.

* **`hr_bpm`** (float, default `65.0`)
Heart rate in beats per minute. Period `T = 60 / hr_bpm`.

* **`dt`** (float, default `1e-3`)
Time step in seconds. Smaller → more accurate and smoother valve dynamics, but slower. Values between `5e-4` and `2e-3` work well.

* **`log_dir`** (str, default `"outputs"`)
Directory where CSVs and PNGs are written. Created automatically if missing.

* **`mode`** (str, default `"la_lv"`)
Choose the subsystem you want:

  * `la` — Left atrium only (standalone).
  * `lv` — Left ventricle only (+ aortic afterload); LA is replaced by a fixed filling pressure `lv_fill_mmHg`.
  * `la_lv` — Coupled left heart: LA ↔ LV with logical mitral/aortic valves and aortic afterload.
  * `full` — Full heart (L+R): RA ↔ RV ↔ pulmonary ↔ LA ↔ LV ↔ systemic. Lightweight right side.

</details>



### Usage modes

* `la` → left atrium only
* `lv` → left ventricle only
* `la_lv` → a coupled left heart (atrium + ventricle)
* `full` → extendable to full heart (right side + pulmonary circulation)

### Meshing
<details >
  <summary>Meshing parameters </summary>

* **`la_mesh`**, **`lv_mesh`** (str, defaults point to `mesh/idealized_*.msh`)

  * **Use your own simulation mesh**:  

    ```bash
    pumpy-run --mode la_lv --la-mesh "your_A_mesh.msh" --lv-mesh ""your_V_mesh.msh""
    ```

  * **`la_mesh_units`**, **`lv_mesh_units`** (str, default `"auto"`)
      Units for mesh coordinates: `"mm"`, `"cm"`, `"m"`, or `"auto"`.
      Most cardiac meshes are in **mm**.

      ```bash
      --la-mesh-units mm --lv-mesh-units mm
      ```

</details>



### Output

<details open>
  <summary>plots parameters </summary>
  
* **`make_plots`** (bool, default `True`)
If true, saves PNGs for the last beat PV loops and the main time‑series panel.

</details> 


### Valve opening

<details>
  <summary> Valve Opening Parameters</summary>

* **`valve_tau_open`** (s, default `0.02`)
Time constant for *toward‑open* transitions of valve opening fraction `s∈[0,1]`.

* **`valve_tau_close`** (s, default `0.05`)
Time constant for *toward‑closed* transitions.

* **`valve_slope_mmHg`** (mmHg, default `1.0`)
Steepness of the sigmoid that maps pressure difference (ΔP) to the target opening.

</details>

### Filling and respiration

<details>
  <summary>Filling and respiration Parameters</summary>

* **`lv_fill_mmHg`** (mmHg, default `12.0`)
Only used in `mode="lv"`. Acts as the left atrial surrogate pressure supplying the LV.

* **`resp_mmHg`** (mmHg, default `2.0`)
Amplitude of a slow sinusoidal modulation (≈0.25 Hz) applied to pulmonary venous pressure.

</details> 

### Atrial stiffness

<details>

  <summary>Atrial Stiffness Parameters</summary>

* **`E_la_min`** (mmHg/mL, default `0.08`)

  ```bash
  pumpy-run --mode la_lv --E-la-min 0.2 --E-la-max 0.6
  ```

  Minimum LA elastance (baseline compliance). Adjusts LA stiffness, directly changing LA loop width and pressure.

  Typical band: `0.06–0.30` mmHg/mL. Pair with `E_la_max` if you expose it; here we scale the LA activation around this minimum.

</details>

### Systemic afterload (Windkessel)

<details>
  <summary>Windkessel parameters</summary>

* **`R_sys`** (Pa·s/m³, default `1.5e8`)
Systemic resistance. Sets mean arterial pressure for a given cardiac output.
Rule of thumb: $R \approx (\text{MAP}-\text{Pv}) / \text{CO}$. Raising `R_sys` increases MAP (for the same heart).

* **`C_sys`** (m³/Pa, default `1.3e-8`)
Systemic compliance. Sets pulse pressure width.
Rule of thumb: $C \approx \text{SV} / \text{PP}$. Lower `C_sys` → stiffer arteries → wider pulse pressure.

Defaults are chosen so a healthy LV lands near \~120/80 mmHg with SV ≈ 60–80 mL after a few beats.

</details>

### Frank–Starling feedback

<details>
  <summary>Starling parameters</summary>

* **`starling_gain`** (dimensionless, default `0.3`)
Scales LV contractility based on how full the ventricle is relative to a reference EDV.

* `0.0` → off (contractility fixed).
* `0.1–0.4` → gentle physiological preload sensitivity.
* > `0.5` can destabilize if afterload/filling are extreme.

</details>


## Testing

integrated `pytest` test module that runs a no-mesh simulation and asserts physiological ranges.

Run by:

```bash
pytest -q
```

It will:

* Run a no-mesh LA+LV simulation.
* Check that CSVs are generated.
* Verify LV volumes/pressures and LA volumes/pressures fall in physiological ranges.

---

## Philosophy

pumpy is not a clinical model, and it's intended use does not include clinical reseach. It’s meant to be a starting point for curiois individuals to learn, experiment, and have ultimately have fun with physiology. The code is kept for this reason as simple, hackable, and modular as possible! Everyone is welcome (and highly encourged to) extend it and tweak it. Here are some ideas:

* Add new, more complex afterload models (3-element Windkessel, inertance).
* Add failure presets (HFpEF, HFrEF, hypertension).
* Use your own mesh and
* Implement your data by setting up a data assimilation pipeline.
* ... or just generate pretty PV loops for your slides :D

## Citation

If you use this package in your research, please cite:

```bash
@software{PumPy,
  title = {Pumpy - A Cardiac Simulation Package for learning physiology},
  author = {Lesly Perlaza Buitrago},
  year = {2025},
  url = {https://github.com/Lezgo11/pumpy}
}
```

## Contributing

PRs welcome! If you’ve got ideas for new presets, better valve dynamics, or fun visualizations, send them in.
Bug reports and feature requests by [email](leslyperlaza@gmail.com).

## Credits

* **dev:** Lez
* **Motivation**: This code is part of the Sustainable Computational Engineering course at RWTH Aachen. Many thanks to the course Lecturer and supervisor, Anil and Ana, for the inspiration and tips!
* **Inspiration:** countless sleepless nights studying physiology diagrams suring my bachelors.

---

## Licensing

MIT license: free to use, hack, and share. See more under [LICENSE](LICENSE)
