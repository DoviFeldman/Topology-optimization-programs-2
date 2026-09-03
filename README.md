# Topology Optimization Programs

A small, self-contained studio for **topology optimization (TO)** — the technique
that grows lightweight, load-bearing, organic-looking structures (great for 3D
printing). Open it in the browser, pick a program, set a few sliders, hit **Run**,
and get a result you can download (a PNG for 2D, a printable **STL** for 3D).

Two solvers are built in and **always run** (pure NumPy/SciPy, no external TO
package, no accounts):

| Program | What it does | Output |
|---|---|---|
| **2D SIMP** | Classic density-based compliance minimization (DTU `top88` style). MBB beam / cantilever / bridge load cases. | Density image (PNG) |
| **3D SIMP → STL** | 3D compliance minimization (DTU `top3d` element formulation) on a clamped, loaded box. | 3D preview + **STL** |

`notes.txt` holds the two reference notes on the wider TO tool landscape
(PyTopo3D, OpenPISCO, ToOptiX/FreeCAD, Z88 Arion, ToPy, …) and 3D-printing tips.

---

## Run it (easiest → in GitHub Codespaces)

1. On GitHub: **Code ▸ Codespaces ▸ Create codespace on main**.
   The devcontainer installs the dependencies automatically.
2. In the Codespace terminal:
   ```bash
   bash run.sh
   ```
3. Codespaces pops up a **"Open in Browser"** for port **8501** — click it.
   That's the app. (If it doesn't pop up, open the **Ports** tab and click the
   8501 URL.)

## Run it locally (Mac / Linux)

```bash
pip install -r requirements.txt
streamlit run app.py       # or: bash run.sh
```
Then open http://localhost:8501. Works natively on Apple Silicon.

## Run a solver without the UI (script / CLI)

Everything is importable, so you can script it or run on your Mac's hardware
directly:

```bash
python -m engines.simp2d      # quick MBB-beam demo, prints convergence
python -m engines.simp3d      # quick 3D demo
```
```python
from engines import simp2d, simp3d, stl_io
rho2d = simp2d.optimize(nelx=120, nely=40, volfrac=0.5, load="cantilever")
rho3d = simp3d.optimize(nelx=32, nely=16, nelz=16, volfrac=0.3)
stl_io.voxels_to_stl(rho3d, "part.stl", threshold=0.5, spacing=1.0)  # printable
```

---

## The other tools from `notes.txt`

These aren't reimplemented here — they're separate projects. Where they run:

**Runs in Codespaces (Linux, headless) — install alongside this app:**
- **PyTopo3D** — pure-Python 3D SIMP with direct STL import/export. Best for a
  3D-printing workflow.
  ```bash
  pip install git+https://github.com/jihoonkim888/PyTopo3D.git
  ```
- **ToPy** — Python TO library driven by `.tpd` problem files: `pip install topy`
- **OpenPISCO** — feature-rich platform (overhang detection, STL export, thermal/
  buckling/modal): `conda install -c conda-forge openpisco`

**Desktop / GUI — best on your Mac (Apple Silicon), not a browser:**
- **Z88 Arion** — free GUI TO app (macOS/Unix builds available).
- **ToOptiX + FreeCAD addon** — runs inside FreeCAD's GUI.
- **CalculiX** — FEA solver, great for *validating* a printed design.

The **"Other tools & how to run"** page inside the app repeats these install
commands so you have them handy while testing.

---

## How the built-in solvers work (short version)

SIMP represents each element's material as a density `x ∈ [0,1]`, penalizes
intermediate values (stiffness `∝ x^p`), runs a finite-element solve, and uses
the optimality-criteria update + a density filter to march toward a stiff,
minimum-compliance layout at your target volume fraction. For 3D printing you
**threshold** the density (keep `x ≥ 0.5`) and export STL — exactly what the
3D program does. Always sanity-check a printed design with FEA (see `notes.txt`).
