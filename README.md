# Topology Optimization Programs

A small studio for **topology optimization (TO)** — the technique that grows
lightweight, load-bearing, organic structures (great for 3D printing). Open it in
the browser, optimize inside a box **or inside your own uploaded STL**, and
download a printable STL.

## How many programs, and where each runs

**There are 6 topology-optimization programs.** **1** is built in and runs live
in this app (including on your own uploaded STL). The other **5** are separate
open-source tools — this repo documents exactly how to install/run each and where
it works.

| # | Program | Runs live in THIS app | GitHub Codespaces | Your Mac |
|---|---------|:---:|:---:|:---:|
| 1 | **3D SIMP** (built in) | ✅ yes | ✅ | ✅ |
| 2 | **PyTopo3D** | ⛔ install & run separately | ✅ | ✅ |
| 3 | **ToPy** | ⛔ install & run separately | ✅ | ✅ |
| 4 | **OpenPISCO** | ⛔ install & run separately | ✅ | ✅ |
| 5 | **Z88 Arion** (GUI) | ⛔ desktop app | ❌ | ✅ |
| 6 | **ToOptiX + FreeCAD** (GUI) | ⛔ desktop app | ❌ | ✅ |

- ✅ **Works now, no setup:** the built-in **3D SIMP** (this app).
- ✅ **Run in Codespaces (Linux):** 3D SIMP, PyTopo3D, ToPy, OpenPISCO — **4 of 6**.
- ✅ **Run on your Mac (Apple Silicon):** **all 6** — the two GUI apps only run on
  the desktop, not a browser.

*(CalculiX is also documented as an FEA tool for **validating** a printed design —
it isn't a TO program itself.)*

The built-in **3D SIMP** is pure NumPy/SciPy — no external TO package, no
accounts — so it always runs.

## Bring your own geometry (STL upload)

In the **3D SIMP → STL** page, set **Design domain = "Upload your own STL"**. Your
STL is voxelized into the *design domain*: material is only ever placed **inside
your shape**. Pick which face is clamped, which face is loaded and in what
direction, set a volume budget, and Run. You get a 3D preview and a downloadable,
printable STL of the optimized part. (PyTopo3D also imports STL via its own CLI;
Z88/ToOptiX take geometry through FreeCAD.)

---

## Run it (easiest → GitHub Codespaces)

1. On GitHub: **Code ▸ Codespaces ▸ Create codespace on main** (the devcontainer
   installs dependencies automatically).
2. In the terminal: `bash run.sh`
3. Click the **"Open in Browser"** popup for port **8501** (or the **Ports** tab).

## Run it locally (Mac / Linux)

```bash
pip install -r requirements.txt
streamlit run app.py       # or: bash run.sh
```
Open http://localhost:8501. Native on Apple Silicon.

## Run the solver without the UI (script it / use your Mac's hardware)

```python
from engines import simp3d, stl_io
# box domain:
rho = simp3d.optimize(nelx=32, nely=16, nelz=16, volfrac=0.3,
                      fixed_face="x-", load_face="x+", load_dir="-y")
# or inside your own STL:
domain, meta = stl_io.stl_to_domain("my_part.stl", resolution=48)
rho = simp3d.optimize(domain=domain, **{k: meta[k] for k in ("nelx","nely","nelz")},
                      volfrac=0.35)
stl_io.voxels_to_stl(rho, "optimized.stl", threshold=0.5, spacing=meta["pitch"])
```

---

## The other tools (from `notes.txt`)

**Runs in Codespaces (Linux) — install alongside this app:**
- **PyTopo3D** — pure-Python 3D SIMP, direct STL import/export.
  `pip install git+https://github.com/jihoonkim888/PyTopo3D.git`
- **ToPy** — `pip install topy` (`.tpd` problem files)
- **OpenPISCO** — `conda install -c conda-forge openpisco`

**Desktop / GUI — best on your Mac, not a browser:**
- **Z88 Arion** — free GUI TO app (macOS/Unix builds).
- **ToOptiX + FreeCAD addon** — runs inside FreeCAD's GUI.
- **CalculiX** — FEA solver for *validating* a printed design.

The **"Other tools & how to run"** page in the app repeats these commands.

---

## How the built-in solver works (short version)

SIMP gives each element a density `x ∈ [0,1]`, penalizes intermediate values
(stiffness `∝ x^p`), runs a finite-element solve, and uses the optimality-criteria
update + a density filter to march toward a stiff, minimum-compliance layout at
your target volume fraction. For printing you **threshold** the density (keep
`x ≥ 0.5`) and export STL. Always sanity-check a printed design with FEA
(`notes.txt`).
