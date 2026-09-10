# Topology Optimization Studio — local Mac build

A studio for **topology optimization**: the technique that grows lightweight,
load-bearing, organic structures from a solid part. Upload an STL, say where it
is held and where it is pushed, and get a printable, optimized STL back.

Three real topology-optimization programs are wired in, **all running natively
on Apple Silicon**:

| # | Engine | Source | Runs here |
|---|---|---|---|
| 1 | **3D SIMP** | this repo, rewritten for speed | ✅ built into the app |
| 2 | **PyTopo3D** | [`jihoonkim888/PyTopo3D`](https://github.com/jihoonkim888/PyTopo3D), upstream | ✅ built into the app |
| 3 | **ToPy** | [`williamhunter/ToPy`](https://github.com/williamhunter/ToPy), upstream, ported to Python 3 here | ✅ built into the app |

---

## Quick start

From inside this `mac-studio` folder (after cloning the repo):

```bash
bash setup.sh
```

```bash
bash run.sh
```

Then open <http://localhost:8501>.

Upload your STL (or pick one already in the folder), choose which axis points
up, set the resolution, pick a load case, pick an engine, set the sliders, and
press **Run optimization**. You get a live 3-D preview and an STL download.

---

## Results on the tracker holder

The whole point of this build was to optimize
`USE THIS Tracker holder FINAL (same as V0.9).stl` as a **pillar being squashed
from the top**. 30+ runs across all three engines are in `runs/`.

* **[`FINDINGS.md`](FINDINGS.md)** — what was learned: which settings matter,
  which load cases work, how the three engines compare, and what went wrong.
* **[`REPORT.md`](REPORT.md)** — auto-generated table of every run with its
  exact settings, plus contact-sheet images.

The short version of the recipe that produces the good-looking result:

> Multiple load cases (squash **plus** sideways sway) · volume fraction
> **0.20–0.26** · filter radius **3.5–6 voxels** · Heaviside projection on ·
> mounting base and top rim forced solid as *keep* regions.

A pillar squashed straight down and clamped across its whole base optimizes to a
**solid slab** — that is the correct answer to that question, and it is why the
sideways load cases matter. See §1 of `FINDINGS.md`.

---

## Repository layout

```
app.py                    the Streamlit UI (upload STL, set settings, run, download)
run.sh / setup.sh         launch / one-time install
engines/
  simp_fast.py            engine 1: 3D SIMP, rewritten (AMG+CG, multi-load-case,
                          Heaviside projection, keep regions)
  simp3d.py               the original engine, kept for reference
  mesh_io.py              STL -> voxel design domain; density -> smoothed STL
  topy_compat.py          makes ToPy importable and fast on Python 3
runners/
  run_simp.py             engine 1 job runner + the load-case presets
  run_pytopo3d.py         engine 2 job runner
  run_topy.py             engine 3 job runner
  batch.py                run many jobs in parallel, one process each
  make_report.py          collect runs/ into REPORT.md
external/
  PyTopo3D/               upstream, installed with --no-deps (see below)
  ToPy/                   upstream, vendored and ported to Python 3
  pysparse_shim/          SciPy-backed stand-in for PySparse, for ToPy
                          (PyTopo3D is not stored here; setup.sh clones it)
best/                     the best optimized STLs
previews/                 contact sheets used by REPORT.md
input_tracker_holder.stl  the original part
jobs/                     the job specs for each wave of runs
runs/<id>/                per run: preview.png and record.json (settings + full
                          convergence history). The per-run .stl and density.npy
                          files are regenerated locally and not stored in git.
```

## Scripting it without the UI

```python
from engines import simp_fast as sf, mesh_io

domain, meta = mesh_io.stl_to_domain("my_part.stl", resolution=110, up_axis="z")
res = sf.optimize(
    domain=domain, volfrac=0.24, penal=3.0, rmin=4.0, max_iter=70,
    projection=True, beta_max=32.0,
    supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
    load_cases=[[{"face": "z+", "pattern": "all", "dir": "-z"}],
                [{"face": "z+", "pattern": "all", "dir": "+x"}],
                [{"face": "z+", "pattern": "all", "dir": "+y"}]],
    case_weights=[1.0, 0.3, 0.3])
mesh, info = mesh_io.density_to_mesh(res["rho"], pitch=meta["pitch"], smooth_iters=10)
mesh.export("optimized.stl")
```

Or run a whole sweep:

```bash
.venv/bin/python runners/batch.py jobs/wave4.json 3
```

## Notes on the macOS install

* **PyTopo3D** depends on `pypardiso`, which needs Intel MKL and has no
  Apple-Silicon build. `setup.sh` installs it with `--no-deps`; upstream already
  falls back to SciPy's solver by itself. The runner additionally swaps in an
  AMG-preconditioned CG solver so higher resolutions are usable.
* **ToPy** upstream is Python 2.7 + PySparse (unmaintained, no arm64 build). It
  is vendored here, mechanically ported to Python 3, and backed by
  `external/pysparse_shim/`. **The `topy` package on PyPI is an unrelated
  typo-fixing tool** — `pip install topy` gets you the wrong thing.
* **Z88 Arion** is Windows-only in practice and **ToOptiX** needs the FreeCAD
  GUI, so neither is wired into this app.
