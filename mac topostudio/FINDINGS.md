# Topology optimization of the tracker holder — what was run and what came out

**Part:** `USE THIS Tracker holder FINAL  (same as V0.9).stl` — 33 × 24 × 78 mm,
watertight, 28,521 mm³ of material, a twin-pocket channel with a ~3.7 mm back
plate and side rails.

**Load story, for every run:** the force comes from the top and the part stands
on its base — a pillar being squashed.

**Machine:** Apple M5, 10 cores, 16 GB. Everything below ran locally.

---

## 1. The single most important finding

**A pillar squashed straight down, clamped across its whole base, has vertical
columns as its exact optimum.** Compliance minimization under pure axial
compression does not produce a lattice — it produces a solid slab, because
routing the load anywhere other than straight down is strictly worse. Three
separate runs confirmed it (`w1_axial_full_base`, `w1_axial_ring_base`,
`w1_axial_corner_feet` — all came back as near-solid plates at 30% volume).

That is correct physics, not a solver failure, and it is the thing to know
before turning any dial.

**What produces the topology-optimized look is giving the structure something to
resist sideways.** The fix used throughout the rest of this work is *multiple
load cases*: case 1 is the vertical squash, cases 2 and 3 push the top sideways
in X and Y, and the optimizer must find one structure that survives all of them.
That is what forces diagonal bracing, branching, and the organic voids.

---

## 2. Which knobs actually changed the look

All comparisons at resolution 88, preset `squash_plus_sway_both`, 55 iterations.

| Knob | What it did | Verdict |
|---|---|---|
| **Volume fraction** | 0.45 → solid slab. 0.30 → a few holes. 0.20 → real branching struts and open voids. | **Biggest lever.** 0.16–0.26 is the interesting band. |
| **Filter radius `r_min`** | 1.5 → nearly solid plate (too weak to consolidate material). 2.2 → some holes. 3.5 → large clean organic voids and thick branching members. | **Second biggest, and counterintuitive** — *wider* filter looks *more* topology-optimized, because it forces material into few thick members instead of smearing it into a plate. |
| **Keep regions** | Forcing the mounting base and top rim solid turned a perforated plate into a proper truss between two solid ends. | **Third lever, and the one that made it click.** |
| **Heaviside projection** | Off → blurry, plate-like, grey. On (β→16/32) → crisp struts with clean edges. | Always leave on. |
| **Penalization `p`** | 3.0 → standard. 5.0 → slightly crisper, fewer members, no real gain over projection. | Leave at 3. |
| **Resolution** | 64 → coarse. 88 → good. 120 → finer members, smoother surfaces. | Buys detail, costs time. |
| **Support pattern** | Whole base → plate. Corner feet → branching legs and diagonal bracing. | Corner feet are the more dramatic look; whole base is the more printable one. |

**The recipe that works:** multi-load-case + volume fraction ≈ 0.20–0.26 +
`r_min` ≈ 3.5–6 voxels + Heaviside projection + solid keep regions at the
mounting ends.

---

## 3. Load-case presets tried

Every one of these is "force from the top"; they differ in how the load enters
and where the base is held.

| Preset | Description | Result |
|---|---|---|
| `axial_full_base` | Whole top down, whole base clamped | Solid slab |
| `axial_ring_base` | Whole top down, base held on a perimeter ring | Solid slab |
| `axial_corner_feet` | Whole top down, base on four corner feet | Slab with two legs |
| `eccentric_back_edge` | Load on the top's back edge only | Slab |
| `squash_plus_sway` | Squash + sway in X only | Nearly a slab — X sway is resisted in the back plate's own plane |
| `squash_plus_sway_both` | Squash + sway in X **and** Y | **Real structure.** Y sway is out-of-plane, which is what forces 3-D bracing |
| `sway_light` / `sway_heavy` | Same, sway weighted 0.15 / 0.70 | Both work; weight changes how aggressive the bracing is |
| `sway_y_only` | Squash + out-of-plane sway only | Works — confirms Y is the case that matters |
| `sway_on_feet` | Squash + sway on four corner feet | Branching legs, most sculptural of the wave-2 set |
| `squash_plus_twist` | Squash + a twisting case, on corner feet | Tall splayed legs |
| `sway_pinned_base` | Base held in Z only | **Ill-posed** — see §7 |
| `feet_omni` | Squash + sway X + sway Y + twist, on corner feet | Used for the hero run |

---

## 4. How stiff are they, really?

The optimizer's own compliance number is measured on a *grey* density field
mid-run. `runners/validate.py` re-solves from scratch on the **thresholded 0/1
design that was actually exported to STL**, and against the untouched solid part
as the reference. Every design is scored twice: under the load case it was
optimized for, and under one **common reference case**
(`squash_plus_sway_both`, whole base clamped) so runs that used different
presets sit on the same axis.

`stiff/mass` above 1.0 means the optimized part is **stiffer per gram** than the
solid original.

| Run | Engine | Mass kept | Stiffness kept (own case) | Stiffness kept (common case) | stiff/mass (common) |
|---|---|---:|---:|---:|---:|
| `w2_rmin35` | 3D SIMP | 30% | 41.8% | 41.8% | **1.38** |
| `f4_feet` | 3D SIMP | 20% | 31.1% | 25.6% | **1.28** |
| `w2_vf20` | 3D SIMP | 20% | 27.5% | 27.5% | **1.36** |
| `w2_story_sway_on_feet` | 3D SIMP | 30% | 47.6% | 40.7% | **1.35** |
| `f2_lace` | 3D SIMP | 26% | 18.1% | 18.1% | 0.70 |
| `w2_keepends` | 3D SIMP | 25% | 14.2% | 14.2% | 0.56 |
| `f3_chunky` | 3D SIMP | 24% | 10.8% | 10.8% | 0.45 |
| `f1_hero` | 3D SIMP | 20% | 4.6% | 3.0% | 0.15 |
| `w3_topy_baseline` | ToPy | 30% | 15.6% | 15.6% | 0.52 |
| `f6_topy_chunky` | ToPy | 28% | 5.9% | 5.9% | 0.21 |
| `f5_topy_keep` | ToPy | 26% | 4.2% | 4.2% | 0.16 |
| `f7_pt_long` | PyTopo3D | 22% | 0.2% | 0.2% | 0.01 |
| `f8_pt_wide` | PyTopo3D | 21% | **no load path** | – | – |
| `w1_axial_full_base` (the slab) | 3D SIMP | 30% | **74.7%** | **4.3%** | 0.14 |

Three things fall out of this table.

**1. The boring slab is the best possible answer — to exactly one question.**
`w1_axial_full_base` keeps 74.7% of the solid part's stiffness on 30% of the
mass under pure axial squash: nothing else comes close. Score the same slab
under the multi-directional reference load and it keeps **4.3%**. That is the
whole argument for multi-load-case optimization in one row.

**2. The prettiest result is not the strongest.** `f1_hero` is the most
sculptural thing here and structurally the weakest of the SIMP runs — very thin
members, and supports at four small corner feet. `f4_feet` is nearly as
good-looking, at the same 20% mass, and is **8× stiffer**. If the part has to
actually hold something, `f4_feet` is the one to print.

**3. Keep regions cost stiffness.** `w2_keepends` and `f2_lace`/`f3_chunky` all
spend a chunk of their volume budget on solid end blocks that carry no load
efficiently, so the remaining truss is thinner than it would otherwise be. That
is a fair trade for keeping the part usable, but it is a trade, not free.

---

## 5. The three engines

All three run natively on this Mac. They are genuinely different programs, not
three names for the same code.

| | 3D SIMP (this repo) | PyTopo3D | ToPy |
|---|---|---|---|
| Source | rewritten here | `jihoonkim888/PyTopo3D`, upstream | `williamhunter/ToPy`, upstream |
| Method | SIMP + OC + density filter + Heaviside projection | SIMP + OC + sensitivity filter | SIMP + OC + **grey-scale filter** (`eta` damping, `q` continuation) |
| Load cases | **many, weighted** | one | one |
| Forced-solid regions | yes (`keep`) | no | yes (`ACTV_ELEM`) |
| Design domain from STL | yes | yes (`obstacle_mask`) | yes (`PASV_ELEM`) |
| Linear solver | AMG-preconditioned CG | SciPy `spsolve` upstream; AMG swapped in here | PySparse upstream; SciPy/AMG here |

### What each needed to run on an Apple-Silicon Mac

* **3D SIMP** — rewritten (see §7). No external dependency beyond NumPy/SciPy/pyamg.
* **PyTopo3D** — installs fine except for `pypardiso`, which requires Intel MKL
  and has no arm64 build. Installed with `--no-deps`; upstream already falls back
  to SciPy `spsolve` on its own. An AMG solver is swapped in here to lift the
  resolution ceiling.
* **ToPy** — the hardest. Upstream is Python 2.7 and depends on PySparse, which
  is unmaintained and has no arm64 build. Two pieces of work were needed:
  1. `external/pysparse_shim/` — a stand-in package providing
     `spmatrix.ll_mat_sym`, `superlu.factorize`, `itsolvers.pcg` and
     `precon.ssor`, backed by SciPy and pyamg.
  2. `external/ToPy/` vendored and mechanically ported to Python 3 (`xrange`,
     `has_key`, `np.in1d`), plus a vectorized `_updateK` in
     `engines/topy_compat.py` (upstream loops in Python over every element for
     every FE solve).

  Note: the `topy` package **on PyPI is a typo-fixing tool, not this ToPy** —
  the repo's `notes.txt` recommendation of `pip install topy` installs the wrong
  thing.

### How they compare on the same problem

* **ToPy** gives the cleanest, smoothest single-body results — every run came
  back as exactly 1 connected component. Its grey-scale filter is doing the same
  job as this repo's Heaviside projection.
* **PyTopo3D** produced the worst results on this part, and the FE check in §4
  says how badly: its exports keep 0.1–0.2% of the solid part's stiffness, and
  two of them are outright **singular** — disconnected fragments that cannot be
  both loaded and supported. The cause is that the thresholded design shatters:
  25–69 separate pieces, of which the exporter keeps one. More iterations (40 →
  120) did not fix it, and a wider filter made it worse (`f8_pt_wide` eroded
  away from the loaded face entirely). PyTopo3D has no Heaviside projection and
  filters only the sensitivities, so at the volume fractions that look good on
  this thin-walled part it never resolves into connected members. It is not
  broken — it is the wrong tool for a part this thin.
* **3D SIMP (this repo)** is the only one of the three that takes multiple load
  cases, which for this part is the difference between a slab and a structure.
  For the other two, the cases are summed into one force vector — a weaker
  formulation, and §4 shows the cost: ToPy's designs keep 4–16% of the solid
  stiffness where the SIMP designs at comparable mass keep 25–42%. ToPy is
  optimizing honestly, just for a different (easier, single-direction) problem.

**Verdict for this part:** use the built-in 3D SIMP engine. ToPy is worth
running for its distinctive smooth, sculpted look — it makes the prettiest
single-body shapes of the three — but score it before you print it. PyTopo3D
is best kept for chunkier parts than this one.

---

## 6. Two conventions worth knowing about

* **ToPy's `VOL_FRAC` is a fraction of the whole voxel grid**, not of the design
  elements. With ~50% of the bounding box empty, asking ToPy for 0.30 gives a
  very different part than asking the other two for 0.30. The runner rescales by
  `n_design / n_total` so "30% of the part" means the same thing everywhere.
* **ToPy's `PASV_ELEM` / `ACTV_ELEM` indexing** is `ely + elx·nely + elz·nelx·nely`
  (z outermost, then x, then y innermost), which is *not* the C-order flatten of
  its own `desvars` array. Getting this wrong silently voids the wrong elements.

---

## 7. Things that went wrong, and what they mean

* **`sway_pinned_base` is ill-posed.** Holding the base in Z only, while pushing
  the top sideways, leaves the part free to slide and spin — a mechanism with no
  equilibrium and a singular stiffness matrix. CG reported "indefinite matrix"
  and the fallback direct solve ground for 50 minutes on nonsense. The preset now
  also pins X and Y at the corners, which is the minimum restraint that makes it
  well posed.
* **Keep regions can eat the whole volume budget.** The mounting-base + top-rim
  keep regions are 15% of the design domain. A run asking for a 16% volume
  fraction therefore left 1% for the actual structure and collapsed into
  disconnected fragments. The engine now refuses this with an explanatory error
  and warns when the margin is under 3%.
* **A stale AMG hierarchy silently corrupts CG.** Reusing a multigrid hierarchy
  built for an earlier iteration's stiffness matrix is not a valid SPD
  preconditioner for the current one; CG then converges on its own preconditioned
  residual while returning a wrong displacement field. The compliance history
  looked plausible (monotone, converging) but was wrong by a factor of 17. The
  solver now checks the **true** residual `‖b − Kx‖` before accepting any answer
  and falls back to a direct solve if it fails.
* **The exported mesh is about one voxel oversized.** Extracting the 0.5
  isosurface from a binary voxel field puts the surface half a voxel outside the
  last solid cell on each side, so every STL here comes out ~0.7–0.9 mm larger
  than the 33 × 24 × 78 mm original (measured: 33.8 × 24.7 × 78.2 for
  `f1_hero`). It shrinks with resolution. If exact outer dimensions matter for
  fit, raise the resolution, or raise the density threshold above 0.5 in the
  app's Advanced panel.
* **Load patterns can miss the geometry.** A "central pad on top" load found no
  material, because at the very top of this part only the back plate exists.
  Those runs now fail immediately with a message naming the load case, rather
  than producing garbage.

---

## 8. What was changed in the built-in engine, and why

The original `engines/simp3d.py` is correct but not usable past about a 30³ grid:
it builds its density filter with a five-deep Python loop over every element
pair, and solves with `spsolve`. `engines/simp_fast.py` keeps the method and
replaces the machinery:

* element stiffness by 2×2×2 Gauss quadrature instead of a magic constant table,
  so the node ordering is ours and verifiable (rigid-body modes give zero energy;
  the assembled matrix matches a naive dense assembly);
* only elements **inside** the design domain are assembled — empty space around an
  uploaded STL costs nothing, and dofs touching no material are constrained away
  instead of being propped up by a 10⁻⁹ stiffness;
* the triplet → CSR mapping is computed once and refilled each iteration;
* AMG-preconditioned CG with the six rigid-body modes as the near-nullspace
  (translations alone are not enough — the solver stalls once the design goes
  near-0/1);
* density filter built from a small stencil, vectorized over the grid;
* multiple weighted load cases;
* Heaviside projection with β continuation;
* forced-solid `keep` regions;
* marching-cubes + Taubin smoothing for the output mesh instead of emitting cube
  faces, which is what makes the result look moulded rather than Minecraft.

**Measured effect:** a 40 × 12 × 24 validation cantilever went from 209 s with
`spsolve` to 43 s with AMG, to the identical compliance (8.5780). The engine was
validated against the textbook 3-D cantilever before any of the runs above.

---

## 9. Files

* `runs/<id>/` — for every run: the optimized `.stl`, the raw density field
  (`density.npy`), a three-view `preview.png`, and `record.json` with the exact
  settings and the full convergence history.
* [`REPORT.md`](REPORT.md) — auto-generated table of every run with its settings.
* `previews/` — contact sheets.
