#!/usr/bin/env python
"""
Run one 3D-SIMP job from a JSON config and write everything a report needs:
optimized STL, density field, preview render, convergence history.

    python runners/run_simp.py config.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines import simp_fast as sf
from engines import mesh_io
from render import render_mesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- load presets
# Every preset is "the force comes from the top". They differ in how the load
# enters and where the base is held, which is what decides whether you get a
# boring slab or a braced, organic structure.
LOAD_PRESETS = {
    "axial_full_base": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "all", "dir": "-z"}]],
        blurb="Whole top pressed down, whole base clamped (pure axial squash)."),
    "axial_corner_feet": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.22, "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "all", "dir": "-z"}]],
        blurb="Whole top pressed down, base held only at four corner feet."),
    "axial_ring_base": dict(
        supports=[{"face": "z-", "pattern": "ring", "frac": 0.18, "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "all", "dir": "-z"}]],
        blurb="Whole top pressed down, base held on a perimeter ring."),
    "pad_on_full_base": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "center", "frac": 0.35, "dir": "-z"}]],
        blurb="Load concentrated on a central pad on top, whole base clamped."),
    "pad_on_corner_feet": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.22, "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "center", "frac": 0.35, "dir": "-z"}]],
        blurb="Central top pad down onto four corner feet (X-brace territory)."),
    "squash_plus_sway": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
        ],
        case_weights=[1.0, 0.35],
        blurb="Axial squash plus a second load case pushing the top sideways in X."),
    "squash_plus_sway_both": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.3, 0.3],
        blurb="Axial squash plus sideways sway in both X and Y (3 load cases)."),
    "squash_plus_twist": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.22, "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "strip_x", "frac": 0.3, "dir": "+y"},
             {"face": "z+", "pattern": "strip_y", "frac": 0.3, "dir": "-x"}],
        ],
        case_weights=[1.0, 0.3],
        blurb="Axial squash on corner feet plus a twisting second case."),
    "eccentric_back_edge": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[[{"face": "z+", "pattern": "edge_v0", "frac": 0.35, "dir": "-z"}]],
        blurb="Load on the back edge of the top, spreading down into the full base."),
    "sway_heavy": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.7, 0.7],
        blurb="Axial squash with heavy sway in both X and Y "
              "(sway weighted 0.7 instead of 0.3)."),
    "sway_light": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.15, 0.15],
        blurb="Axial squash with only a light sway in X and Y "
              "(sway weighted 0.15) — closest to pure compression that "
              "still produces structure."),
    "sway_y_only": dict(
        supports=[{"face": "z-", "pattern": "all", "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.35],
        blurb="Axial squash plus sway out of the back plate's plane (Y) only."),
    "sway_on_feet": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.25, "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.3, 0.3],
        blurb="Squash + sway in X and Y, standing on four corner feet."),
    "sway_pinned_base": dict(
        # Z held across the whole base (a bearing surface), but X and Y held
        # only at the corners. Holding Z alone leaves the part free to slide and
        # spin under the sway cases -- a mechanism, with no equilibrium and a
        # singular stiffness matrix. The corner pins are the minimum restraint
        # that makes the problem well posed.
        supports=[{"face": "z-", "pattern": "all", "dofs": "z"},
                  {"face": "z-", "pattern": "corners", "frac": 0.15,
                   "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
        ],
        case_weights=[1.0, 0.3, 0.3],
        blurb="Squash + sway, base free to slide sideways (only Z held) — "
              "the base is a bearing surface, not a weld."),
    "feet_omni": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.20, "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
            [{"face": "z+", "pattern": "strip_u", "frac": 0.3, "dir": "+y"},
             {"face": "z+", "pattern": "strip_v", "frac": 0.3, "dir": "-x"}],
        ],
        case_weights=[1.0, 0.3, 0.3, 0.25],
        blurb="Squash + sway in X and Y + a twist, all standing on four corner "
              "feet. The most demanding case here: the load has to reach four "
              "small feet while resisting push from every direction, which is "
              "what forces diagonal bracing on every face."),
    "feet_omni_wide": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.32, "dofs": "xyz"}],
        load_cases=[
            [{"face": "z+", "pattern": "all", "dir": "-z"}],
            [{"face": "z+", "pattern": "all", "dir": "+x"}],
            [{"face": "z+", "pattern": "all", "dir": "+y"}],
            [{"face": "z+", "pattern": "strip_u", "frac": 0.3, "dir": "+y"},
             {"face": "z+", "pattern": "strip_v", "frac": 0.3, "dir": "-x"}],
        ],
        case_weights=[1.0, 0.3, 0.3, 0.25],
        blurb="As feet_omni but with wider feet — a more printable base."),
}


def build_keep(domain, specs):
    """
    Force some elements solid (non-designable), so the part keeps the features
    that make it usable while the rest is optimized away.

    Each spec is a dict:
      {"type": "box",  "x": [0,1], "y": [0,1], "z": [0,0.1]}   fractional box
      {"type": "skin", "thickness": 1, "faces": "xy"}          outer skin layers
    """
    import numpy as _np
    keep = _np.zeros(domain.shape, dtype=bool)
    nx, ny, nz = domain.shape
    for spec in specs or []:
        kind = spec.get("type", "box")
        if kind == "box":
            sl = []
            for n, ax in zip((nx, ny, nz), "xyz"):
                lo, hi = spec.get(ax, [0.0, 1.0])
                sl.append(slice(int(round(lo * n)), max(int(round(hi * n)), 1)))
            keep[tuple(sl)] = True
        elif kind == "skin":
            t = int(spec.get("thickness", 1))
            from scipy.ndimage import binary_erosion
            core = binary_erosion(domain, iterations=t)
            keep |= domain & ~core
        else:
            raise ValueError("unknown keep spec %r" % kind)
    return keep & domain


def build_config(**kw):
    cfg = dict(
        id="run", engine="simp", stl="input_tracker_holder.stl",
        resolution=72, up_axis="z",
        volfrac=0.35, penal=3.0, rmin=2.2, max_iter=50, tol=0.004, move=0.2,
        preset="axial_full_base", projection=True, beta_max=16.0,
        beta_double_every=12, penal_continuation=True, seed_noise=0.0,
        threshold=0.5, smooth_iters=8, keep_largest=True, note="",
        keep=None,
    )
    cfg.update(kw)
    return cfg


def run(cfg, outdir=None):
    t0 = time.time()
    outdir = outdir or os.path.join(ROOT, "runs", cfg["id"])
    os.makedirs(outdir, exist_ok=True)

    stl = cfg["stl"] if os.path.isabs(cfg["stl"]) else os.path.join(ROOT, cfg["stl"])
    domain, meta = mesh_io.stl_to_domain(stl, resolution=cfg["resolution"],
                                         up_axis=cfg.get("up_axis", "z"))

    preset = LOAD_PRESETS[cfg["preset"]]
    keep = build_keep(domain, cfg.get("keep")) if cfg.get("keep") else None
    res = sf.optimize(
        domain=domain, keep=keep,
        volfrac=cfg["volfrac"], penal=cfg["penal"], rmin=cfg["rmin"],
        max_iter=cfg["max_iter"], tol=cfg["tol"], move=cfg.get("move", 0.2),
        supports=preset["supports"], load_cases=preset["load_cases"],
        case_weights=preset.get("case_weights"),
        projection=cfg["projection"], beta_max=cfg["beta_max"],
        beta_double_every=cfg["beta_double_every"],
        penal_continuation=cfg["penal_continuation"],
        seed_noise=cfg.get("seed_noise", 0.0),
        verbose=cfg.get("verbose", True),
    )

    np.save(os.path.join(outdir, "density.npy"), res["rho"].astype(np.float32))

    mesh, minfo = mesh_io.density_to_mesh(
        res["rho"], threshold=cfg["threshold"], pitch=meta["pitch"],
        smooth_iters=cfg["smooth_iters"], keep_largest=cfg["keep_largest"])
    stl_out = os.path.join(outdir, "%s.stl" % cfg["id"])
    mesh_io.save_stl(mesh, stl_out)

    title = "%s | %s | vf=%.2f p=%.1f rmin=%.1f res=%d" % (
        cfg["id"], cfg["preset"], cfg["volfrac"], cfg["penal"], cfg["rmin"],
        cfg["resolution"])
    png = os.path.join(outdir, "preview.png")
    render_mesh(mesh.vertices, mesh.faces, png, title)

    record = dict(
        config=cfg, preset_blurb=preset["blurb"], voxel=meta, mesh=minfo,
        elements=res["na"], dofs=res["nfree"],
        keep_elements=int(keep.sum()) if keep is not None else 0,
        keep_fraction=res.get("keep_fraction", 0.0),
        free_volfrac=res.get("free_volfrac"),
        compliance=res["compliance"], volume_fraction=res["volume"],
        stiffness_vs_solid=None,
        seconds=res["seconds"], solve_seconds=res["solve_seconds"],
        iterations=len(res["hist"]), history=res["hist"],
        part_volume_mm3=minfo["volume_mm3"],
        original_volume_mm3=meta["mesh_volume_mm3"],
        mass_saving=1.0 - minfo["volume_mm3"] / meta["mesh_volume_mm3"]
        if meta["mesh_volume_mm3"] else None,
        stl=os.path.relpath(stl_out, ROOT),
        preview=os.path.relpath(png, ROOT),
        wall_seconds=time.time() - t0,
    )
    with open(os.path.join(outdir, "record.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        cfg = build_config(**json.load(f))
    try:
        rec = run(cfg)
        print("DONE %s  C=%.4e  vol=%.3f  %.0fs" %
              (cfg["id"], rec["compliance"], rec["volume_fraction"],
               rec["wall_seconds"]), flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
