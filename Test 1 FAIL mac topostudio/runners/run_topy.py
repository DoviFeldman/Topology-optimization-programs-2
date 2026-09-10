#!/usr/bin/env python
"""
Engine #3 — ToPy (williamhunter/ToPy), run through `engines/topy_compat`.

ToPy is a different optimizer from the other two, not just a different
implementation: it uses an OC update with a *grey-scale filter* (GSF), its own
`eta` damping factor, and `q`-continuation on top of the usual `p`. Those are
the knobs varied here.

Geometry from an STL is fed in as ToPy's own PASV_ELEM (passive/void elements),
so material can only be placed inside the part — the same design domain the
other two engines get.

ToPy solves a single load case, so a multi-case preset is combined into one
load vector (noted in the report).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines import mesh_io, simp_fast as sf, topy_compat
from render import render_mesh
from runners.run_simp import LOAD_PRESETS, build_config, build_keep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _my_nodes_to_topy(node_idx, nx, ny, nz):
    """Our node index (ix + iy*nx1 + iz*nx1*ny1) -> ToPy's 1-based node id."""
    nx1, ny1 = nx + 1, ny + 1
    ix = node_idx % nx1
    iy = (node_idx // nx1) % ny1
    iz = node_idx // (nx1 * ny1)
    return 1 + iy + ix * ny1 + iz * nx1 * ny1


def _bc_for_topy(preset, domain, nx, ny, nz):
    """Turn the shared face/pattern presets into ToPy node lists and values."""
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1

    # only nodes that actually touch material may carry a BC
    live = np.zeros((nx1, ny1, nz1), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                live[dx:dx + nx, dy:dy + ny, dz:dz + nz] |= domain
    live_flat = live.reshape(-1, order="F")

    fixed_nodes = {"x": [], "y": [], "z": []}
    for s in preset["supports"]:
        fx, _F, _ = sf.build_bc(nx, ny, nz, [s], [])
        fx = fx.reshape(-1, 3)
        for k, ax in enumerate("xyz"):
            sel = np.flatnonzero(fx[:, k] & live_flat)
            if len(sel):
                fixed_nodes[ax].append(_my_nodes_to_topy(sel, nx, ny, nz))

    # combine every load case into one vector (ToPy is single-case)
    ndof = 3 * nx1 * ny1 * nz1
    Ftot = np.zeros(ndof)
    weights = preset.get("case_weights") or [1.0] * len(preset["load_cases"])
    for lc, w in zip(preset["load_cases"], weights):
        _fx, F, _ = sf.build_bc(nx, ny, nz, [], lc)
        F[~np.repeat(live_flat, 3)] = 0.0
        s = np.abs(F).sum()
        if s:
            Ftot += F / s * w

    Fn = Ftot.reshape(-1, 3)
    load_nodes, load_vals = {}, {}
    for k, ax in enumerate("xyz"):
        sel = np.flatnonzero(Fn[:, k])
        load_nodes[ax] = _my_nodes_to_topy(sel, nx, ny, nz) if len(sel) else np.array([])
        load_vals[ax] = Fn[sel, k] if len(sel) else np.array([])
    if sum(len(v) for v in load_vals.values()) == 0:
        raise ValueError("no load reaches the design domain")

    cat = lambda lst: (np.unique(np.concatenate(lst)) if lst else np.array([]))
    return ({ax: cat(fixed_nodes[ax]) for ax in "xyz"}, load_nodes, load_vals)


def run(cfg, outdir=None):
    t0 = time.time()
    outdir = outdir or os.path.join(ROOT, "runs", cfg["id"])
    os.makedirs(outdir, exist_ok=True)

    topy = topy_compat.install()
    from topy.parser import config2dict

    stl = cfg["stl"] if os.path.isabs(cfg["stl"]) else os.path.join(ROOT, cfg["stl"])
    domain, meta = mesh_io.stl_to_domain(stl, resolution=cfg["resolution"],
                                         up_axis=cfg.get("up_axis", "z"))
    nx, ny, nz = domain.shape
    preset = LOAD_PRESETS[cfg["preset"]]
    fixed, lnodes, lvals = _bc_for_topy(preset, domain, nx, ny, nz)

    # ToPy's PASV_ELEM is indexed in the order its own remap table enumerates
    # elements: z outermost, then x, then y innermost (see
    # Topology.update_desvars_oc). That is  ely + elx*nely + elz*nelx*nely.
    ix, iy, iz = np.nonzero(~domain)
    pasv = (iy + ix * ny + iz * nx * ny).astype(int)

    # ToPy applies VOL_FRAC to the WHOLE grid, while this repo's engine and
    # PyTopo3D apply it to the design elements only. Rescale so that "30% of the
    # part" means the same thing in all three.
    n_total = nx * ny * nz
    n_design = int(domain.sum())
    volfrac_topy = float(cfg["volfrac"]) * n_design / n_total

    # "keep" regions map onto ToPy's own ACTV_ELEM (forced-solid elements),
    # indexed the same way as PASV_ELEM.
    actv = np.array([], dtype=int)
    keep_mask = build_keep(domain, cfg.get("keep")) if cfg.get("keep") else None
    if keep_mask is not None and keep_mask.any():
        kx, ky, kz = np.nonzero(keep_mask)
        actv = (ky + kx * ny + kz * nx * ny).astype(int)

    conf = {
        "PROB_TYPE": "comp", "PROB_NAME": cfg["id"],
        "ETA": str(cfg.get("eta", "0.4")), "DOF_PN": 3,
        "VOL_FRAC": volfrac_topy,
        "FILT_RAD": float(cfg["rmin"]),
        "P_FAC": float(cfg.get("p_start", 1.0)),
        "ELEM_K": "H8",
        "NUM_ELEM_X": nx, "NUM_ELEM_Y": ny, "NUM_ELEM_Z": nz,
        "NUM_ITER": int(cfg["max_iter"]),
        "FXTR_NODE_X": fixed["x"], "FXTR_NODE_Y": fixed["y"],
        "FXTR_NODE_Z": fixed["z"],
        "LOAD_NODE_X": lnodes["x"], "LOAD_NODE_Y": lnodes["y"],
        "LOAD_NODE_Z": lnodes["z"],
        "LOAD_VALU_X": lvals["x"], "LOAD_VALU_Y": lvals["y"],
        "LOAD_VALU_Z": lvals["z"],
        # ToPy's p- and q-continuation (its grey-scale filter)
        "P_MAX": float(cfg["penal"]), "P_HOLD": int(cfg.get("p_hold", 8)),
        "P_INCR": float(cfg.get("p_incr", 0.2)), "P_CON": float(cfg.get("p_con", 1)),
        "Q_FAC": 1.0, "Q_MAX": float(cfg.get("q_max", 2.0)),
        "Q_HOLD": int(cfg.get("q_hold", 12)), "Q_INCR": float(cfg.get("q_incr", 0.05)),
        "Q_CON": float(cfg.get("q_con", 1)),
        "PASV_ELEM": pasv, "ACTV_ELEM": actv,
    }
    if cfg.get("approx"):
        conf["APPROX"] = cfg["approx"]

    t = topy.Topology(config=conf)
    t.set_top_params()

    hist = []
    for i in range(int(cfg["max_iter"])):
        t.fea()
        t.sens_analysis()
        t.filter_sens_sigmund()
        t.update_desvars_oc()
        hist.append(dict(it=i + 1, objective=float(t.objfval),
                         vol=float(t.desvars.mean()), change=float(t.change),
                         p=float(t.p), q=float(t.q), eta=float(np.mean(t.eta))))
        print("  it %3d  obj=%.4e  vol=%.3f  chg=%.3f  p=%.2f  q=%.2f" %
              (i + 1, t.objfval, t.desvars.mean(), t.change, t.p, t.q), flush=True)
        if t.change < cfg.get("tol", 0.01) and i > 10:
            break

    # ToPy desvars are (nelz, nely, nelx) -> back to our (nx, ny, nz)
    rho = np.transpose(np.asarray(t.desvars), (2, 1, 0))
    rho[~domain] = 0.0
    np.save(os.path.join(outdir, "density.npy"), rho.astype(np.float32))

    mesh, minfo = mesh_io.density_to_mesh(
        rho, threshold=cfg["threshold"], pitch=meta["pitch"],
        smooth_iters=cfg["smooth_iters"], keep_largest=cfg["keep_largest"])
    stl_out = os.path.join(outdir, "%s.stl" % cfg["id"])
    mesh_io.save_stl(mesh, stl_out)
    png = os.path.join(outdir, "preview.png")
    render_mesh(mesh.vertices, mesh.faces, png,
                "%s | ToPy | %s | vf=%.2f eta=%s q_max=%.1f res=%d" %
                (cfg["id"], cfg["preset"], cfg["volfrac"], cfg.get("eta", "0.4"),
                 cfg.get("q_max", 2.0), cfg["resolution"]), color="#8fbf7f")

    record = dict(
        config=cfg, engine_name="ToPy 0.4.0 (ported to Python 3 + SciPy)",
        preset_blurb=preset["blurb"] + "  [load cases combined into one]",
        voxel=meta, mesh=minfo, elements=int(domain.sum()),
        objective=float(t.objfval), volume_fraction=float(rho[domain].mean()),
        volfrac_passed_to_topy=volfrac_topy,
        keep_elements=int(len(actv)),
        volfrac_note="ToPy's VOL_FRAC covers the whole voxel grid; it was "
                     "rescaled by n_design/n_total so the target matches the "
                     "other two engines.",
        iterations=len(hist), history=hist,
        part_volume_mm3=minfo["volume_mm3"],
        original_volume_mm3=meta["mesh_volume_mm3"],
        mass_saving=1.0 - minfo["volume_mm3"] / meta["mesh_volume_mm3"]
        if meta["mesh_volume_mm3"] else None,
        stl=os.path.relpath(stl_out, ROOT), preview=os.path.relpath(png, ROOT),
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
        print("DONE %s  obj=%.4e  vol=%.3f  %.0fs" %
              (cfg["id"], rec["objective"], rec["volume_fraction"],
               rec["wall_seconds"]), flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
