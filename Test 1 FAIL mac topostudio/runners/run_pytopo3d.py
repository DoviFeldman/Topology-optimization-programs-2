#!/usr/bin/env python
"""
Engine #2 — PyTopo3D (jihoonkim888/PyTopo3D), driven from the same job config
as the built-in engine so the runs are comparable.

PyTopo3D is the real upstream package: we hand it an `obstacle_mask` built from
the uploaded STL (so material can only go inside the part), a `support_mask`,
and a `force_field`, and let its own SIMP loop run.

Two notes on making it work on an Apple-Silicon Mac:

  * upstream prefers PyPardiso, which needs Intel MKL and has no arm64 build.
    It falls back to SciPy `spsolve` on its own — correct, but single-core and
    with heavy 3-D fill-in, so it caps the usable resolution.
  * `--amg` swaps in the same AMG-preconditioned CG this repo's own engine uses,
    which lifts that cap. It is verified against the true residual and falls
    back to `spsolve` whenever AMG does not converge, so the answer is the same
    either way.

PyTopo3D takes a single load case, so a `sway` preset's cases are combined into
one force vector here (that is a weaker formulation than true multi-case, and
the report says so).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from engines import mesh_io, simp_fast as sf
from render import render_mesh
from runners.run_simp import LOAD_PRESETS, build_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_amg_solver(free_dofs, grid):
    """Point PyTopo3D's solver factory at our verified AMG+CG solver."""
    import pytopo3d.utils.solver as S

    solver = sf._Solver(np.asarray(free_dofs), grid)
    calls = {"n": 0, "fallbacks": 0}

    def _solve(K, b):
        x, nit = solver.solve(K, b)
        calls["n"] += 1
        if nit < 0:
            calls["fallbacks"] += 1
        return x

    S.solvers["cpu"] = _solve
    S.solvers["cpu_name"] = "pyamg SA-AMG + CG (installed by topopt-studio)"
    return calls


def _element_masks(domain, preset, nx, ny, nz):
    """
    Translate this repo's face/pattern boundary conditions into the element-level
    masks PyTopo3D wants. Only elements that actually exist in the design domain
    can carry a support or a load.
    """
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1

    def nodes_to_elements(node_mask_flat):
        nodesel = node_mask_flat.reshape((nx1, ny1, nz1), order="F")
        el = np.zeros((nx, ny, nz), dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    el |= nodesel[dx:dx + nx, dy:dy + ny, dz:dz + nz]
        return el & domain

    # supports
    sup = np.zeros(nx * ny * nz, dtype=bool)
    sup_nodes = np.zeros(nx1 * ny1 * nz1, dtype=bool)
    for s in preset["supports"]:
        fixed, _F, _ = sf.build_bc(nx, ny, nz, [s], [])
        sup_nodes |= fixed.reshape(-1, 3).any(axis=1)
    support_mask = nodes_to_elements(sup_nodes)

    # loads: combine every case into one force field (PyTopo3D is single-case)
    force = np.zeros((nx, ny, nz, 3))
    weights = preset.get("case_weights") or [1.0] * len(preset["load_cases"])
    for lc, w in zip(preset["load_cases"], weights):
        _fx, F, _ = sf.build_bc(nx, ny, nz, [], lc)
        F = F.reshape(-1, 3)
        live = np.abs(F).sum()
        if live == 0:
            continue
        F = F / live * w
        for comp in range(3):
            nz_nodes = np.flatnonzero(F[:, comp])
            if not len(nz_nodes):
                continue
            m = np.zeros(nx1 * ny1 * nz1)
            m[nz_nodes] = F[nz_nodes, comp]
            g = m.reshape((nx1, ny1, nz1), order="F")
            acc = np.zeros((nx, ny, nz))
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        acc += g[dx:dx + nx, dy:dy + ny, dz:dz + nz]
            force[..., comp] += acc / 8.0
    force[~domain] = 0.0
    return support_mask, force


def run(cfg, outdir=None):
    from pytopo3d.core.optimizer import top3d

    t0 = time.time()
    outdir = outdir or os.path.join(ROOT, "runs", cfg["id"])
    os.makedirs(outdir, exist_ok=True)

    stl = cfg["stl"] if os.path.isabs(cfg["stl"]) else os.path.join(ROOT, cfg["stl"])
    domain, meta = mesh_io.stl_to_domain(stl, resolution=cfg["resolution"],
                                         up_axis=cfg.get("up_axis", "z"))
    nx, ny, nz = domain.shape
    preset = LOAD_PRESETS[cfg["preset"]]
    support_mask, force = _element_masks(domain, preset, nx, ny, nz)

    # PyTopo3D indexes (nely, nelx, nelz); ours is (nx, ny, nz).
    to_p = lambda a: np.ascontiguousarray(np.transpose(a, (1, 0, 2)))
    obstacle_p = to_p(~domain)
    support_p = to_p(support_mask)
    force_p = np.ascontiguousarray(np.transpose(force, (1, 0, 2, 3)))

    solver_stats = None
    if cfg.get("amg", True):
        from pytopo3d.utils.assembly import build_supports
        ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
        freedofs0, _fixed = build_supports(nx, ny, nz, ndof, support_p)
        solver_stats = _install_amg_solver(freedofs0, (nx, ny, nz))

    rho_p = top3d(
        nelx=nx, nely=ny, nelz=nz,
        volfrac=cfg["volfrac"], penal=cfg["penal"], rmin=cfg["rmin"],
        disp_thres=cfg["threshold"],
        obstacle_mask=obstacle_p, support_mask=support_p, force_field=force_p,
        tolx=cfg.get("tol", 0.01), maxloop=cfg["max_iter"],
    )
    if isinstance(rho_p, tuple):
        rho_p = rho_p[0]
    rho = np.transpose(np.asarray(rho_p), (1, 0, 2))     # back to (nx, ny, nz)
    rho[~domain] = 0.0
    np.save(os.path.join(outdir, "density.npy"), rho.astype(np.float32))

    mesh, minfo = mesh_io.density_to_mesh(
        rho, threshold=cfg["threshold"], pitch=meta["pitch"],
        smooth_iters=cfg["smooth_iters"], keep_largest=cfg["keep_largest"])
    stl_out = os.path.join(outdir, "%s.stl" % cfg["id"])
    mesh_io.save_stl(mesh, stl_out)
    png = os.path.join(outdir, "preview.png")
    render_mesh(mesh.vertices, mesh.faces, png,
                "%s | PyTopo3D | %s | vf=%.2f p=%.1f rmin=%.1f res=%d" %
                (cfg["id"], cfg["preset"], cfg["volfrac"], cfg["penal"],
                 cfg["rmin"], cfg["resolution"]), color="#c98f5a")

    record = dict(
        config=cfg, engine_name="PyTopo3D 0.3.0 (upstream)",
        preset_blurb=preset["blurb"] + "  [load cases combined into one]",
        voxel=meta, mesh=minfo,
        elements=int(domain.sum()),
        volume_fraction=float(rho[domain].mean()),
        solver="pyamg SA-AMG + CG" if cfg.get("amg", True) else "SciPy spsolve",
        solver_fallbacks=solver_stats["fallbacks"] if solver_stats else None,
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
        print("DONE %s  vol=%.3f  %.0fs" %
              (cfg["id"], rec["volume_fraction"], rec["wall_seconds"]), flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
