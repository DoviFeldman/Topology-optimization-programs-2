"""
STL in -> optimized STL out, in one call.

Everything is expressed relative to the part (member size as a fraction of the
part's longest side, detail as an element budget), so a quick preview and a
final run are the SAME design problem at two resolutions: the preview shows
you what the final will look like, and the final adds crispness -- not a
different shape.
"""
from __future__ import annotations

import time

import numpy as np

from . import surface, topopt, voxel

# ---------------------------------------------------------------- the looks
STYLES = {
    "skeleton": dict(volfrac=0.12, member=0.10,
                     blurb="Thin branching bones with lots of open space. "
                           "Lightest, most dramatic."),
    "organic": dict(volfrac=0.16, member=0.14,
                    blurb="The classic generative-design look: smooth limbs "
                          "that fork and merge."),
    "sculpted": dict(volfrac=0.28, member=0.16,
                     blurb="Thick flowing vaults and arches, few openings. "
                           "Strongest and easiest to print."),
    "lace": dict(volfrac=0.14, member=0.05,
                 blurb="A thin skin pierced by many small holes. Wants the "
                       "'fine' detail level."),
}

# element budgets: (elements in the design domain, quality schedule)
DETAIL = {
    "preview": dict(elements=14_000, quality="preview"),
    "standard": dict(elements=40_000, quality="final"),
    "fine": dict(elements=110_000, quality="final"),
}

_DOWN = [{"face": "z+", "dir": "-z"}]
_SWAY = [[{"face": "z+", "dir": "-z"}], [{"face": "z+", "dir": "+x"}],
         [{"face": "z+", "dir": "+y"}]]


def _pad(face, pattern, frac, d):
    return {"face": face, "pattern": pattern, "frac": frac, "dir": d}


LOADS = {
    "stand": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.28}],
        load_cases=[[_pad("z+", "center", 0.45, "-z")],
                    [_pad("z+", "center", 0.45, "+x")],
                    [_pad("z+", "center", 0.45, "+y")]],
        weights=[1.0, 0.35, 0.35],
        blurb="Stands on four corner feet; a pad on top is pressed down and "
              "pushed sideways. Gives legs that branch up to the pad."),
    "pillar": dict(
        supports=[{"face": "z-"}], load_cases=_SWAY, weights=[1.0, 0.3, 0.3],
        blurb="Whole base held, whole top pressed down and pushed sideways."),
    "pillar_feet": dict(
        supports=[{"face": "z-", "pattern": "corners", "frac": 0.28}],
        load_cases=_SWAY, weights=[1.0, 0.3, 0.3],
        blurb="Whole top loaded, standing on four corner feet."),
    "cantilever": dict(
        supports=[{"face": "x-"}],
        load_cases=[[_pad("x+", "center", 0.5, "-z")],
                    [_pad("x+", "center", 0.5, "+y")]],
        weights=[1.0, 0.3],
        blurb="Bolted to a wall on its left (x-) face, loaded on the far end."),
    "shelf": dict(
        supports=[{"face": "x-"}],
        load_cases=[[{"face": "z+", "dir": "-z"}],
                    [{"face": "z+", "dir": "+y"}]],
        weights=[1.0, 0.25],
        blurb="Bolted to a wall on its left (x-) face, carrying weight on top."),
    "hanger": dict(
        supports=[{"face": "z+"}],
        load_cases=[[_pad("x+", "center", 0.6, "-z")],
                    [_pad("x+", "center", 0.6, "+y")]],
        weights=[1.0, 0.3],
        blurb="Bolted by its top face, carrying a load on its far (x+) end "
              "-- an L-bracket hanging from a ceiling."),
    "bridge": dict(
        supports=[{"face": "z-", "pattern": "edge_u0", "frac": 0.15},
                  {"face": "z-", "pattern": "edge_u1", "frac": 0.15}],
        load_cases=[[_pad("z+", "strip_u", 0.3, "-z")],
                    [_pad("z+", "strip_u", 0.3, "+y")]],
        weights=[1.0, 0.25],
        blurb="Rests on its two ends (along x), loaded across the middle."),
}


def choose_resolution(mesh, up_axis, elements):
    """Voxels along the longest side so the part has about `elements` voxels."""
    occ, g = voxel.voxelize(mesh, 40, up_axis)
    vol = occ.sum() * g["pitch"] ** 3
    pitch = (vol / float(elements)) ** (1.0 / 3.0)
    return int(np.clip(round(max(g["extents"]) / pitch), 16, 220))


def build_keep(domain, supports, load_cases, thickness):
    """Solid pads under every support and load so mounting faces survive."""
    keep = np.zeros_like(domain)
    if thickness <= 0:
        return keep
    for spec in list(supports) + [ld for lc in load_cases for ld in lc]:
        keep |= topopt.face_elements(domain, spec, depth=int(thickness))
    return keep


def run(stl, up_axis="z", style="organic", load="stand", detail="standard",
        volfrac=None, member=None, supports=None, load_cases=None, weights=None,
        pad_thickness=0.035, keep_extra=None, resolution=None,
        smooth_iters=40, progress=None):
    """
    Returns (mesh, info). `stl` is a path/file or an already-loaded trimesh.
    volfrac / member override the style; supports / load_cases override `load`.
    pad_thickness: solid pad under loads and supports, fraction of longest side.
    """
    t0 = time.time()
    mesh_in = stl if hasattr(stl, "faces") else voxel.load_mesh(stl)
    st, dt = STYLES[style], DETAIL[detail]
    volfrac = volfrac or st["volfrac"]
    member = member or st["member"]
    if supports is None:
        lp = LOADS[load]
        supports, load_cases, weights = lp["supports"], lp["load_cases"], lp["weights"]
    res = resolution or choose_resolution(mesh_in, up_axis, dt["elements"])
    domain, grid = voxel.voxelize(mesh_in, res, up_axis)
    n = max(domain.shape)
    rmin = member * n / 2.0
    # A member cannot be thicker than the part's own walls. On a thin-walled
    # part (a channel, a bracket made of plates) the style's member size is
    # capped to the wall, and the result is a wall with organic windows.
    from scipy import ndimage as ndi
    half_wall = float(ndi.distance_transform_edt(np.pad(domain, 1)).max())
    thin_walled = rmin > 0.8 * half_wall
    rmin = max(min(rmin, 0.8 * half_wall), 1.5)
    keep = build_keep(domain, supports, load_cases,
                      max(round(pad_thickness * n), 1) if pad_thickness else 0)
    if keep_extra is not None:
        keep |= keep_extra(domain) & domain
    kf = keep.sum() / max(domain.sum(), 1)
    if thin_walled:
        # windows in a plate need more of the plate left than struts in a block
        volfrac = max(volfrac, 0.30 if volfrac < 0.2 else 0.40)
    vf = max(volfrac, kf + 0.06)

    def cb(h):
        if progress:
            progress(h)

    result = topopt.optimize(domain, volfrac=vf, rmin=rmin, supports=supports,
                             load_cases=load_cases, case_weights=weights,
                             keep=keep, quality=dt["quality"], callback=cb)
    t_opt = time.time() - t0
    out, minfo = surface.extract(result, grid, mesh_in, smooth_iters=smooth_iters)
    solid = float(abs(mesh_in.volume)) if mesh_in.is_watertight else None
    info = dict(
        style=style, load=load, detail=detail, resolution=res,
        grid=list(domain.shape), voxel_mm=grid["pitch"], elements=result["elements"],
        rmin_voxels=rmin, member_mm=2 * rmin * grid["pitch"], thin_walled=bool(thin_walled),
        volfrac=vf, keep_fraction=float(kf), iterations=result["iterations"],
        multires=result["multires"], compliance=result["compliance"],
        grayness=result["hist"][-1]["grayness"],
        seconds_optimize=t_opt, seconds_mesh=time.time() - t0 - t_opt,
        seconds=time.time() - t0, mesh=minfo,
        mass_kept=(minfo["volume_mm3"] / solid) if solid else None,
        history=result["hist"])
    return out, info, result, grid
