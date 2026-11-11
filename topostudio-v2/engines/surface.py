"""
Density field -> printable STL.

This is where the look comes from. The first studio marching-cubed the raw
0/1 voxel field at solver resolution, with empty space outside the part
counted as "no material": the part's own flat faces got melted, every strut
showed voxel steps, and thin grey members turned into spikes.

Here instead:
  1. the SMOOTH (filtered, pre-projection) density is used -- its 0.5 level
     set is the design the solver analysed, but it varies gently, so the
     iso-surface lands between voxels instead of on their steps;
  2. that field is continued past the part's boundary, upsampled 3-4x with
     cubic interpolation and lightly blurred, then marching-cubed;
  3. the organic surface is Taubin-smoothed (no shrinkage);
  4. the result is boolean-INTERSECTED with the original STL, so every outer
     face, hole and edge of the input part stays exactly as modelled and only
     the carved-out interior is organic.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

from . import voxel


def _decimate(mesh, target_faces):
    if target_faces and len(mesh.faces) > target_faces:
        try:
            import fast_simplification
            v, f = fast_simplification.simplify(
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32),
                target_count=int(target_faces), agg=4)
            out = trimesh.Trimesh(v.astype(np.float64), f, process=False)
            if out.is_watertight:          # never trade a closed mesh for a small one
                return out
        except Exception:
            pass
    return mesh


def _largest(mesh, min_frac=0.03):
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 1, 1
    vol = np.array([abs(p.volume) for p in parts])
    keep = [p for p, v in zip(parts, vol) if v >= min_frac * vol.max()]
    return trimesh.util.concatenate(keep), len(parts), len(keep)


def extract(result, grid, mesh_in=None, upsample=None, smooth_iters=40,
            level=0.5, blur=1.0, thin=1.2, target_faces=600_000, exact_outer=True):
    """
    result : dict from topopt.optimize (needs rho_smooth, domain, keep)
    grid   : dict from voxel.voxelize
    mesh_in: the original trimesh (STL coordinates); enables the exact cut.
    Returns (trimesh in the STL's own coordinates, info dict).
    """
    dom = result["domain"]
    f = np.where(dom, result["rho_smooth"], 0.0)
    f[result["keep"]] = 1.0
    pitch = grid["pitch"]
    k = upsample or int(np.clip(round(260.0 / max(dom.shape)), 2, 5))
    P = 3
    domp = np.pad(dom, P)
    fp = np.pad(f, P)
    # continue the density outward so its iso-surface crosses the part boundary
    # cleanly instead of closing on it
    idx = ndi.distance_transform_edt(~domp, return_distances=False,
                                     return_indices=True)
    fp = fp[tuple(idx)]
    up = np.clip(ndi.zoom(fp, k, order=3, mode="nearest", grid_mode=True), 0.0, 1.0)
    info = dict(upsample=k, fine_grid=list(up.shape))
    origin = np.asarray(grid["origin"]) - P * pitch

    # the part itself on the fine grid
    if mesh_in is not None:
        occ, _ = voxel.voxelize(mesh_in, pitch=pitch / k, origin=origin,
                                shape=up.shape, up_axis="xyz"[grid["perm"][2]])
    else:
        occ = np.repeat(np.repeat(np.repeat(domp, k, 0), k, 1), k, 2)

    # grey-scale opening INSIDE the part: erase webs, whiskers and skins lying
    # against a face that are thinner than about one solver voxel -- too thin
    # to carry load or to print -- without touching real members
    up *= occ
    w = max(int(round(thin * k)), 1)
    if w > 1:
        up = ndi.grey_dilation(ndi.grey_erosion(up, size=(w, w, w), mode="nearest"),
                               size=(w, w, w), mode="nearest") * occ
    use_exact = bool(exact_outer and mesh_in is not None and mesh_in.is_watertight)
    # continue the field straight out through the part's faces, so the organic
    # surface crosses them squarely and the exact cut is clean
    idx = tuple(ndi.distance_transform_edt(~occ, return_distances=False,
                                           return_indices=True))
    up = ndi.gaussian_filter(up[idx], blur * k, mode="nearest")

    # match the solver's material budget (a fixed 0.5 cut drops thin, slightly
    # grey members and leaves the part lighter and weaker than asked)
    target = result.get("volfrac")
    if target:
        vals = up[occ]
        if len(vals) > 5_000_000:
            vals = vals[::7]
        level = float(np.clip(np.quantile(vals, 1.0 - target), 0.25, level))
    info["level"] = level

    # Make the field constant along the normal in a thin band under every
    # face of the part (copy it out from just below the surface). The organic
    # surface then meets each face squarely; without this it can graze a face
    # at a shallow angle and the exact cut leaves hair-thin wedges there.
    d = max(1, int(round(0.4 * k)))
    core = ndi.binary_erosion(occ, iterations=d)
    if core.any():
        dist, cidx = ndi.distance_transform_edt(~core, return_indices=True)
        near = dist <= 2 * d + 2
        up = np.where(near, up[tuple(cidx)], up)
    if not use_exact:
        up = np.minimum(up, 2 * level * ndi.gaussian_filter(occ.astype(float), 0.8))

    up[0, :, :] = up[-1, :, :] = 0.0
    up[:, 0, :] = up[:, -1, :] = 0.0
    up[:, :, 0] = up[:, :, -1] = 0.0
    if up.max() <= level:
        raise ValueError("the optimized density is empty -- nothing to mesh")
    v, fc, _, _ = measure.marching_cubes(up, level=level)
    v = origin + (v + 0.5) * (pitch / k)
    mesh = trimesh.Trimesh(voxel.to_world(v, grid), fc, process=False)
    info["raw_faces"] = int(len(mesh.faces))
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53,
                                        iterations=int(smooth_iters))
    mesh = _decimate(mesh, target_faces)
    if mesh.volume < 0:
        mesh.invert()

    info["exact_outer"] = False
    if use_exact:
        try:
            cut = trimesh.boolean.intersection([mesh, mesh_in], engine="manifold",
                                              check_volume=False)
            if len(cut.faces) > 0:
                mesh = cut
                info["exact_outer"] = True
        except Exception as exc:              # pragma: no cover
            info["boolean_error"] = str(exc)[:200]
        if not info["exact_outer"]:
            return extract(result, grid, mesh_in, upsample, smooth_iters, level,
                           blur, thin, target_faces, exact_outer=False)

    mesh, n_parts, n_kept = _largest(mesh)
    mesh.fix_normals()
    info.update(pieces=n_parts, pieces_kept=n_kept, faces=int(len(mesh.faces)),
                watertight=bool(mesh.is_watertight),
                volume_mm3=float(abs(mesh.volume)),
                extents=[float(e) for e in mesh.extents])
    return mesh, info
