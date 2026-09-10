"""
STL in, STL out.

  stl_to_domain   : voxelize an STL into a boolean (nx, ny, nz) design domain.
  density_to_mesh : marching-cubes the density field, smooth it, keep the
                    largest connected body -> a printable, organic-looking mesh.

Marching cubes (rather than emitting cube faces) is what makes the result look
like a topology-optimized part instead of Minecraft.
"""

from __future__ import annotations

import numpy as np
import trimesh
from skimage import measure


def stl_to_domain(path_or_file, resolution=64, up_axis="z", pad=0):
    """
    Voxelize an STL. Returns (domain (nx,ny,nz) bool, meta).

    `resolution` = voxels along the model's longest axis.
    `up_axis` names which model axis should end up as the grid's +Z ("up").
    """
    mesh = trimesh.load(path_or_file, file_type="stl", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])

    perm = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}[up_axis]
    if perm != (0, 1, 2):
        mesh.vertices = mesh.vertices[:, list(perm)]
        mesh.apply_translation(-mesh.bounds[0])

    pitch = float(mesh.extents.max()) / max(int(resolution), 4)
    vox = mesh.voxelized(pitch=pitch)
    try:
        vox = vox.fill()
    except Exception:
        pass
    occ = np.asarray(vox.matrix, dtype=bool)   # (nx, ny, nz)
    if pad:
        occ = np.pad(occ, pad, constant_values=False)
    meta = dict(pitch=pitch, shape=occ.shape, filled=int(occ.sum()),
                extents=tuple(float(e) for e in mesh.extents),
                origin=tuple(float(o) for o in vox.transform[:3, 3]),
                volume_mm3=float(occ.sum()) * pitch ** 3,
                mesh_volume_mm3=float(mesh.volume) if mesh.is_watertight else None)
    return occ, meta


def density_to_mesh(rho, threshold=0.5, pitch=1.0, smooth_iters=12,
                    keep_largest=True, taubin=True, min_component_frac=0.02):
    """
    Density field -> smoothed trimesh. Returns (mesh, info).
    """
    rho = np.asarray(rho, dtype=float)
    # pad with zeros so the outer surface is always closed
    f = np.pad(rho, 1, constant_values=0.0)
    if f.max() < threshold:
        raise ValueError("nothing above threshold %.2f" % threshold)
    verts, faces, normals, _ = measure.marching_cubes(f, level=threshold)
    verts = (verts - 1.0) * pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    info = {"raw_faces": int(len(mesh.faces))}
    if keep_largest:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            vols = np.array([abs(p.volume) if p.is_watertight else
                             p.area ** 1.5 for p in parts])
            biggest = vols.max()
            keep = [p for p, v in zip(parts, vols) if v >= min_component_frac * biggest]
            info["components"] = len(parts)
            info["components_kept"] = len(keep)
            mesh = trimesh.util.concatenate(keep)
        else:
            info["components"] = 1
            info["components_kept"] = 1

    if smooth_iters > 0 and len(mesh.faces):
        try:
            if taubin:
                trimesh.smoothing.filter_taubin(mesh, lamb=0.55, nu=-0.54,
                                                iterations=int(smooth_iters))
            else:
                trimesh.smoothing.filter_laplacian(mesh, iterations=int(smooth_iters))
        except Exception as exc:  # pragma: no cover
            info["smooth_error"] = str(exc)

    mesh.fix_normals()
    info.update(faces=int(len(mesh.faces)), verts=int(len(mesh.vertices)),
                watertight=bool(mesh.is_watertight),
                volume_mm3=float(abs(mesh.volume)))
    return mesh, info


def save_stl(mesh, path):
    mesh.export(path)
    return path
