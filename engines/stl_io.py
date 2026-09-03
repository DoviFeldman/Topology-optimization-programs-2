"""
Threshold a 3D density field into a printable surface mesh and write STL.

The optimizer returns a density field in [0, 1]. Following the standard 3D-print
workflow (see notes.txt), we keep voxels above a threshold (e.g. 0.5) and emit
the exposed cube faces as triangles — a watertight "blocky" surface you can slice
and print, or import into FreeCAD/CalculiX for validation. No external mesh
library is needed; STL is written directly.

Axis convention: density has shape (nely, nelx, nelz) -> array axes (y, x, z),
mapped to coordinates (x, y, z) scaled by `spacing` (mm per voxel).
"""

import struct
import numpy as np


# For each of the 6 face directions: (neighbor delta (dy,dx,dz), 4 corner
# offsets (dx,dy,dz) in CCW order as seen from outside so the normal points out).
_FACES = {
    "+x": ((0, 1, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    "-x": ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    "+y": ((1, 0, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    "-y": ((-1, 0, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    "+z": ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    "-z": ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
}


def voxel_mesh(density, threshold=0.5, spacing=1.0):
    """
    Build the surface triangle mesh of the thresholded voxels.

    Returns triangles as an array of shape (n_tri, 3, 3) (each triangle = 3 xyz
    vertices). Only faces on the boundary of the solid region are emitted.
    """
    solid = np.asarray(density) >= threshold  # (ny, nx, nz)
    ny, nx, nz = solid.shape
    tris = []

    for (ddy, ddx, ddz), corners in _FACES.values():
        # A face is exposed where the voxel is solid and its neighbor is not
        # solid (or is outside the grid).
        shifted = np.zeros_like(solid)
        ys = slice(max(ddy, 0), ny + min(ddy, 0))
        xs = slice(max(ddx, 0), nx + min(ddx, 0))
        zs = slice(max(ddz, 0), nz + min(ddz, 0))
        ys2 = slice(max(-ddy, 0), ny + min(-ddy, 0))
        xs2 = slice(max(-ddx, 0), nx + min(-ddx, 0))
        zs2 = slice(max(-ddz, 0), nz + min(-ddz, 0))
        neigh = np.zeros_like(solid)
        neigh[ys2, xs2, zs2] = solid[ys, xs, zs]
        exposed = solid & ~neigh
        jj, ii, kk = np.nonzero(exposed)  # y, x, z indices of exposed voxels

        # corners are (dx, dy, dz) offsets; voxel origin at (x=ii, y=jj, z=kk)
        c = np.array(corners, dtype=float)  # (4, 3)
        base = np.stack([ii, jj, kk], axis=1).astype(float)  # (m, 3) as (x,y,z)
        quad = base[:, None, :] + c[None, :, :]              # (m, 4, 3)
        quad *= spacing
        # two triangles per quad: (0,1,2) and (0,2,3)
        t1 = quad[:, [0, 1, 2], :]
        t2 = quad[:, [0, 2, 3], :]
        tris.append(t1)
        tris.append(t2)

    if not tris:
        return np.zeros((0, 3, 3))
    return np.concatenate(tris, axis=0)


def _normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def write_stl(tris, path, solid_name="topopt"):
    """Write triangles (n,3,3) to a binary STL file at `path`."""
    tris = np.asarray(tris, dtype=np.float32)
    nrm = _normals(tris).astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)                      # 80-byte header
        f.write(struct.pack("<I", len(tris)))    # triangle count
        for i in range(len(tris)):
            f.write(struct.pack("<3f", *nrm[i]))
            for v in tris[i]:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))        # attribute byte count
    return path


def voxels_to_stl(density, path, threshold=0.5, spacing=1.0):
    """Threshold a density field and write it straight to an STL file."""
    tris = voxel_mesh(density, threshold=threshold, spacing=spacing)
    write_stl(tris, path)
    return tris.shape[0]
