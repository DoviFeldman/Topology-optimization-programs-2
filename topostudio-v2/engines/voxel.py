"""
STL -> voxel design domain.

A voxel is "inside" when its CENTRE is inside the mesh (ray parity along Z), so
a 33 mm wide part at 0.4125 mm pitch is exactly 80 voxels wide -- no half-voxel
growth -- and the grid keeps the transform back to the STL's own coordinates.
Cost is proportional to the number of triangles plus the number of voxels, so a
1M-triangle scan voxelizes in seconds.
"""
from __future__ import annotations

import numpy as np
import trimesh

_PERM = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}


def load_mesh(path_or_file):
    mesh = trimesh.load(path_or_file, file_type="stl", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.merge_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _parity_fill(tri, shape, pitch):
    """tri: (n,3,3) triangle corners in grid frame (origin at grid corner)."""
    nx, ny, nz = shape
    # tiny irrational nudge so no ray runs exactly along a shared triangle edge
    ox, oy = 0.5 + 3.7e-5, 0.5 + 7.1e-5
    p = tri / pitch
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    i0 = np.clip(np.ceil(x.min(1) - ox).astype(np.int64), 0, nx)
    i1 = np.clip(np.floor(x.max(1) - ox).astype(np.int64) + 1, 0, nx)
    j0 = np.clip(np.ceil(y.min(1) - oy).astype(np.int64), 0, ny)
    j1 = np.clip(np.floor(y.max(1) - oy).astype(np.int64) + 1, 0, ny)
    w, h = np.maximum(i1 - i0, 0), np.maximum(j1 - j0, 0)
    cnt = w * h
    counts = np.zeros((nx, ny, nz + 1), dtype=np.int32)
    order = np.flatnonzero(cnt > 0)
    CH = 4_000_000                                 # candidate samples per chunk
    start = 0
    csum = np.cumsum(cnt[order])
    while start < len(order):
        base = csum[start - 1] if start else 0
        stop = int(np.searchsorted(csum, base + CH, side="right"))
        stop = max(stop, start + 1)
        t = order[start:stop]
        c = cnt[t]
        tid = np.repeat(t, c)
        k = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        ii = i0[tid] + k % w[tid]
        jj = j0[tid] + k // w[tid]
        px, py = ii + ox, jj + oy
        ax, ay, bx, by, cx, cy = (x[tid, 0], y[tid, 0], x[tid, 1], y[tid, 1],
                                  x[tid, 2], y[tid, 2])
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(den) > 1e-14
        den = np.where(ok, den, 1.0)
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        l2 = 1.0 - l0 - l1
        hit = ok & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        zz = l0 * z[tid, 0] + l1 * z[tid, 1] + l2 * z[tid, 2]
        kz = np.clip(np.ceil(zz[hit] - 0.5).astype(np.int64), 0, nz)
        np.add.at(counts, (ii[hit], jj[hit], kz), 1)
        start = stop
    return (np.cumsum(counts, axis=2)[:, :, :nz] % 2) == 1


def voxelize(mesh, resolution=64, up_axis="z", pitch=None, origin=None, shape=None):
    """
    Returns (domain bool (nx,ny,nz), grid) where grid = dict(pitch, origin, perm)
    and a voxel (i,j,k) has its centre at origin + (i+.5, j+.5, k+.5) * pitch in
    the up-axis-permuted frame; `to_world` maps points back to STL coordinates.
    """
    perm = _PERM[up_axis]
    v = np.asarray(mesh.vertices, dtype=np.float64)[:, list(perm)]
    lo, hi = v.min(0), v.max(0)
    if pitch is None:
        pitch = float((hi - lo).max()) / max(int(resolution), 4)
    if origin is None:
        n = np.maximum(np.ceil((hi - lo) / pitch - 1e-6).astype(int), 1)
        origin = (lo + hi) / 2.0 - n * pitch / 2.0   # centre the part in the grid
        shape = tuple(int(a) for a in n)
    tri = v[np.asarray(mesh.faces)] - np.asarray(origin)
    occ = _parity_fill(tri, shape, pitch)
    grid = dict(pitch=float(pitch), origin=[float(a) for a in origin],
                perm=perm, shape=tuple(occ.shape),
                extents=[float(a) for a in (hi - lo)])
    return occ, grid


def to_world(points, grid):
    """Permuted-frame points -> original STL coordinates."""
    out = np.empty_like(points)
    out[:, list(grid["perm"])] = points
    return out


def coarsen(mask, mode="any"):
    """Halve a boolean grid (2x2x2 blocks)."""
    nx, ny, nz = mask.shape
    m = np.pad(mask, ((0, nx % 2), (0, ny % 2), (0, nz % 2)))
    b = m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2, m.shape[2] // 2, 2)
    s = b.sum(axis=(1, 3, 5))
    return s > 0 if mode == "any" else s >= 4
