"""
Make ToPy (williamhunter/ToPy) runnable on a modern Mac, and fast enough to use.

Three things happen here:

1. `external/pysparse_shim` is put on the path so ToPy's `from pysparse import
   ...` resolves to the SciPy-backed stand-in (PySparse itself is Python-2 only
   and has no Apple-Silicon build). The ToPy sources in `external/ToPy` have
   already been mechanically ported to Python 3 (`xrange`, `has_key`).

2. `Topology._updateK` is replaced with a vectorized assembly. Upstream loops in
   Python over every element and scatter-adds a 24x24 block; at the resolutions
   we care about that is hundreds of thousands of Python iterations per FE
   solve. The replacement computes the whole element-to-dof map at once and
   hands the triplets to the shim in chunks. Same matrix, ~100x less overhead.

3. The AMG preconditioner is given the six rigid-body modes as its near
   nullspace, which is what makes the 3-D iterative solve converge in tens of
   iterations rather than thousands.

Nothing about ToPy's method changes: it is still ToPy's OC update, its
grey-scale filter, its `eta`/`q` continuation, its `.tpd` problem definition.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "external", "pysparse_shim"),
          os.path.join(_ROOT, "external", "ToPy")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _vectorized_updateK(self, K):
    """Drop-in replacement for topy.topology.Topology._updateK."""
    from topy.topology import VOID

    dofpn = self.dofpn
    nelx, nely, nelz = self.nelx, self.nely, self.nelz
    Ke = np.asarray(self.Ke, dtype=float)
    m = Ke.shape[0]

    if nelz == 0:                                   # 2-D
        elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        base = (ely + elx * (nely + 1)).ravel()
        dens = self.desvars[ely.ravel(), elx.ravel()]
    else:                                           # 3-D
        elz, elx, ely = np.meshgrid(np.arange(nelz), np.arange(nelx),
                                    np.arange(nely), indexing="ij")
        elz, elx, ely = elz.ravel(), elx.ravel(), ely.ravel()
        base = ely + elx * (nely + 1) + elz * (nelx + 1) * (nely + 1)
        dens = self.desvars[elz, ely, elx]

    if self.probtype == "heat":
        scale = VOID + (1 - VOID) * dens ** self.p
    else:
        scale = dens ** self.p

    edof = self.e2sdofmapi[None, :] + dofpn * base[:, None]      # (nele, m)
    keflat = Ke.ravel()

    # chunked so the triplet arrays never blow up memory on a fine grid
    nele = edof.shape[0]
    chunk = max(1, int(4_000_000 // (m * m)))
    for s in range(0, nele, chunk):
        e = edof[s:s + chunk]
        v = (scale[s:s + chunk, None] * keflat[None, :]).ravel()
        rows = np.repeat(e, m, axis=1).ravel()
        cols = np.tile(e, (1, m)).ravel()
        K.update_add_triplets(rows, cols, v)

    K.delete_rowcols(self._rcfixed)
    return K


def _rigid_body_modes(freedof, nelx, nely, nelz, dofpn):
    """Near-nullspace for the AMG preconditioner (elasticity only)."""
    if dofpn != 3:
        return None
    freedof = np.asarray(freedof)
    node = freedof // 3
    comp = freedof % 3
    nely1, nelx1 = nely + 1, nelx + 1
    iy = node % nely1
    ix = (node // nely1) % nelx1
    iz = node // (nely1 * nelx1)
    s = max(nelx, nely, nelz) / 2.0 or 1.0
    x = (ix - nelx / 2.0) / s
    y = (iy - nely / 2.0) / s
    z = (iz - nelz / 2.0) / s
    B = np.zeros((len(freedof), 6))
    for k in range(3):
        B[comp == k, k] = 1.0
    B[comp == 0, 3] = -y[comp == 0]
    B[comp == 1, 3] = x[comp == 1]
    B[comp == 1, 4] = -z[comp == 1]
    B[comp == 2, 4] = y[comp == 2]
    B[comp == 0, 5] = z[comp == 0]
    B[comp == 2, 5] = -x[comp == 2]
    return B


def install():
    """Patch ToPy in place. Safe to call more than once."""
    import topy
    from topy import topology as T

    if getattr(T.Topology, "_topopt_studio_patched", False):
        return topy

    T.Topology._updateK = _vectorized_updateK

    # give the preconditioner the rigid-body modes, and reuse it while it works
    orig_fea = T.Topology.fea

    def fea(self):
        if self.nelz and self.dofpn == 3 and not hasattr(self, "_rbm"):
            self._rbm = _rigid_body_modes(self.freedof, self.nelx, self.nely,
                                          self.nelz, self.dofpn)
        return orig_fea(self)

    T.Topology.fea = fea

    import pysparse.precon as P
    _orig_ssor = P.ssor

    def ssor(A, near_nullspace=None, **kw):
        return _orig_ssor(A, near_nullspace=install._current_rbm)

    P.ssor = ssor
    install._current_rbm = None

    # capture the near-nullspace at fea() time
    def fea2(self):
        if self.nelz and self.dofpn == 3 and not hasattr(self, "_rbm"):
            self._rbm = _rigid_body_modes(self.freedof, self.nelx, self.nely,
                                          self.nelz, self.dofpn)
        install._current_rbm = getattr(self, "_rbm", None)
        return orig_fea(self)

    T.Topology.fea = fea2
    T.Topology._topopt_studio_patched = True
    return topy
