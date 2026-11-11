"""
3D SIMP topology optimization, built to reach a finished design fast.

What is different from the first studio's `simp_fast.py`:

  * coarse-to-fine: the layout is found on a half-resolution grid (8x fewer
    elements), then refined at full resolution for a handful of iterations;
  * the stiffness matrix is assembled as 3x3 node blocks (BSR) with 64 adds
    instead of a 576-entries-per-element triplet sort -- ~30x less memory;
  * CG is warm-started from the previous iteration's displacements and solved
    to an engineering tolerance, with the AMG preconditioner reused until it
    stops paying for itself;
  * the density filter is an FFT convolution, so a wide filter costs the same
    as a narrow one;
  * the run stops by itself when the design has stopped changing, and returns
    the smooth (pre-projection) density that the surface extractor needs.

Array convention: (nx, ny, nz), +Z up.
"""
from __future__ import annotations

import time

import numpy as np
import pyamg
from scipy import fft as sfft
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import cg

from .voxel import coarsen

_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
_AX = {"x": 0, "y": 1, "z": 2}


def hex8_KE(nu=0.3):
    """24x24 stiffness of a unit cube, E=1, 2x2x2 Gauss."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] = lam + 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    xi = _CORNERS * 2.0 - 1.0
    g = 1 / np.sqrt(3.0)
    KE = np.zeros((24, 24))
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                dN = 0.125 * np.stack([
                    xi[:, 0] * (1 + xi[:, 1] * b) * (1 + xi[:, 2] * c),
                    (1 + xi[:, 0] * a) * xi[:, 1] * (1 + xi[:, 2] * c),
                    (1 + xi[:, 0] * a) * (1 + xi[:, 1] * b) * xi[:, 2]], 1) * 2.0
                B = np.zeros((6, 24))
                B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[:, 0], dN[:, 1], dN[:, 2]
                B[3, 0::3], B[3, 1::3] = dN[:, 1], dN[:, 0]
                B[4, 1::3], B[4, 2::3] = dN[:, 2], dN[:, 1]
                B[5, 0::3], B[5, 2::3] = dN[:, 2], dN[:, 0]
                KE += B.T @ D @ B * 0.125
    return KE


# ------------------------------------------------------------ regions on faces
def region_mask(shape, face, pattern="all", frac=0.3):
    """
    Boolean mask over a (n0,n1,n2) grid selecting part of one bounding-box face.
    pattern: all | center | ring | corners | strip_u | strip_v | edge_u0 |
             edge_u1 | edge_v0 | edge_v1      (u,v = in-plane axes, x<y<z order)
    """
    ax = _AX[face[0]]
    inpl = [a for a in range(3) if a != ax]
    nu_, nv_ = shape[inpl[0]], shape[inpl[1]]
    fu = (np.arange(nu_) / max(nu_ - 1, 1))[:, None]
    fv = (np.arange(nv_) / max(nv_ - 1, 1))[None, :]
    one = np.ones((nu_, nv_), dtype=bool)
    if pattern == "all":
        sel = one
    elif pattern == "center":
        sel = (np.abs(fu - .5) <= frac / 2) & (np.abs(fv - .5) <= frac / 2)
    elif pattern == "ring":
        sel = (fu <= frac) | (fu >= 1 - frac) | (fv <= frac) | (fv >= 1 - frac)
    elif pattern == "corners":
        sel = ((fu <= frac) | (fu >= 1 - frac)) & ((fv <= frac) | (fv >= 1 - frac))
    elif pattern == "strip_u":
        sel = (np.abs(fu - .5) <= frac / 2) & one
    elif pattern == "strip_v":
        sel = (np.abs(fv - .5) <= frac / 2) & one
    elif pattern == "edge_u0":
        sel = (fu <= frac) & one
    elif pattern == "edge_u1":
        sel = (fu >= 1 - frac) & one
    elif pattern == "edge_v0":
        sel = (fv <= frac) & one
    elif pattern == "edge_v1":
        sel = (fv >= 1 - frac) & one
    else:
        raise ValueError("unknown pattern %r" % pattern)
    return ax, sel


def face_elements(domain, spec, depth=1):
    """
    Elements of the part that a face spec lands on: looking along the face's
    axis, the first `depth` solid voxels in every column of the selected
    region -- but only where the part comes within `reach` (a fraction of the
    part's size along that axis, default 0.12) of the bounding-box face. So
    "top" follows a domed or stepped top surface, yet does not also grab a
    ledge halfway down the part.
    """
    ax, sel = region_mask(domain.shape, spec["face"], spec.get("pattern", "all"),
                          spec.get("frac", 0.3))
    d = np.moveaxis(domain, ax, 2)
    if spec["face"][1] == "+":
        d = d[:, :, ::-1]
    first = np.cumsum(d, axis=2)
    n = d.shape[2]
    band = max(int(np.ceil(spec.get("reach", 0.12) * n)), 1)
    near = np.arange(n)[None, None, :] < band + depth - 1
    pick = d & (first <= depth) & sel[:, :, None] & near
    if spec["face"][1] == "+":
        pick = pick[:, :, ::-1]
    return np.moveaxis(pick, 2, ax)


def _face_nodes(domain, spec):
    """Grid-node mask (nx+1,ny+1,nz+1) of the outer face of face_elements."""
    el = face_elements(domain, spec, 1)
    ax = _AX[spec["face"][0]]
    nodes = np.zeros(tuple(n + 1 for n in domain.shape), dtype=bool)
    hi = spec["face"][1] == "+"
    for du in (0, 1):
        for dv in (0, 1):
            off = [0, 0, 0]
            inpl = [a for a in range(3) if a != ax]
            off[inpl[0]], off[inpl[1]], off[ax] = du, dv, int(hi)
            sl = tuple(slice(o, o + n) for o, n in zip(off, domain.shape))
            nodes[sl] |= el
    return nodes


# ------------------------------------------------------------------- filter
class ConeFilter:
    """Linear cone density filter restricted to the domain, by FFT."""

    def __init__(self, domain, rmin):
        self.dom = domain
        r = max(int(np.ceil(rmin)) - 1, 0)
        self.fs = [sfft.next_fast_len(n + r, real=True) for n in domain.shape]
        k = np.zeros(self.fs)
        rng = np.arange(-r, r + 1)
        dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
        w = np.maximum(rmin - np.sqrt(dx * dx + dy * dy + dz * dz), 0.0)
        k[np.ix_(rng % self.fs[0], rng % self.fs[1], rng % self.fs[2])] = w
        self.Kh = sfft.rfftn(k)
        self.Hs = np.maximum(self._conv(domain.astype(float))[domain], 1e-9)

    def _conv(self, g):
        nx, ny, nz = self.dom.shape
        out = sfft.irfftn(sfft.rfftn(g, s=self.fs, workers=-1) * self.Kh,
                          s=self.fs, workers=-1)
        return out[:nx, :ny, :nz]

    def fwd(self, x):
        g = np.zeros(self.dom.shape)
        g[self.dom] = x
        return np.clip(self._conv(g)[self.dom] / self.Hs, 0.0, 1.0)

    def adj(self, y):
        g = np.zeros(self.dom.shape)
        g[self.dom] = y / self.Hs
        return self._conv(g)[self.dom]


# --------------------------------------------------------------- FE problem
class _FE:
    def __init__(self, domain, supports, load_cases, nu=0.3):
        self.domain = domain
        nx, ny, nz = domain.shape
        nx1, ny1 = nx + 1, ny + 1
        ex, ey, ez = np.nonzero(domain)
        self.na = na = len(ex)
        gnode = np.empty((na, 8), dtype=np.int64)
        for c in range(8):
            dx, dy, dz = _CORNERS[c]
            gnode[:, c] = (ex + dx) + (ey + dy) * nx1 + (ez + dz) * nx1 * ny1
        used, inv = np.unique(gnode.ravel(), return_inverse=True)
        self.nn = nn = len(used)
        self.enode = enode = inv.reshape(na, 8).astype(np.int32)
        self.edof = (3 * enode[:, :, None] + np.arange(3)[None, None, :]
                     ).reshape(na, 24)
        self.coords = np.stack([used % nx1, (used // nx1) % ny1,
                                used // (nx1 * ny1)], 1).astype(float)

        def nodes_of(spec):
            m = _face_nodes(domain, spec).reshape(-1, order="F")
            return np.flatnonzero(m[used])

        fixed = np.zeros(3 * nn, dtype=bool)
        for s in supports:
            n = nodes_of(s)
            if len(n) == 0:
                raise ValueError("a support lands on no material: %r" % (s,))
            for ch in s.get("dofs", "xyz"):
                fixed[3 * n + _AX[ch]] = True
        F = np.zeros((3 * nn, len(load_cases)))
        for ci, lc in enumerate(load_cases):
            for ld in lc:
                n = nodes_of(ld)
                d = ld.get("dir", "-z")
                F[3 * n + _AX[d[-1]], ci] += (-1.0 if d[0] == "-" else 1.0) * ld.get("mag", 1.0)
            F[fixed, ci] = 0.0
            tot = np.abs(F[:, ci]).sum()
            if tot == 0:
                raise ValueError("load case %d lands on no free material" % (ci + 1))
            F[:, ci] /= tot
        self.fixed, self.F = fixed, F

        # ---- BSR sparsity: one 3x3 block per connected node pair
        key = (enode[:, :, None].astype(np.int64) * nn + enode[:, None, :]).reshape(-1)
        ukey, slot = np.unique(key, return_inverse=True)
        self.slot = slot.reshape(na, 64).astype(np.int32)
        rows = (ukey // nn).astype(np.int32)
        self.indices = (ukey % nn).astype(np.int32)
        self.indptr = np.zeros(nn + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=nn), out=self.indptr[1:])
        self.nblk = len(ukey)
        fr = (~fixed).reshape(nn, 3)
        self.bmask = (fr[rows][:, :, None] & fr[self.indices][:, None, :])
        self.diag_blk = np.flatnonzero(rows == self.indices)
        self.fix_diag = fixed.reshape(nn, 3).astype(float)
        self.KE = hex8_KE(nu)
        self.KEb = self.KE.reshape(8, 3, 8, 3).transpose(0, 2, 1, 3).reshape(64, 3, 3)

        # rigid-body modes: the near-nullspace AMG needs for elasticity
        c = self.coords - self.coords.mean(0)
        c /= max(np.abs(c).max(), 1.0)
        B = np.zeros((3 * nn, 6))
        for k in range(3):
            B[k::3, k] = 1.0
        B[0::3, 3], B[1::3, 3] = -c[:, 1], c[:, 0]
        B[1::3, 4], B[2::3, 4] = -c[:, 2], c[:, 1]
        B[0::3, 5], B[2::3, 5] = c[:, 2], -c[:, 0]
        B[fixed] = 0.0
        self.B = B
        self.U = np.zeros_like(F)
        self.ml = None
        self.base_its = None
        self.builds = 0

    def assemble(self, E):
        data = np.zeros((self.nblk, 3, 3))
        for p in range(64):
            data[self.slot[:, p]] += E[:, None, None] * self.KEb[p]
        data *= self.bmask
        d = data[self.diag_blk]
        d[:, [0, 1, 2], [0, 1, 2]] += self.fix_diag
        data[self.diag_blk] = d
        n = 3 * self.nn
        return bsr_matrix((data, self.indices, self.indptr), shape=(n, n))

    def solve(self, K, rtol=1e-2):
        """PCG for every load case, warm-started; returns max CG iterations."""
        worst = 0
        for attempt in range(2):
            if self.ml is None:
                self.ml = pyamg.smoothed_aggregation_solver(
                    K, B=self.B, max_coarse=300, symmetry="symmetric",
                    strength=("symmetric", {"theta": 0.05}),
                    smooth=("energy", {"krylov": "cg", "maxiter": 3, "degree": 1}),
                    presmoother=("gauss_seidel", {"sweep": "symmetric"}),
                    postsmoother=("gauss_seidel", {"sweep": "symmetric"}))
                self.M = self.ml.aspreconditioner(cycle="V")
                self.builds += 1
                self.base_its = None
            worst, ok = 0, True
            for ci in range(self.F.shape[1]):
                its = [0]

                def cnt(_):
                    its[0] += 1
                u, info = cg(K, self.F[:, ci], x0=self.U[:, ci], rtol=rtol,
                             maxiter=400, M=self.M, callback=cnt)
                if info != 0 or not np.isfinite(u).all():
                    ok = False
                    break
                self.U[:, ci] = u
                worst = max(worst, its[0])
            if ok:
                if self.base_its is None:
                    self.base_its = max(worst, 4)
                elif worst > 2.2 * self.base_its:
                    self.ml = None        # preconditioner has gone stale
                return worst
            self.ml = None
            self.U[:] = 0.0
        raise RuntimeError(
            "the linear solve failed -- the part is probably not held firmly "
            "enough by its supports (it can slide or spin).")

    def energies(self):
        """Per-element strain energy u^T KE u for each load case: (na, ncase)."""
        out = np.empty((self.na, self.F.shape[1]))
        for ci in range(self.F.shape[1]):
            Ue = self.U[:, ci][self.edof]
            out[:, ci] = np.einsum("ij,ij->i", Ue @ self.KE, Ue)
        return out


def _project(xt, b):
    t = np.tanh(b * 0.5)
    return (t + np.tanh(b * (xt - 0.5))) / (2 * t)


def _dproject(xt, b):
    return b * (1 - np.tanh(b * (xt - 0.5)) ** 2) / (2 * np.tanh(b * 0.5))


def _stage(domain, keep, x, volfrac, rmin, supports, load_cases, weights,
           penal, betas, p_start, max_per_beta, min_per_beta, tol, cb, tag,
           hist, t0):
    """One resolution level. Returns (x, xTilde, xPhys)."""
    fe = _FE(domain, supports, load_cases)
    flt = ConeFilter(domain, rmin)
    keep_a = keep[domain]
    na = fe.na
    Emin = 1e-5
    w = np.asarray(weights, float)
    target = volfrac * na
    x = x.copy()
    x[keep_a] = 1.0
    p_cur = p_start
    xt = flt.fwd(x)
    dc_prev = None
    for bi, beta in enumerate(betas):
        # The textbook OC update (exponent 0.5) sits exactly on its stability
        # limit at p = 3; projection pushes it over and the design flip-flops
        # between two states forever. A softer exponent plus averaging the
        # sensitivities with the previous iteration removes that mode.
        eta = 0.5 if beta <= 1 else 0.3
        move = 0.2 if beta <= 4 else 0.1
        calm = 0
        for k in range(max_per_beta):
            xp = _project(xt, beta)
            xp[keep_a] = 1.0
            E = Emin + xp ** p_cur * (1 - Emin)
            K = fe.assemble(E)
            cgits = fe.solve(K)
            ce = fe.energies()
            comp = float((E[:, None] * ce * w).sum())
            dc = -p_cur * xp ** (p_cur - 1) * (1 - Emin) * (ce @ w)
            dpr = _dproject(xt, beta)
            dc = flt.adj(dc * dpr)
            if dc_prev is not None:
                dc = 0.5 * (dc + dc_prev)
            dc_prev = dc
            dv = np.maximum(flt.adj(dpr), 1e-12)
            ocb = (np.maximum(-dc, 0.0) / dv) ** eta
            hi = (float(ocb.max()) * 1e4) ** (1.0 / eta) + 1e-300
            lo = hi * 1e-30
            for _ in range(80):
                mid = np.sqrt(lo * hi)
                xn = np.clip(x * ocb / mid ** eta,
                             np.maximum(x - move, 0.0), np.minimum(x + move, 1.0))
                xn[keep_a] = 1.0
                xtn = flt.fwd(xn)
                xpn = _project(xtn, beta)
                xpn[keep_a] = 1.0
                if xpn.sum() > target:
                    lo = mid
                else:
                    hi = mid
                if hi / lo < 1.0 + 1e-5:
                    break
            change = float(np.abs(xn - x).mean())
            x, xt = xn, xtn
            gray = float(np.mean(4 * xpn * (1 - xpn)))
            h = dict(it=len(hist) + 1, level=tag, compliance=comp, change=change,
                     beta=beta, penal=p_cur, cg=cgits, grayness=gray,
                     seconds=time.time() - t0, elements=na)
            hist.append(h)
            if cb:
                cb(h)
            if p_cur < penal:
                p_cur = min(penal, p_cur + 0.25)
                continue
            calm = calm + 1 if change < tol else 0
            if k + 1 >= min_per_beta and calm >= 1:
                break
    xp = _project(xt, betas[-1])
    xp[keep_a] = 1.0
    xt = xt.copy()
    xt[keep_a] = 1.0
    return x, xt, xp, fe


def optimize(domain, volfrac=0.25, rmin=3.0, supports=None, load_cases=None,
             case_weights=None, keep=None, penal=3.0, quality="final",
             multires=True, callback=None, x_init=None):
    """
    domain : bool (nx,ny,nz) -- where material may exist.
    rmin   : filter radius in voxels of THIS grid (sets member thickness).
    keep   : bool grid of voxels forced solid.
    quality: "preview" (short schedule) or "final" (run to convergence).
    Returns dict(rho, rho_smooth, hist, seconds, ...); rho_smooth is the
    filtered field whose 0.5 level set is the design surface.
    """
    t0 = time.time()
    domain = np.asarray(domain, dtype=bool)
    keep = np.zeros_like(domain) if keep is None else (np.asarray(keep, bool) & domain)
    supports = supports or [{"face": "z-"}]
    load_cases = load_cases or [[{"face": "z+", "dir": "-z"}]]
    weights = case_weights or [1.0] * len(load_cases)
    na = int(domain.sum())
    kf = keep.sum() / max(na, 1)
    if kf >= volfrac - 0.02:
        raise ValueError(
            "the forced-solid regions are already %.0f%% of the part but the "
            "material budget is %.0f%% -- raise the budget or shrink them."
            % (100 * kf, 100 * volfrac))
    hist = []
    final = quality == "final"
    tol = 0.002 if final else 0.004     # mean |change| of the design per iteration
    use_mr = multires and final and min(domain.shape) >= 12 and na > 20000

    if use_mr:
        dc_ = coarsen(domain)
        kc = coarsen(keep) & dc_
        vf_c = min(max(volfrac * na / (8.0 * dc_.sum()), kc.sum() / dc_.sum() + 0.03), 0.95)
        xc = np.full(int(dc_.sum()), vf_c)
        xc, _, _, _ = _stage(dc_, kc, xc, vf_c, max(rmin / 2.0, 1.2), supports, load_cases,
                             weights, penal, [1.0, 2.0, 4.0], 1.0,
                             18 if final else 12, 4, tol, callback, "coarse",
                             hist, t0)
        g = np.zeros(dc_.shape)
        g[dc_] = xc
        nx, ny, nz = domain.shape
        up = np.repeat(np.repeat(np.repeat(g, 2, 0), 2, 1), 2, 2)[:nx, :ny, :nz]
        x = up[domain]
        betas = [4.0, 8.0, 16.0] if final else [4.0, 8.0]
        x, xt, xp, fe = _stage(domain, keep, x, volfrac, rmin, supports, load_cases,
                               weights, penal, betas, penal,
                               6 if final else 3, 3 if final else 2, 2 * tol,
                               callback, "fine", hist, t0)
    else:
        x = np.full(na, volfrac) if x_init is None else x_init[domain]
        betas = [1.0, 2.0, 4.0, 8.0, 16.0] if final else [1.0, 2.0, 4.0, 8.0]
        x, xt, xp, fe = _stage(domain, keep, x, volfrac, rmin, supports, load_cases,
                               weights, penal, betas, 1.0,
                               18 if final else 12, 4 if final else 3, tol,
                               callback, "single", hist, t0)

    rho = np.zeros(domain.shape)
    rho[domain] = xp
    smooth = np.zeros(domain.shape)
    smooth[domain] = xt
    return dict(rho=rho, rho_smooth=smooth, domain=domain, keep=keep, hist=hist,
                volfrac=float(volfrac),
                elements=na, dofs=3 * fe.nn, seconds=time.time() - t0,
                compliance=hist[-1]["compliance"], iterations=len(hist),
                multires=use_mr, amg_builds=fe.builds)
