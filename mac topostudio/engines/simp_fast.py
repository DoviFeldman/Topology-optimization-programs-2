"""
Fast 3D SIMP topology optimization (engine #1, "3D SIMP" — rewritten).

Same method as the repo's original `engines/simp3d.py` (density-based SIMP,
optimality-criteria update, density filter), but rebuilt so it is usable at the
resolutions a Mac can actually handle:

  * numerically integrated H8 element (no magic constant table) so the node
    ordering is ours and provably consistent with the assembly;
  * only elements INSIDE the design domain are assembled — void space outside an
    uploaded STL costs nothing, and dofs that touch no active element are
    constrained away instead of being propped up by a 1e-9 stiffness;
  * the triplet -> CSR mapping is computed once and reused every iteration;
  * the linear solve is CG preconditioned by smoothed-aggregation AMG (pyamg)
    with the three rigid-body translations as the near-nullspace, instead of a
    direct `spsolve` (that is the difference between "minutes per iteration" and
    "seconds per iteration");
  * the density filter is built from a small stencil, vectorized over the grid,
    instead of a five-deep Python loop over every element pair;
  * optional Heaviside projection (with beta continuation) to drive the design
    to crisp 0/1 -- this is what produces clean struts rather than grey mush.

Array convention: everything is (nx, ny, nz), x/y/z in voxel units, +Z is up.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

try:
    import pyamg
    _HAVE_AMG = True
except Exception:  # pragma: no cover
    _HAVE_AMG = False


# --------------------------------------------------------------------------
# element
# --------------------------------------------------------------------------

# local node ordering of the trilinear hex, as (dx, dy, dz) grid offsets
_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int64)

_XI = _CORNERS * 2.0 - 1.0  # natural coords in [-1, 1]


def hex8_KE(E=1.0, nu=0.3, h=1.0):
    """24x24 stiffness matrix of a cube element of side `h`, by 2x2x2 Gauss."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    g = 1.0 / np.sqrt(3.0)
    KE = np.zeros((24, 24))
    J_det = (h / 2.0) ** 3
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                # shape function derivatives w.r.t. natural coords
                dN = np.empty((8, 3))
                for n in range(8):
                    x, y, z = _XI[n]
                    dN[n, 0] = 0.125 * x * (1 + y * b) * (1 + z * c)
                    dN[n, 1] = 0.125 * (1 + x * a) * y * (1 + z * c)
                    dN[n, 2] = 0.125 * (1 + x * a) * (1 + y * b) * z
                dNdx = dN * (2.0 / h)  # dxi/dx = 2/h for a cube of side h
                B = np.zeros((6, 24))
                for n in range(8):
                    bx, by, bz = dNdx[n]
                    B[0, 3 * n + 0] = bx
                    B[1, 3 * n + 1] = by
                    B[2, 3 * n + 2] = bz
                    B[3, 3 * n + 0] = by
                    B[3, 3 * n + 1] = bx
                    B[4, 3 * n + 1] = bz
                    B[4, 3 * n + 2] = by
                    B[5, 3 * n + 0] = bz
                    B[5, 3 * n + 2] = bx
                KE += B.T @ D @ B * J_det
    return KE


# --------------------------------------------------------------------------
# boundary conditions
# --------------------------------------------------------------------------

_AX = {"x": 0, "y": 1, "z": 2}


def _node_grid(nx, ny, nz):
    return nx + 1, ny + 1, nz + 1


def node_ids(ix, iy, iz, nx, ny, nz):
    nx1, ny1, _ = _node_grid(nx, ny, nz)
    return ix + iy * nx1 + iz * nx1 * ny1


def face_node_mask(face, nx, ny, nz):
    """Boolean (nx1, ny1, nz1) mask of the nodes lying on a named box face."""
    nx1, ny1, nz1 = _node_grid(nx, ny, nz)
    m = np.zeros((nx1, ny1, nz1), dtype=bool)
    ax, sgn = face[0], face[1]
    idx = {"x": nx1, "y": ny1, "z": nz1}[ax] - 1 if sgn == "+" else 0
    if ax == "x":
        m[idx, :, :] = True
    elif ax == "y":
        m[:, idx, :] = True
    else:
        m[:, :, idx] = True
    return m


def pattern_mask(shape, face, pattern="all", frac=0.3):
    """
    Restrict a face mask to a sub-pattern, in the two in-plane directions.

    pattern: all | center | ring | corners | strip_u | strip_v |
             edge_u0 | edge_u1 | edge_v0 | edge_v1 | half_u | half_v
    frac:    size of the feature as a fraction of the face (0..1)

    (u, v) are the two in-plane axes of the face, in x<y<z order: for the z
    faces u is x and v is y; for an x face u is y and v is z.
    """
    nx1, ny1, nz1 = shape
    ax = face[0]
    if ax == "x":
        u, v = np.meshgrid(np.arange(ny1), np.arange(nz1), indexing="ij")
        nu, nv = ny1, nz1
    elif ax == "y":
        u, v = np.meshgrid(np.arange(nx1), np.arange(nz1), indexing="ij")
        nu, nv = nx1, nz1
    else:
        u, v = np.meshgrid(np.arange(nx1), np.arange(ny1), indexing="ij")
        nu, nv = nx1, ny1

    fu = u / max(nu - 1, 1)
    fv = v / max(nv - 1, 1)
    if pattern == "all":
        sel = np.ones_like(fu, dtype=bool)
    elif pattern == "center":
        sel = (np.abs(fu - 0.5) <= frac / 2) & (np.abs(fv - 0.5) <= frac / 2)
    elif pattern == "ring":
        sel = (fu <= frac) | (fu >= 1 - frac) | (fv <= frac) | (fv >= 1 - frac)
    elif pattern == "corners":
        sel = ((fu <= frac) | (fu >= 1 - frac)) & ((fv <= frac) | (fv >= 1 - frac))
    elif pattern in ("strip_u", "strip_x"):
        sel = np.abs(fu - 0.5) <= frac / 2
    elif pattern in ("strip_v", "strip_y"):
        sel = np.abs(fv - 0.5) <= frac / 2
    elif pattern == "edge_u0":
        sel = fu <= frac
    elif pattern == "edge_u1":
        sel = fu >= 1 - frac
    elif pattern == "edge_v0":
        sel = fv <= frac
    elif pattern == "edge_v1":
        sel = fv >= 1 - frac
    elif pattern == "half_u":
        sel = fu <= 0.5
    elif pattern == "half_v":
        sel = fv <= 0.5
    else:
        raise ValueError("unknown pattern %r" % pattern)
    return sel


def build_bc(nx, ny, nz, supports, loads):
    """
    supports : list of dicts {face, pattern, frac, dofs:'xyz'}
    loads    : list of dicts {face, pattern, frac, dir:'-z', mag:1.0}
    Returns (fixed_dof_mask, F) over all dofs.
    """
    nx1, ny1, nz1 = _node_grid(nx, ny, nz)
    nnode = nx1 * ny1 * nz1
    ndof = 3 * nnode
    fixed = np.zeros(ndof, dtype=bool)
    F = np.zeros(ndof)

    for s in supports:
        fm = face_node_mask(s["face"], nx, ny, nz)
        pm = pattern_mask((nx1, ny1, nz1), s["face"], s.get("pattern", "all"),
                          s.get("frac", 0.3))
        sub = np.zeros_like(fm)
        ax = s["face"][0]
        idx = np.argmax(fm.any(axis=tuple(a for a in range(3) if a != _AX[ax])))
        # place the 2-D pattern back on the face plane
        if ax == "x":
            sub[np.flatnonzero(fm.any(axis=(1, 2)))[0], :, :] = pm
        elif ax == "y":
            sub[:, np.flatnonzero(fm.any(axis=(0, 2)))[0], :] = pm
        else:
            sub[:, :, np.flatnonzero(fm.any(axis=(0, 1)))[0]] = pm
        nodes = np.flatnonzero(sub.reshape(-1, order="F"))
        for ch in s.get("dofs", "xyz"):
            fixed[3 * nodes + _AX[ch]] = True

    for ld in loads:
        fm = face_node_mask(ld["face"], nx, ny, nz)
        pm = pattern_mask((nx1, ny1, nz1), ld["face"], ld.get("pattern", "all"),
                          ld.get("frac", 0.3))
        sub = np.zeros_like(fm)
        ax = ld["face"][0]
        if ax == "x":
            sub[np.flatnonzero(fm.any(axis=(1, 2)))[0], :, :] = pm
        elif ax == "y":
            sub[:, np.flatnonzero(fm.any(axis=(0, 2)))[0], :] = pm
        else:
            sub[:, :, np.flatnonzero(fm.any(axis=(0, 1)))[0]] = pm
        nodes = np.flatnonzero(sub.reshape(-1, order="F"))
        ld_dir = ld.get("dir", "-z")
        sign = -1.0 if ld_dir[0] == "-" else 1.0
        axis = _AX[ld_dir[-1]]
        F[3 * nodes + axis] += sign * ld.get("mag", 1.0)

    return fixed, F, None


# --------------------------------------------------------------------------
# density filter
# --------------------------------------------------------------------------

def build_filter(active_idx, shape, rmin):
    """
    Linear "cone" density filter over the active elements only.

    active_idx : (nx,ny,nz) int array; -1 where inactive, else the active index.
    Returns (H csr, Hs) with H shape (na, na).
    """
    nx, ny, nz = shape
    na = int(active_idx.max()) + 1
    r = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w = rmin - np.sqrt(dx * dx + dy * dy + dz * dz)
                if w <= 0:
                    continue
                sx = slice(max(dx, 0), nx + min(dx, 0))
                sy = slice(max(dy, 0), ny + min(dy, 0))
                sz = slice(max(dz, 0), nz + min(dz, 0))
                tx = slice(max(-dx, 0), nx + min(-dx, 0))
                ty = slice(max(-dy, 0), ny + min(-dy, 0))
                tz = slice(max(-dz, 0), nz + min(-dz, 0))
                a = active_idx[tx, ty, tz]
                b = active_idx[sx, sy, sz]
                m = (a >= 0) & (b >= 0)
                if not m.any():
                    continue
                rows.append(a[m])
                cols.append(b[m])
                vals.append(np.full(int(m.sum()), w))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    H = coo_matrix((vals, (rows, cols)), shape=(na, na)).tocsr()
    Hs = np.asarray(H.sum(1)).ravel()
    return H, Hs


# --------------------------------------------------------------------------
# linear solver
# --------------------------------------------------------------------------

def rigid_body_modes(free_dofs, nx, ny, nz):
    """
    The six rigid-body modes evaluated at every free dof.

    Smoothed-aggregation AMG needs these as the near-nullspace to be an
    effective preconditioner for 3-D elasticity — the three translations alone
    are not enough and the solver stalls once the design goes near-0/1.
    """
    nx1, ny1 = nx + 1, ny + 1
    node = free_dofs // 3
    comp = free_dofs % 3
    ix = node % nx1
    iy = (node // nx1) % ny1
    iz = node // (nx1 * ny1)
    # centre the coordinates so the rotation modes are well scaled
    x = ix - nx / 2.0
    y = iy - ny / 2.0
    z = iz - nz / 2.0
    s = max(nx, ny, nz) / 2.0 or 1.0
    x, y, z = x / s, y / s, z / s

    B = np.zeros((len(free_dofs), 6))
    for k in range(3):
        B[comp == k, k] = 1.0
    # rotation about z: (-y, x, 0)
    B[comp == 0, 3] = -y[comp == 0]
    B[comp == 1, 3] = x[comp == 1]
    # rotation about x: (0, -z, y)
    B[comp == 1, 4] = -z[comp == 1]
    B[comp == 2, 4] = y[comp == 2]
    # rotation about y: (z, 0, -x)
    B[comp == 0, 5] = z[comp == 0]
    B[comp == 2, 5] = -x[comp == 2]
    return B


class _Solver:
    """CG preconditioned by smoothed-aggregation AMG."""

    def __init__(self, free_dofs, grid, tol=1e-9, verbose=False):
        self.free = free_dofs
        self.tol = tol
        self.verbose = verbose
        self.B = rigid_body_modes(free_dofs, *grid)
        self.small = len(free_dofs) < 3000
        self.ml = None
        self.reuse = 0
        self.reuse_limit = 3
        self.builds = 0

    def _build(self, K):
        return pyamg.smoothed_aggregation_solver(
            K, B=self.B, max_coarse=500,
            symmetry="hermitian",
            strength=("symmetric", {"theta": 0.0}),
            aggregate="standard",
            smooth=("energy", {"krylov": "cg", "maxiter": 2, "degree": 1,
                               "weighting": "local"}),
            presmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
            postsmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
            max_levels=15,
        )

    def solve(self, K, b, x0=None):
        """Solve K x = b, verifying the TRUE residual before accepting."""
        from scipy.sparse.linalg import spsolve
        K = K.tocsr()
        nb = np.linalg.norm(b)
        if self.small or not _HAVE_AMG:
            return spsolve(K, b), 0

        # A hierarchy built for an earlier iteration's matrix is not a valid SPD
        # preconditioner for the current one; CG then "converges" on its own
        # (preconditioned) residual while returning a wrong x. So always check
        # the true residual ||b - K x|| and rebuild whenever it is not small.
        for attempt in (0, 1):
            if self.ml is None or self.reuse >= self.reuse_limit:
                self.ml = self._build(K)
                self.reuse = 0
                self.builds += 1
            res = []
            x = self.ml.solve(b, tol=self.tol, accel="cg", maxiter=300,
                              residuals=res)
            true_rel = np.linalg.norm(b - K @ x) / (nb + 1e-300)
            if true_rel <= 1e-7:
                self.reuse += 1
                return x, len(res)
            self.ml = None
        return spsolve(K, b), -1


# --------------------------------------------------------------------------
# main optimizer
# --------------------------------------------------------------------------

def optimize(domain=None, shape=None, volfrac=0.3, penal=3.0, rmin=2.0,
             max_iter=60, tol=0.005, move=0.2,
             supports=None, loads=None, load_cases=None, case_weights=None,
             projection=False, beta0=1.0, beta_max=16.0, beta_double_every=15,
             keep=None, nu=0.3, callback=None, verbose=True, seed_noise=0.0,
             penal_continuation=True):
    """
    Run SIMP.

    domain  : bool (nx,ny,nz) design domain (True = material may be placed).
              If None, a solid box of `shape` is used.
    keep    : bool (nx,ny,nz) elements forced solid (non-designable).
    supports/loads : see build_bc.
    load_cases : list of load-spec lists, optimized as INDEPENDENT load cases
              (weighted-sum compliance). A pillar loaded only straight down has
              vertical columns as its exact optimum -- adding a second case (a
              sideways push, say) is what makes the optimizer produce a braced,
              organic structure instead of a slab.
    projection : Heaviside projection with beta continuation (crisper struts).

    Returns dict with 'rho' (nx,ny,nz), history, timings.
    """
    t_start = time.time()
    if domain is None:
        assert shape is not None
        domain = np.ones(shape, dtype=bool)
    domain = np.asarray(domain, dtype=bool)
    nx, ny, nz = domain.shape
    nx1, ny1, nz1 = _node_grid(nx, ny, nz)
    ndof = 3 * nx1 * ny1 * nz1

    if keep is None:
        keep = np.zeros_like(domain)
    keep = np.asarray(keep, dtype=bool) & domain

    active_idx = np.full(domain.shape, -1, dtype=np.int64)
    na = int(domain.sum())
    if na == 0:
        raise ValueError("design domain is empty")
    active_idx[domain] = np.arange(na)

    # ---- element -> dof map, only for active elements
    ex, ey, ez = np.nonzero(domain)
    nodes = np.empty((na, 8), dtype=np.int64)
    for c in range(8):
        dx, dy, dz = _CORNERS[c]
        nodes[:, c] = (ex + dx) + (ey + dy) * nx1 + (ez + dz) * nx1 * ny1
    edof = np.empty((na, 24), dtype=np.int64)
    edof[:, 0::3] = 3 * nodes
    edof[:, 1::3] = 3 * nodes + 1
    edof[:, 2::3] = 3 * nodes + 2

    # ---- boundary conditions
    supports = supports or [{"face": "z-", "pattern": "all", "dofs": "xyz"}]
    if load_cases is None:
        load_cases = [loads or [{"face": "z+", "pattern": "all", "dir": "-z"}]]
    ncase = len(load_cases)
    w = np.ones(ncase) if case_weights is None else np.asarray(case_weights, float)

    fixed = np.zeros(ndof, dtype=bool)
    Fs = []
    for lc in load_cases:
        fx, Fc, _ = build_bc(nx, ny, nz, supports, lc)
        fixed |= fx
        Fs.append(Fc)

    # any node not touching an active element carries no stiffness -> constrain it
    touched = np.zeros(nx1 * ny1 * nz1, dtype=bool)
    touched[nodes.ravel()] = True
    fixed |= np.repeat(~touched, 3)
    for ci, Fc in enumerate(Fs):
        # loads that landed on dead or fixed nodes are dropped, then renormalized
        Fc[fixed] = 0.0
        live = np.abs(Fc).sum()
        if live == 0:
            raise ValueError(
                "load case %d reaches no material — check its face/pattern "
                "against your geometry" % ci)
        Fs[ci] = Fc / live   # unit total load per case: compliances stay comparable
    F = np.stack(Fs, axis=1)                       # (ndof, ncase)

    free = np.flatnonzero(~fixed)
    dof_new = np.full(ndof, -1, dtype=np.int64)
    dof_new[free] = np.arange(len(free))
    edof_f = dof_new[edof]                     # -1 where constrained
    nfree = len(free)

    # ---- assembly pattern (built once)
    KE = hex8_KE(1.0, nu, 1.0)
    KEflat = KE.ravel()
    ii = np.repeat(edof_f, 24, axis=1).ravel()
    jj = np.tile(edof_f, (1, 24)).ravel()
    good = (ii >= 0) & (jj >= 0)
    ii = ii[good].astype(np.int32)
    jj = jj[good].astype(np.int32)
    ke_pat = np.tile(KEflat, na)[good]
    el_of = np.repeat(np.arange(na, dtype=np.int32), 576)[good]
    # collapse duplicates once: sort by (i, j) and remember group boundaries
    order = np.lexsort((jj, ii))
    ii_s, jj_s = ii[order], jj[order]
    ke_s = ke_pat[order]
    el_s = el_of[order]
    newgrp = np.empty(len(ii_s), dtype=bool)
    newgrp[0] = True
    newgrp[1:] = (ii_s[1:] != ii_s[:-1]) | (jj_s[1:] != jj_s[:-1])
    grp_start = np.flatnonzero(newgrp)
    Kr = ii_s[grp_start]
    Kc = jj_s[grp_start]
    nnz = len(grp_start)
    del ii, jj, ke_pat, el_of, ii_s, jj_s, order, newgrp, good

    # coo -> csr reorders the entries; record where each collapsed group lands so
    # that later iterations can refill `data` without rebuilding the structure.
    Kproto = coo_matrix((np.arange(1, nnz + 1, dtype=np.float64), (Kr, Kc)),
                        shape=(nfree, nfree)).tocsr()
    Kproto.sum_duplicates()
    Kproto.sort_indices()
    csr_perm = Kproto.data.astype(np.int64) - 1     # csr slot -> group index
    Kproto.data = np.ones(nnz)

    # ---- filter
    H, Hs = build_filter(active_idx, domain.shape, rmin)

    Emin, Emax = 1e-6, 1.0
    x = np.full(na, volfrac)
    if seed_noise > 0:
        rng = np.random.default_rng(0)
        x = np.clip(x + seed_noise * (rng.random(na) - 0.5), 0.01, 1.0)
    keep_a = keep[domain]
    x[keep_a] = 1.0
    xTilde = x.copy()
    beta = beta0

    keep_frac = keep_a.mean() if na else 0.0
    if keep_frac >= volfrac:
        raise ValueError(
            "the forced-solid 'keep' regions are %.1f%% of the design domain, "
            "which already exceeds the %.1f%% volume budget — nothing is left "
            "to build a structure from. Raise volfrac above %.2f, or shrink the "
            "keep regions." % (100 * keep_frac, 100 * volfrac, keep_frac))
    if keep_frac > 0 and volfrac - keep_frac < 0.03:
        print("  WARNING: keep regions use %.1f%% of the domain, leaving only "
              "%.1f%% for the optimized structure — expect a thin, possibly "
              "disconnected result." % (100 * keep_frac, 100 * (volfrac - keep_frac)),
              flush=True)

    solver = _Solver(free, (nx, ny, nz), verbose=verbose)
    U = np.zeros((ndof, ncase))
    hist = []
    p_cur = penal if not penal_continuation else min(penal, 1.5)

    def project(xt, b):
        if not projection:
            return xt
        eta = 0.5
        return ((np.tanh(b * eta) + np.tanh(b * (xt - eta))) /
                (np.tanh(b * eta) + np.tanh(b * (1 - eta))))

    def dproject(xt, b):
        if not projection:
            return np.ones_like(xt)
        eta = 0.5
        return (b * (1 - np.tanh(b * (xt - eta)) ** 2) /
                (np.tanh(b * eta) + np.tanh(b * (1 - eta))))

    xPhys = project(np.asarray(H @ (x / Hs)), beta)
    xPhys[keep_a] = 1.0
    change = 1.0
    t_solve = 0.0

    for it in range(1, max_iter + 1):
        # --- assemble
        vol_analyzed = float(xPhys.mean())
        Evals = Emin + xPhys ** p_cur * (Emax - Emin)
        sK = ke_s * Evals[el_s]
        data = np.add.reduceat(sK, grp_start)
        K = csr_matrix((data[csr_perm], Kproto.indices, Kproto.indptr),
                       shape=(nfree, nfree))

        # --- solve (one linear system per load case, same matrix)
        t0 = time.time()
        nit = 0
        for ci in range(ncase):
            u, n_ = solver.solve(K, F[free, ci])
            U[free, ci] = u
            nit = max(nit, n_)
        t_solve += time.time() - t0

        # --- compliance + sensitivities (weighted sum over load cases)
        ce = np.zeros(na)
        compliance = 0.0
        for ci in range(ncase):
            Ue = U[edof, ci]                          # (na, 24)
            ce_c = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
            ce += w[ci] * ce_c
            compliance += w[ci] * float((Evals * ce_c).sum())
        dc = -p_cur * xPhys ** (p_cur - 1) * (Emax - Emin) * ce
        dv = np.ones(na)

        dpr = dproject(np.asarray(H @ (x / Hs)), beta)
        dc = np.asarray(H @ ((dc * dpr) / Hs))
        dv = np.asarray(H @ ((dv * dpr) / Hs))

        # --- OC update
        l1, l2 = 1e-12, 1e12
        target = volfrac * na
        xnew = x
        while (l2 - l1) / (l1 + l2) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(np.clip(x * np.sqrt(np.maximum(-dc, 1e-30) / (dv * lmid)),
                                   x - move, x + move), 0.0, 1.0)
            xnew[keep_a] = 1.0
            xt = np.asarray(H @ (xnew / Hs))
            xp = project(xt, beta)
            xp[keep_a] = 1.0
            if xp.sum() > target:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        xPhys = xp

        gray = float(np.mean(4 * xPhys * (1 - xPhys)))
        hist.append(dict(it=it, compliance=compliance, vol=float(xPhys.mean()),
                         vol_analyzed=vol_analyzed,
                         change=change, penal=p_cur, beta=beta, cg=nit,
                         grayness=gray))
        if verbose:
            print("  it %3d  C=%.4e  vol=%.3f  chg=%.3f  p=%.2f  beta=%.1f  "
                  "gray=%.3f  cg=%d" % (it, compliance, xPhys.mean(), change,
                                        p_cur, beta, gray, nit), flush=True)
        if callback:
            callback(hist[-1], xPhys, active_idx)

        # --- continuation
        if penal_continuation and p_cur < penal:
            p_cur = min(penal, p_cur + 0.25)
        elif projection and it % beta_double_every == 0 and beta < beta_max:
            beta = min(beta_max, beta * 2)
            change = 1.0
        elif change < tol and it > 12:
            break

    rho = np.zeros(domain.shape)
    rho[domain] = xPhys
    return dict(rho=rho, domain=domain, hist=hist, na=na, nfree=nfree,
                keep_fraction=float(keep_frac),
                free_volfrac=float((volfrac - keep_frac) / max(1 - keep_frac, 1e-9)),
                seconds=time.time() - t_start, solve_seconds=t_solve,
                compliance=hist[-1]["compliance"], volume=hist[-1]["vol"])
