"""
3D SIMP topology optimization (self-contained, no external TO library).

NumPy/SciPy implementation of 3D density-based SIMP, following the element
formulation of the DTU / Liu & Tovar "top3d" educational code (2014). It
minimizes compliance for a clamped-and-loaded solid box and returns a 3D density
field, which `stl_io.voxels_to_stl` can turn into a printable STL.

You can optimize inside the full box, OR inside an arbitrary DESIGN DOMAIN given
as a boolean voxel mask (e.g. voxelized from an uploaded STL). Elements outside
the domain are held void (passive), so material is only ever placed inside your
uploaded shape. Supports and the load are applied on selectable faces of the box.

Depends only on NumPy and SciPy, so it runs anywhere.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _lk_H8(nu=0.3):
    """24x24 stiffness matrix for a trilinear 8-node hex element (unit E)."""
    A = np.array([
        [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
        [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
    ])
    k = (1.0 / 144.0) * A.T.dot(np.array([1.0, nu]))

    def m(idx):
        return np.array([[k[i - 1] for i in row] for row in idx])

    K1 = m([[1, 2, 2, 3, 5, 5], [2, 1, 2, 4, 6, 7], [2, 2, 1, 4, 7, 6],
            [3, 4, 4, 1, 8, 8], [5, 6, 7, 8, 1, 2], [5, 7, 6, 8, 2, 1]])
    K2 = m([[9, 8, 12, 6, 4, 7], [8, 9, 12, 5, 3, 5], [10, 10, 13, 7, 4, 6],
            [6, 5, 11, 9, 2, 10], [4, 3, 5, 2, 9, 12], [11, 4, 6, 12, 10, 13]])
    K3 = m([[6, 7, 4, 9, 12, 8], [7, 6, 4, 10, 13, 10], [5, 5, 3, 8, 12, 9],
            [9, 10, 2, 6, 11, 5], [12, 13, 10, 11, 6, 4], [2, 12, 9, 4, 5, 3]])
    K4 = m([[14, 11, 11, 13, 10, 10], [11, 14, 11, 12, 9, 8], [11, 11, 14, 12, 8, 9],
            [13, 12, 12, 14, 7, 7], [10, 9, 8, 7, 14, 11], [10, 8, 9, 7, 11, 14]])
    K5 = m([[1, 2, 8, 3, 5, 4], [2, 1, 8, 4, 6, 11], [8, 8, 1, 5, 11, 6],
            [3, 4, 5, 1, 8, 2], [5, 6, 11, 8, 1, 8], [4, 11, 6, 2, 8, 1]])
    K6 = m([[14, 11, 7, 13, 10, 12], [11, 14, 7, 12, 9, 2], [7, 7, 14, 10, 2, 9],
            [13, 12, 10, 14, 7, 11], [10, 9, 2, 7, 14, 7], [12, 2, 9, 11, 7, 14]])

    KE = 1.0 / ((nu + 1) * (1 - 2 * nu)) * np.block([
        [K1, K2, K3, K4],
        [K2.T, K5, K6, K3.T],
        [K3.T, K6, K5.T, K2.T],
        [K4, K3, K2, K1.T],
    ])
    return KE


# Node numbering matches top3d: node(iy, ix, iz), 1-indexed,
#   n = 1 + iy + ix*(nely+1) + iz*(nely+1)*(nelx+1),
# and node n's dofs (0-indexed) are 3*(n-1) + [0, 1, 2] = [x, y, z].
_AXIS = {"x": 0, "y": 1, "z": 2}


def _face_nodes(face, nelx, nely, nelz):
    """1-indexed node numbers on a named box face ('x-','x+','y-','y+','z-','z+')."""
    ax = face[0]
    hi = face[1] == "+"
    if ax == "x":
        ix = nelx if hi else 0
        iy, iz = np.meshgrid(np.arange(nely + 1), np.arange(nelz + 1), indexing="ij")
        ix = np.full(iy.shape, ix)
    elif ax == "y":
        iy = nely if hi else 0
        ix, iz = np.meshgrid(np.arange(nelx + 1), np.arange(nelz + 1), indexing="ij")
        iy = np.full(ix.shape, iy)
    else:  # z
        iz = nelz if hi else 0
        iy, ix = np.meshgrid(np.arange(nely + 1), np.arange(nelx + 1), indexing="ij")
        iz = np.full(iy.shape, iz)
    n1 = 1 + iy + ix * (nely + 1) + iz * (nely + 1) * (nelx + 1)
    return n1.flatten()


def _bc(nelx, nely, nelz, fixed_face, load_face, load_dir):
    """Return (fixed_dofs_0indexed, F) for the chosen support/load faces."""
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    fixed_nodes = _face_nodes(fixed_face, nelx, nely, nelz)
    fixed = np.concatenate([3 * (fixed_nodes - 1) + a for a in (0, 1, 2)])

    load_nodes = _face_nodes(load_face, nelx, nely, nelz)
    axis = _AXIS[load_dir[-1]]
    sign = -1.0 if load_dir[0] == "-" else 1.0
    F = np.zeros(ndof)
    F[3 * (load_nodes - 1) + axis] = sign
    return np.unique(fixed), F


def optimize(nelx=32, nely=16, nelz=16, volfrac=0.3, penal=3.0, rmin=1.5,
             max_iter=40, tol=0.01, domain=None,
             fixed_face="x-", load_face="x+", load_dir="-y", callback=None):
    """
    Run 3D SIMP optimization.

    domain : optional boolean array of shape (nely, nelx, nelz). Where False, the
             element is held void (material may only be placed where True). Use
             this to optimize inside an uploaded, voxelized STL shape.
    fixed_face / load_face : one of 'x-','x+','y-','y+','z-','z+'.
    load_dir : e.g. '-y' (down), '+z', ... direction of the applied load.

    Returns a density field of shape (nely, nelx, nelz) in [0, 1].
    Callback signature: callback(it, change, compliance, volume).
    """
    Emin, Emax = 1e-9, 1.0
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    KE = _lk_H8()

    # Element -> dof connectivity (top3d numbering).
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) * (1 + nelz) + 1).reshape(
        (1 + nely, 1 + nelx, 1 + nelz), order="F")
    edofVec = (3 * nodenrs[:-1, :-1, :-1] + 1).flatten(order="F")
    off = np.array([0, 1, 2,
                    3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely + 0, 3 * nely + 1, 3 * nely + 2,
                    -3, -2, -1])
    off = np.concatenate([off, 3 * (nely + 1) * (nelx + 1) + off])
    edofMat = (edofVec[:, None] + off[None, :]) - 1  # 0-indexed dofs
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()

    fixed, F = _bc(nelx, nely, nelz, fixed_face, load_face, load_dir)
    free = np.setdiff1d(np.arange(ndof), fixed)
    U = np.zeros(ndof)

    # Passive (void) elements from the design-domain mask. FE element index is
    # el = j + i*nely + k*nelx*nely  ->  matches domain[j, i, k] via order='F'.
    if domain is not None:
        passive_void = ~np.asarray(domain, dtype=bool).flatten(order="F")
    else:
        passive_void = np.zeros(nele, dtype=bool)
    active = ~passive_void
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError("design domain is empty — nothing to optimize")

    # Density filter (linear, radius rmin) over the 3D element grid.
    iH, jH, sH = [], [], []
    rc = int(np.ceil(rmin)) - 1
    for k in range(nelz):
        for i in range(nelx):
            for j in range(nely):
                e1 = k * nelx * nely + i * nely + j
                for kk in range(max(k - rc, 0), min(k + rc + 1, nelz)):
                    for ii in range(max(i - rc, 0), min(i + rc + 1, nelx)):
                        for jj in range(max(j - rc, 0), min(j + rc + 1, nely)):
                            fac = rmin - np.sqrt((i - ii) ** 2 + (j - jj) ** 2 + (k - kk) ** 2)
                            if fac > 0:
                                e2 = kk * nelx * nely + ii * nely + jj
                                iH.append(e1)
                                jH.append(e2)
                                sH.append(fac)
    H = coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsc()
    Hs = np.asarray(H.sum(1)).flatten()

    x = np.zeros(nele)
    x[active] = volfrac
    xPhys = x.copy()
    change = 1.0
    compliance = 0.0

    for it in range(1, max_iter + 1):
        sK = ((KE.flatten()[np.newaxis]).T *
              (Emin + xPhys ** penal * (Emax - Emin))).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]
        U[free] = spsolve(K, F[free])

        ce = (np.dot(U[edofMat], KE) * U[edofMat]).sum(1)
        compliance = float(((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum())
        dc = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv = np.ones(nele)

        dc = np.asarray(H * (dc / Hs))
        dv = np.asarray(H * (dv / Hs))

        l1, l2, move = 0.0, 1e9, 0.2
        target_vol = volfrac * n_active
        xnew = x.copy()
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
                   np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xnew[passive_void] = 0.0
            xPhys = np.asarray(H * xnew) / Hs
            xPhys[passive_void] = 0.0
            if xPhys.sum() > target_vol:
                l1 = lmid
            else:
                l2 = lmid

        change = float(np.linalg.norm(xnew - x, np.inf))
        x = xnew

        if callback is not None:
            callback(it, change, compliance, float(xPhys[active].mean()))
        if change < tol:
            break

    # FE element el = j + i*nely + k*nelx*nely  ->  reshape to (nely, nelx, nelz).
    return xPhys.reshape((nelz, nelx, nely)).transpose(2, 1, 0)


if __name__ == "__main__":
    def cb(it, ch, c, v):
        print("it %2d  change %.3f  compliance %.2f  vol %.3f" % (it, ch, c, v))
    rho = optimize(nelx=24, nely=12, nelz=12, volfrac=0.3, max_iter=25, callback=cb)
    print("density shape (nely,nelx,nelz):", rho.shape, "mean", round(float(rho.mean()), 3))
