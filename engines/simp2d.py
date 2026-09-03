"""
2D SIMP topology optimization (self-contained, no external TO library).

This is a NumPy/SciPy implementation of the classic density-based SIMP method
(Solid Isotropic Material with Penalization), in the spirit of the well-known
DTU "top88" educational code (Andreassen et al., 2011). It minimizes structural
compliance (maximizes stiffness) for a given material volume fraction.

Nothing here depends on any external optimizer — it always runs anywhere NumPy
and SciPy are installed. Output is a density field in [0, 1] on an nely x nelx
grid, which you can threshold and, for the 3D version, export to STL.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _element_stiffness(E=1.0, nu=0.3):
    """Stiffness matrix for a bilinear 4-node plane-stress quad element."""
    k = np.array([
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ])
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


# Node numbering is column-major: node(i, j) = i * (nely + 1) + j, for column
# i in [0, nelx] and row j in [0, nely] (j = 0 is the top). Each node has two
# dofs: x = 2*node, y = 2*node + 1.
def _node(i, j, nely):
    return i * (nely + 1) + j


def load_case(name, nelx, nely):
    """Return (fixed_dofs, F) for a named boundary-condition / load preset."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    name = (name or "mbb").lower()

    if name == "mbb":
        # Half MBB beam: vertical load at the top-left corner; left edge on
        # rollers (x fixed, symmetry line); bottom-right corner on a roller.
        F[2 * _node(0, 0, nely) + 1] = -1.0
        fixed = [2 * _node(0, j, nely) for j in range(nely + 1)]      # left x-dofs
        fixed.append(2 * _node(nelx, nely, nely) + 1)                 # bottom-right y-dof
    elif name == "cantilever":
        # Cantilever fixed along the whole left edge, point load down at the
        # middle of the right (free) edge.
        F[2 * _node(nelx, nely // 2, nely) + 1] = -1.0
        fixed = []
        for j in range(nely + 1):
            fixed.append(2 * _node(0, j, nely))       # x
            fixed.append(2 * _node(0, j, nely) + 1)   # y
    elif name == "bridge":
        # Deck loaded downward along the top edge; supported (pinned) at the
        # two bottom corners.
        top_nodes = [_node(i, 0, nely) for i in range(nelx + 1)]
        for n in top_nodes:
            F[2 * n + 1] = -1.0 / len(top_nodes)
        fixed = []
        for n in (_node(0, nely, nely), _node(nelx, nely, nely)):
            fixed.append(2 * n)       # x
            fixed.append(2 * n + 1)   # y
    else:
        raise ValueError("unknown load case: %r (use mbb, cantilever, bridge)" % name)

    return np.array(sorted(set(fixed)), dtype=int), F


def optimize(nelx=120, nely=40, volfrac=0.5, penal=3.0, rmin=2.4,
             load="mbb", max_iter=60, tol=0.01, callback=None):
    """
    Run 2D SIMP optimization.

    Returns the physical density field as an (nely, nelx) array in [0, 1].
    If `callback` is given it is called as callback(it, change, compliance, rho)
    after every iteration, where rho is the current (nely, nelx) density.
    """
    Emin, Emax = 1e-9, 1.0
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = _element_stiffness()

    # Element -> dof connectivity.
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
                              2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Density filter (linear hat, radius rmin).
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(max(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(min(i + np.ceil(rmin), nelx))
            ll1 = int(max(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(min(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = max(0.0, fac)
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    fixed, F = load_case(load, nelx, nely)
    dofs = np.arange(ndof)
    free = np.setdiff1d(dofs, fixed)
    U = np.zeros(ndof)

    x = volfrac * np.ones(nely * nelx)
    xPhys = x.copy()
    change = 1.0
    compliance = 0.0

    for it in range(1, max_iter + 1):
        # FE analysis.
        sK = ((KE.flatten()[np.newaxis]).T *
              (Emin + xPhys ** penal * (Emax - Emin))).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]
        U[free] = spsolve(K, F[free])

        # Compliance and sensitivities.
        ce = (np.dot(U[edofMat], KE) * U[edofMat]).sum(1)
        compliance = float(((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum())
        dc = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv = np.ones(nely * nelx)

        # Filter sensitivities.
        dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
        dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # Optimality-criteria update.
        l1, l2, move = 0.0, 1e9, 0.2
        xnew = x.copy()
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
                   np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xPhys = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0]
            if xPhys.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid

        change = float(np.linalg.norm(xnew - x, np.inf))
        x = xnew

        rho = xPhys.reshape((nelx, nely)).T
        if callback is not None:
            callback(it, change, compliance, rho)
        if change < tol:
            break

    return xPhys.reshape((nelx, nely)).T


if __name__ == "__main__":
    # tiny smoke test
    def cb(it, ch, c, rho):
        print("it %2d  change %.3f  compliance %.3f  vol %.3f" %
              (it, ch, c, rho.mean()))
    rho = optimize(nelx=90, nely=30, volfrac=0.5, load="mbb", max_iter=40)
    print("final density grid:", rho.shape, "mean", round(float(rho.mean()), 3))
