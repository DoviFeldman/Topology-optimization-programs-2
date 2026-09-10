"""
`pysparse.itsolvers` stand-in: preconditioned conjugate gradients on SciPy.

Signature and return convention match PySparse: the solution is written into
`x` in place and `(info, iterations, relative_error)` is returned, with a
negative `info` meaning "did not converge" (which is what ToPy checks).

The accepted answer is checked against the TRUE residual ||b - A x||, and falls
back to a direct solve if CG has not actually converged — a preconditioner that
has drifted out of date can otherwise let CG report success on a wrong answer.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve


def pcg(A, b, x, tol=1e-8, maxit=8000, K=None):
    A = sp.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    M = getattr(K, "M", K)

    it = {"n": 0}

    def _count(_xk):
        it["n"] += 1

    try:
        y, info = cg(A, b, rtol=tol, maxiter=maxit, M=M, callback=_count)
    except TypeError:                      # SciPy < 1.12 spells it `tol`
        y, info = cg(A, b, tol=tol, maxiter=maxit, M=M, callback=_count)

    nb = np.linalg.norm(b)
    rel = np.linalg.norm(b - A @ y) / (nb + 1e-300)
    if rel > 1e-6:
        y = spsolve(sp.csc_matrix(A), b)
        rel = np.linalg.norm(b - A @ y) / (nb + 1e-300)
        info = 0
    x[:] = y
    return (0 if info == 0 else -1), it["n"], float(rel)
