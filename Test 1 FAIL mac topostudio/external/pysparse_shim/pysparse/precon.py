"""
`pysparse.precon` stand-in.

PySparse's SSOR preconditioner is replaced by smoothed-aggregation AMG when
pyamg is available (far better for 3-D elasticity, which is all ToPy uses this
for), and by Jacobi otherwise. `itsolvers.pcg` knows how to consume either.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

try:
    import pyamg
    _HAVE_AMG = True
except Exception:                                        # pragma: no cover
    _HAVE_AMG = False


class Preconditioner(object):
    def __init__(self, A, near_nullspace=None, kind=None):
        self.A = sp.csr_matrix(A)
        self.kind = kind
        self.M = None
        if kind != "jacobi" and _HAVE_AMG:
            try:
                ml = pyamg.smoothed_aggregation_solver(
                    self.A, B=near_nullspace, max_coarse=500,
                    symmetry="hermitian",
                    strength=("symmetric", {"theta": 0.0}),
                    smooth=("energy", {"krylov": "cg", "maxiter": 2,
                                       "degree": 1, "weighting": "local"}),
                    presmoother=("gauss_seidel",
                                 {"sweep": "symmetric", "iterations": 2}),
                    postsmoother=("gauss_seidel",
                                  {"sweep": "symmetric", "iterations": 2}),
                    max_levels=15)
                self.M = ml.aspreconditioner(cycle="V")
                self.kind = "amg"
            except Exception:
                self.M = None
        if self.M is None:
            d = self.A.diagonal().copy()
            d[d == 0] = 1.0
            inv = 1.0 / d
            self.M = LinearOperator(self.A.shape,
                                    matvec=lambda v: inv * v, dtype=float)
            self.kind = "jacobi"


def ssor(A, near_nullspace=None, omega=1.0, steps=1):
    return Preconditioner(A, near_nullspace=near_nullspace)


def jacobi(A, **kw):
    return Preconditioner(A, kind="jacobi")
