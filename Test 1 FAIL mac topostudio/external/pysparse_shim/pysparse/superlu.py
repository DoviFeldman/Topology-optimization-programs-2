"""`pysparse.superlu` stand-in, backed by SciPy's SuperLU (`splu`)."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class _Factor(object):
    def __init__(self, A):
        self._lu = splu(sp.csc_matrix(A))

    def solve(self, b, x=None):
        """PySparse writes the solution into `x`; return it as well."""
        y = self._lu.solve(np.asarray(b, dtype=float))
        if x is not None:
            x[:] = y
        return y


def factorize(A, **kw):
    return _Factor(A)
