"""
A drop-in stand-in for the parts of PySparse that ToPy uses.

ToPy (williamhunter/ToPy) is written for Python 2.7 and PySparse, which has no
Apple-Silicon build and is effectively unmaintained. Rather than rewrite ToPy,
this package provides the handful of PySparse names it imports, backed by
SciPy (and pyamg for the 3-D iterative solve):

    from pysparse import spmatrix            -> spmatrix.ll_mat_sym(n, n)
    from pysparse import superlu             -> superlu.factorize(csr).solve(b, x)
    from pysparse import itsolvers, precon   -> precon.ssor(A); itsolvers.pcg(...)

Semantics follow PySparse's:
  * `ll_mat_sym` stores a symmetric matrix; `update_add_mask_sym(B, ind, mask)`
    scatter-adds the small dense block B into rows/cols `ind` where `mask` is
    non-zero.
  * `delete_rowcols(mask)` KEEPS the rows/columns where mask is non-zero.
  * `pcg(A, b, x, tol, maxit, K)` writes the solution into `x` in place and
    returns `(info, iterations, relative_error)`, info < 0 meaning failure.
"""

from . import spmatrix, superlu, itsolvers, precon  # noqa: F401

__all__ = ["spmatrix", "superlu", "itsolvers", "precon"]
