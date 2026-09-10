"""`pysparse.spmatrix` stand-in: a triplet-accumulating symmetric sparse matrix."""

import numpy as np
import scipy.sparse as sp


class LLMatSym(object):
    """
    Symmetric sparse matrix built by scatter-adding small dense blocks.

    Entries are accumulated as (row, col, value) triplets and only summed into a
    real sparse matrix when one is asked for, which is what makes assembling a
    few hundred thousand element blocks tolerable in Python.
    """

    def __init__(self, n, m=None):
        self.shape = (int(n), int(m if m is not None else n))
        self._rows = []
        self._cols = []
        self._vals = []
        self._keep = None          # set by delete_rowcols

    # ---------------------------------------------------------------- build
    def update_add_mask_sym(self, B, ind, mask):
        """A[ind[i], ind[j]] += B[i, j] wherever mask[i] and mask[j] are set."""
        ind = np.asarray(ind).ravel()
        mask = np.asarray(mask).ravel().astype(bool)
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            B = np.diag(B)
        elif B.ndim == 3:                 # e.g. [[k...], [k...]] passed by ToPy
            B = np.atleast_2d(np.asarray(B).reshape(B.shape[0], -1))
        if mask.size == ind.size and not mask.all():
            sel = np.flatnonzero(mask)
            ind = ind[sel]
            B = B[np.ix_(sel, sel)]
        n = ind.size
        self._rows.append(np.repeat(ind, n))
        self._cols.append(np.tile(ind, n))
        self._vals.append(B.ravel())

    def update_add_triplets(self, rows, cols, vals):
        """Fast path used by the vectorized assembly in `topy_compat`."""
        self._rows.append(np.asarray(rows))
        self._cols.append(np.asarray(cols))
        self._vals.append(np.asarray(vals, dtype=float))

    # ---------------------------------------------------------------- shape
    def copy(self):
        out = LLMatSym(*self.shape)
        out._rows = list(self._rows)
        out._cols = list(self._cols)
        out._vals = list(self._vals)
        out._keep = None if self._keep is None else self._keep.copy()
        return out

    def delete_rowcols(self, mask):
        """Keep only the rows/columns where `mask` is non-zero (PySparse order)."""
        self._keep = np.flatnonzero(np.asarray(mask).ravel())
        return self

    # ------------------------------------------------------------- convert
    def tocsr(self):
        if not self._rows:
            return sp.csr_matrix(self.shape)
        r = np.concatenate(self._rows)
        c = np.concatenate(self._cols)
        v = np.concatenate(self._vals)
        A = sp.coo_matrix((v, (r, c)), shape=self.shape).tocsr()
        A.sum_duplicates()
        if self._keep is not None:
            A = A[self._keep, :][:, self._keep]
        return A

    # PySparse names
    to_csr = tocsr
    to_sss = tocsr

    def __len__(self):
        return self.shape[0]

    @property
    def nnz(self):
        return sum(len(v) for v in self._vals)


def ll_mat_sym(n, m=None):
    return LLMatSym(n, m)


def ll_mat(n, m=None):
    return LLMatSym(n, m)
