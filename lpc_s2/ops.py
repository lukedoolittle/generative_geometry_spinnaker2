"""Affine steps yielded to the Brian2 accumulator.

Each yielded tuple is one LIF update, ``offset + left @ right``. Scalar
multiples are folded into the synapse weights; Brian2 performs the sum.
"""

import numpy as np


def _as_matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("%s must be a matrix" % name)
    return array


class Ops:
    """PC-ALM linear algebra. Methods are generators of one Brian2 step."""

    def matmul(self, left, right):
        """Yield ``left @ right`` and return the Brian2 result."""
        left = _as_matrix(left, "left")
        right = _as_matrix(right, "right")
        if left.shape[1] != right.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        return (yield (None, left, right))

    def axpy(self, base, scale, other):
        """Yield ``base + scale * other`` for two matrices of the same shape."""
        base = _as_matrix(base, "base")
        other = _as_matrix(other, "other")
        if base.shape != other.shape:
            raise ValueError("axpy shapes differ: %s vs %s" % (base.shape, other.shape))
        width = base.shape[1]
        diagonal = np.eye(width, dtype=np.float64) * float(scale)
        return (yield (base, other, diagonal))

    def sub(self, left, right):
        return (yield from self.axpy(left, -1.0, right))
