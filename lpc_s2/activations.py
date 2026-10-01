"""Elementwise σ used by the feedforward chain."""

import numpy as np

SUPPORTED = ("identity", "tanh", "relu")


def sigma(name, pre):
    """Apply σ to a pre-activation."""
    pre = np.asarray(pre, dtype=np.float64)
    if name == "identity":
        return pre
    if name == "tanh":
        return np.tanh(pre)
    if name == "relu":
        return np.maximum(pre, 0.0)
    raise ValueError("unsupported activation %r; expected one of %s" % (name, ", ".join(SUPPORTED)))


def sigma_prime(name, pre):
    """Elementwise σ′(pre)."""
    pre = np.asarray(pre, dtype=np.float64)
    if name == "identity":
        return np.ones_like(pre)
    if name == "tanh":
        value = np.tanh(pre)
        return 1.0 - value * value
    if name == "relu":
        return (pre > 0.0).astype(np.float64, copy=False)
    raise ValueError("unsupported activation %r; expected one of %s" % (name, ", ".join(SUPPORTED)))
