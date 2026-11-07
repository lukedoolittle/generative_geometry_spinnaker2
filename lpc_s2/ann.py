"""Full-batch backprop baseline for the same chain the PC-ALM demo trains.

No biases, same elementwise σ. The descended objective is the batch mean of
``1/2 ||y − ŷ||^2``, which is the supervised term in the augmented Lagrangian.
Reported errors are the entrywise mean of ``(y − ŷ)^2``, the same quantity
``supervised_mse`` prints for PC-ALM.

``train_backprop`` is the NumPy reference. ``affine_train`` is that same
update as Brian2 affine steps, the path the demo runs.
"""

import numpy as np

from lpc_s2.activations import sigma, sigma_prime
from lpc_s2.ops import Ops


def predict(weights, x, activation):
    """Forward map ŷ = W_L σ(W_{L-1} … σ(W_1 x))."""
    state = np.asarray(x, dtype=np.float64)
    for weight in weights[:-1]:
        state = sigma(activation, state @ weight.T)
    return state @ weights[-1].T


def forward_mse(weights, x, y, activation):
    """Entrywise mean squared error of ``predict``."""
    prediction = predict(weights, x, activation)
    return float(np.mean((prediction - np.asarray(y, dtype=np.float64)) ** 2))


def backprop_gradients(weights, x, y, activation):
    """Batch-mean gradient of mean ``1/2 ||y − ŷ||^2`` through the chain."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pres = []
    states = [x]
    for weight in weights[:-1]:
        pre = states[-1] @ weight.T
        pres.append(pre)
        states.append(sigma(activation, pre))
    prediction = states[-1] @ weights[-1].T
    batch = x.shape[0]
    error = prediction - y
    gradients = [None] * len(weights)
    gradients[-1] = (error.T @ states[-1]) / batch
    adjoint = error @ weights[-1]
    for index in range(len(weights) - 2, -1, -1):
        local = sigma_prime(activation, pres[index]) * adjoint
        gradients[index] = (local.T @ states[index]) / batch
        adjoint = local @ weights[index]
    return gradients


def train_backprop(weights, x, y, steps, eta, activation):
    """Gradient descent for ``steps`` full-batch updates of size ``eta``.

    Copies ``weights``. Returns the entrywise MSE before and after every
    update (``steps + 1`` values) and the trained weights.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    trained = [np.array(weight, dtype=np.float64, copy=True) for weight in weights]
    losses = [forward_mse(trained, x, y, activation)]
    for _ in range(steps):
        gradients = backprop_gradients(trained, x, y, activation)
        for weight, gradient in zip(trained, gradients):
            weight -= eta * gradient
        losses.append(forward_mse(trained, x, y, activation))
    return losses, trained


def _affine_mse(ops, weights, x, y, activation):
    """Entrywise MSE. Hidden matmuls and the readout are Brian2 steps."""
    state = x
    for weight in weights[:-1]:
        pre = yield from ops.matmul(state, weight.T)
        state = sigma(activation, pre)
    prediction = yield from ops.matmul(state, weights[-1].T)
    return float(np.mean((prediction - y) ** 2))


def _affine_gradients(ops, weights, x, y, activation):
    """Same gradient as ``backprop_gradients``, with products on the LIF."""
    pres = []
    states = [x]
    for weight in weights[:-1]:
        pre = yield from ops.matmul(states[-1], weight.T)
        pres.append(pre)
        states.append(sigma(activation, pre))
    prediction = yield from ops.matmul(states[-1], weights[-1].T)
    batch = x.shape[0]
    error = yield from ops.sub(prediction, y)
    scaled_hidden = states[-1] * (1.0 / batch)
    gradients = [(yield from ops.matmul(error.T, scaled_hidden))]
    adjoint = yield from ops.matmul(error, weights[-1])
    for index in range(len(weights) - 2, -1, -1):
        local = sigma_prime(activation, pres[index]) * adjoint
        scaled_state = states[index] * (1.0 / batch)
        gradients.insert(0, (yield from ops.matmul(local.T, scaled_state)))
        adjoint = yield from ops.matmul(local, weights[index])
    return gradients


def affine_train(weights, x, y, x_held, y_held, steps, eta, activation):
    """``train_backprop`` as a generator of Brian2 affine steps.

    Copies ``weights``. Returns training MSEs, held-out MSE, and the
    trained weights. Scalar ``1/batch`` factors stay in NumPy, as they do
    in the PC-ALM weight step. Every matmul and weight update is yielded.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    ops = Ops()
    trained = [np.array(weight, dtype=np.float64, copy=True) for weight in weights]
    losses = [(yield from _affine_mse(ops, trained, x, y, activation))]
    for _ in range(steps):
        gradients = yield from _affine_gradients(ops, trained, x, y, activation)
        for weight, gradient in zip(trained, gradients):
            updated = yield from ops.axpy(weight, -eta, gradient)
            weight[...] = updated
        losses.append((yield from _affine_mse(ops, trained, x, y, activation)))
    heldout_mse = yield from _affine_mse(ops, trained, x_held, y_held, activation)
    return {
        "losses": losses,
        "heldout_mse": heldout_mse,
        "weights": trained,
    }
