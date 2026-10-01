"""PC-ALM inference and the weight step, as Brian2 affine yields.

The schedule matches Algorithm 1 of Seely and Gould (arXiv:2605.31022):
hidden activities start on the forward pass, multipliers start at zero,
then T-1 primal-dual cycles

    h ← h − η_h ∇_h L_ρ
    λ ← λ + α r

and one final primal step. The dual step uses the residual of the updated
activities. A weight step then descends on the batch-mean ∇_W L_ρ.
"""

from dataclasses import dataclass

import numpy as np

from lpc_s2.activations import sigma, sigma_prime
from lpc_s2.ops import Ops


@dataclass
class InferenceState:
    """Activities, multipliers, and residuals after one inference."""

    hiddens: list
    multipliers: list
    residuals: list
    prediction: np.ndarray
    energy: float


def _zeros_like(layers):
    return [np.zeros_like(layer) for layer in layers]


def forward_hiddens(ops, weights, x, activation):
    """h_i = σ(W_i h_{i-1}) for hidden layers. h_0 = x stays clamped."""
    hiddens = []
    previous = x
    for weight in weights[:-1]:
        pre = yield from ops.matmul(previous, weight.T)
        previous = sigma(activation, pre)
        hiddens.append(previous)
    return hiddens


def layer_residuals(ops, weights, hiddens, x, activation):
    """r_i = h_i − σ(W_i h_{i-1}), and the pre-activations those residuals use."""
    states = [x, *hiddens]
    residuals = []
    pres = []
    for index, hidden in enumerate(hiddens):
        pre = yield from ops.matmul(states[index], weights[index].T)
        pres.append(pre)
        residual = yield from ops.sub(hidden, sigma(activation, pre))
        residuals.append(residual)
    return residuals, pres


def activity_gradients(ops, weights, hiddens, multipliers, x, y, rho, activation):
    """∇_h L_ρ. The top hidden layer takes the supervised readout term."""
    residuals, pres = yield from layer_residuals(ops, weights, hiddens, x, activation)
    prediction = yield from ops.matmul(hiddens[-1], weights[-1].T)
    supervised = yield from ops.sub(y, prediction)
    depth = len(weights)
    gradients = []
    for layer in range(1, depth):
        composite = yield from ops.axpy(multipliers[layer - 1], rho, residuals[layer - 1])
        if layer < depth - 1:
            slope = sigma_prime(activation, pres[layer])
            next_composite = yield from ops.axpy(multipliers[layer], rho, residuals[layer])
            mixed = next_composite if activation == "identity" else slope * next_composite
            pulled = yield from ops.matmul(mixed, weights[layer])
            gradients.append((yield from ops.sub(composite, pulled)))
        else:
            pulled = yield from ops.matmul(supervised, weights[-1])
            gradients.append((yield from ops.sub(composite, pulled)))
    return gradients


def infer(ops, weights, x, y, activation, T, eta_h, alpha, rho):
    """One inference budget. Weights are not modified."""
    if T < 1:
        raise ValueError("inference budget T must be at least 1")
    hiddens = yield from forward_hiddens(ops, weights, x, activation)
    multipliers = _zeros_like(hiddens)
    for _ in range(T - 1):
        gradients = yield from activity_gradients(
            ops, weights, hiddens, multipliers, x, y, rho, activation
        )
        updated = []
        for hidden, gradient in zip(hiddens, gradients):
            updated.append((yield from ops.axpy(hidden, -eta_h, gradient)))
        hiddens = updated
        residuals, _pres = yield from layer_residuals(ops, weights, hiddens, x, activation)
        updated_multipliers = []
        for multiplier, residual in zip(multipliers, residuals):
            updated_multipliers.append((yield from ops.axpy(multiplier, alpha, residual)))
        multipliers = updated_multipliers
    gradients = yield from activity_gradients(
        ops, weights, hiddens, multipliers, x, y, rho, activation
    )
    updated = []
    for hidden, gradient in zip(hiddens, gradients):
        updated.append((yield from ops.axpy(hidden, -eta_h, gradient)))
    hiddens = updated
    residuals, _pres = yield from layer_residuals(ops, weights, hiddens, x, activation)
    prediction = yield from ops.matmul(hiddens[-1], weights[-1].T)
    energy = augmented_lagrangian(weights, hiddens, multipliers, x, y, rho, activation, residuals, prediction)
    return InferenceState(
        hiddens=hiddens,
        multipliers=multipliers,
        residuals=residuals,
        prediction=prediction,
        energy=energy,
    )


def weight_gradients(ops, weights, hiddens, multipliers, x, y, rho, activation):
    """Batch-mean ∇_W L_ρ. Hidden layers use c = λ + ρ r. The readout is linear."""
    states = [x, *hiddens]
    batch = x.shape[0]
    gradients = []
    for layer in range(1, len(weights)):
        pre = yield from ops.matmul(states[layer - 1], weights[layer - 1].T)
        residual = yield from ops.sub(states[layer], sigma(activation, pre))
        composite = yield from ops.axpy(multipliers[layer - 1], rho, residual)
        slope = sigma_prime(activation, pre)
        signal = composite if activation == "identity" else slope * composite
        scaled_state = states[layer - 1] * (-1.0 / batch)
        gradients.append((yield from ops.matmul(signal.T, scaled_state)))
    prediction = yield from ops.matmul(hiddens[-1], weights[-1].T)
    readout_error = yield from ops.sub(prediction, y)
    scaled_hidden = hiddens[-1] * (1.0 / batch)
    gradients.append((yield from ops.matmul(readout_error.T, scaled_hidden)))
    return gradients


def train_step(ops, weights, x, y, activation, T, eta_h, alpha, rho, eta_theta):
    """Infer, then apply W ← W − η_θ ∇_W L_ρ in place."""
    state = yield from infer(
        ops, weights, x, y, activation=activation, T=T, eta_h=eta_h, alpha=alpha, rho=rho
    )
    gradients = yield from weight_gradients(
        ops, weights, state.hiddens, state.multipliers, x, y, rho, activation
    )
    for weight, gradient in zip(weights, gradients):
        updated = yield from ops.axpy(weight, -eta_theta, gradient)
        weight[...] = updated
    return state


def supervised_mse(ops, weights, x, y, activation):
    """Mean squared error of the feasible forward readout, over every entry."""
    hiddens = yield from forward_hiddens(ops, weights, x, activation)
    prediction = yield from ops.matmul(hiddens[-1], weights[-1].T)
    return float(np.mean((prediction - y) ** 2))


def augmented_lagrangian(weights, hiddens, multipliers, x, y, rho, activation, residuals, prediction):
    """Batch-mean augmented Lagrangian. Arrays are the Brian2 inference state.

    ``weights`` and ``activation`` are accepted so the probe has the same
    signature as the reference energy. Residuals already include σ.
    """
    del weights, activation
    per_sample = 0.5 * np.sum((y - prediction) ** 2, axis=1)
    for residual, multiplier in zip(residuals, multipliers):
        per_sample = per_sample + np.sum(multiplier * residual, axis=1)
        per_sample = per_sample + 0.5 * rho * np.sum(residual * residual, axis=1)
    return float(np.mean(per_sample))


def mean_abs_residual(residuals):
    if not residuals:
        return 0.0
    total = 0.0
    count = 0
    for residual in residuals:
        total += float(np.sum(np.abs(residual)))
        count += residual.size
    return total / count


def schedule(problem):
    """Train, then infer on the held-out batch. Yields one affine per Brian2 step."""
    ops = Ops()
    weights = problem.weights
    losses = [(yield from supervised_mse(ops, weights, problem.x_train, problem.y_train, problem.activation))]
    for _ in range(problem.steps):
        yield from train_step(
            ops,
            weights,
            problem.x_train,
            problem.y_train,
            activation=problem.activation,
            T=problem.T,
            eta_h=problem.eta_h,
            alpha=problem.alpha,
            rho=problem.rho,
            eta_theta=problem.eta_theta,
        )
        losses.append(
            (yield from supervised_mse(ops, weights, problem.x_train, problem.y_train, problem.activation))
        )
    held = yield from infer(
        ops,
        weights,
        problem.x_held,
        problem.y_held,
        activation=problem.activation,
        T=problem.T,
        eta_h=problem.eta_h,
        alpha=problem.alpha,
        rho=problem.rho,
    )
    held_forward = yield from supervised_mse(
        ops, weights, problem.x_held, problem.y_held, problem.activation
    )
    return {
        "losses": losses,
        "heldout_mse": held_forward,
        "energy": held.energy,
        "mean_abs_residual": mean_abs_residual(held.residuals),
        "prediction0": float(held.prediction[0, 0]),
    }
