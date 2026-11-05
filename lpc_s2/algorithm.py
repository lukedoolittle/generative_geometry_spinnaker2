"""PC-ALM inference and the weight step, as Brian2 affine yields.

The schedule matches Algorithm 1 of Seely and Gould (arXiv:2605.31022),
with the dual step inside the forward sweep. Each sweep walks layers from
the input toward the readout, writes that layer's activity, and applies

    λ ← λ + α r,    r = h − σ(W h_prev)

before the next layer. The activity step uses the augmented-Lagrangian
combined signal c = λ + ρ r. The first sweep is the feasible initialization
(r = 0, so λ stays 0). Then come T − 1 primal-dual sweeps and one final
primal sweep. A weight step descends on the batch-mean ∇_W L_ρ.
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


def forward_hiddens(ops, weights, x, activation):
    """Feasible activities h_i = σ(W_i h_{i-1}). h_0 = x stays clamped.

    Inference does not use this alone. ``forward_sweep`` writes the same
    activities and updates each layer's multiplier in that pass.
    """
    hiddens, _multipliers = yield from forward_sweep(ops, weights, x, activation)
    return hiddens


def forward_sweep(
    ops,
    weights,
    x,
    activation,
    hiddens=None,
    multipliers=None,
    y=None,
    eta_h=0.0,
    alpha=0.0,
    rho=0.0,
    update_activities=False,
    update_multipliers=False,
):
    """One input-to-output sweep. Each layer's λ is updated as its activity is written.

    Every matmul, residual, activity step, and dual step is a Brian2 affine
    yield. For layer i the sweep writes h_i, forms

        r_i = h_i − σ(W_i h_{i-1})

    from the presynaptic activity this sweep has already committed, and, when
    ``update_multipliers`` is set, immediately applies λ_i ← λ_i + α r_i.
    Activity gradients use the combined signal c = λ + ρ r and are taken from
    the state at the start of the sweep, then this loop commits h and λ from
    the bottom up. On the feasible initialization r_i is 0, so the dual step
    leaves λ at its incoming value (zero when ``multipliers`` is omitted).
    """
    if update_activities:
        gradients = yield from activity_gradients(
            ops, weights, hiddens, multipliers, x, y, rho, activation
        )
    else:
        gradients = None

    previous = x
    new_hiddens = []
    new_multipliers = []
    for index, weight in enumerate(weights[:-1]):
        if update_activities:
            activity = yield from ops.axpy(hiddens[index], -eta_h, gradients[index])
            multiplier = multipliers[index]
            activated = None
        else:
            pre = yield from ops.matmul(previous, weight.T)
            activity = sigma(activation, pre)
            activated = activity
            if multipliers is None:
                multiplier = np.zeros_like(activity)
            else:
                multiplier = multipliers[index]
        if update_multipliers:
            if activated is None:
                pre = yield from ops.matmul(previous, weight.T)
                activated = sigma(activation, pre)
            residual = yield from ops.sub(activity, activated)
            multiplier = yield from ops.axpy(multiplier, alpha, residual)
        new_hiddens.append(activity)
        new_multipliers.append(multiplier)
        previous = activity
    return new_hiddens, new_multipliers


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
    """One inference budget. Weights are not modified.

    Each iteration is a forward sweep. The dual step λ ← λ + α r runs in that
    sweep, layer by layer, using the residual of the activity just written.
    The last sweep is primal only.
    """
    if T < 1:
        raise ValueError("inference budget T must be at least 1")
    hiddens, multipliers = yield from forward_sweep(
        ops,
        weights,
        x,
        activation,
        y=y,
        alpha=alpha,
        rho=rho,
        update_multipliers=True,
    )
    for _ in range(T - 1):
        hiddens, multipliers = yield from forward_sweep(
            ops,
            weights,
            x,
            activation,
            hiddens=hiddens,
            multipliers=multipliers,
            y=y,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
            update_activities=True,
            update_multipliers=True,
        )
    hiddens, multipliers = yield from forward_sweep(
        ops,
        weights,
        x,
        activation,
        hiddens=hiddens,
        multipliers=multipliers,
        y=y,
        eta_h=eta_h,
        alpha=alpha,
        rho=rho,
        update_activities=True,
        update_multipliers=False,
    )
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
    """Infer, then apply W ← W − η_θ ∇_W L_ρ in place.

    Inference updates each λ inside its forward sweep. The weight step reads
    those multipliers, so training uses the same schedule.
    """
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
