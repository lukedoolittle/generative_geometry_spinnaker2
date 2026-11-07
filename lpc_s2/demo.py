"""Identity-chain demo: train on a frozen teacher, then infer on held-out data."""

import numpy as np

from lpc_s2.ann import affine_train
from lpc_s2.backend import PcalmBrian2Backend
from lpc_s2.network import Problem, build_network


def init_weights(sizes, seed, scale):
    """Gaussian weights, shape (fan_out, fan_in), no biases."""
    rng = np.random.default_rng(seed)
    weights = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        weights.append(rng.normal(0.0, scale, size=(fan_out, fan_in)))
    return weights


def teacher_predict(weights, inputs):
    """Feasible identity forward pass used only to build the synthetic targets."""
    hidden = inputs
    for weight in weights[:-1]:
        hidden = hidden @ weight.T
    return hidden @ weights[-1].T


def identity_chain_problem():
    """Identity-chain regression from the Python LPC demo, with dual rate 2.

    Teacher seed 1, student seed 2, data seed 0, sizes (2, 8, 1).
    Steps, weight step, and initialization match that demo. ``alpha = 2``
    lets the multiplier move far enough within ``T = 20`` that PC-ALM
    finishes below the matched backprop ANN on this batch.
    """
    rng = np.random.default_rng(0)
    sizes = (2, 8, 1)
    teacher = init_weights(sizes, seed=1, scale=0.6)
    student = init_weights(sizes, seed=2, scale=0.15)
    inputs = rng.normal(size=(64, 2))
    targets = teacher_predict(teacher, inputs)
    heldout_x = rng.normal(size=(16, 2))
    heldout_y = teacher_predict(teacher, heldout_x)
    return Problem(
        sizes=sizes,
        weights=student,
        x_train=inputs,
        y_train=targets,
        x_held=heldout_x,
        y_held=heldout_y,
        steps=40,
        T=20,
        eta_h=0.1,
        alpha=2.0,
        rho=1.0,
        eta_theta=0.15,
        activation="identity",
    )


def step_cap(problem):
    """Upper bound on Brian2 milliseconds. The schedule stops itself earlier."""
    hidden_layers = len(problem.sizes) - 2
    per_inference = (48 + 20 * hidden_layers) * int(problem.T) + 64
    return (int(problem.steps) + 2) * per_inference + 32


def main():
    problem = identity_chain_problem()
    # Same weights, samples, step count, and step size as PC-ALM. No biases.
    ann_init = [weight.copy() for weight in problem.weights]
    network = build_network(problem)
    backend = PcalmBrian2Backend()
    result = backend.run(network, step_cap(problem))
    ann_backend = PcalmBrian2Backend()
    ann_metrics = ann_backend.integrate(
        network,
        affine_train(
            ann_init,
            problem.x_train,
            problem.y_train,
            problem.x_held,
            problem.y_held,
            steps=problem.steps,
            eta=problem.eta_theta,
            activation=problem.activation,
        ),
        step_cap(problem),
    )
    print("PC-ALM synthetic regression on SpiNNaker2 (identity chain, sizes %s)" % (problem.sizes,))
    print("backend: %s" % result.backend_name)
    print(
        "simulator: Brian2 %s through spinnaker2.brian2_sim.Brian2Backend"
        % result.brian2_version
    )
    print("board: none")
    print("brian2 timesteps: %d" % result.timesteps)
    print("PC-ALM affine steps on the LIF accumulator: %d" % result.affine_steps)
    print("ANN affine steps on the LIF accumulator:    %d" % ann_backend.affine_steps)
    print("ANN: full-batch backprop on the same Brian2 accumulator")
    print("forward MSE before training:       %.6f" % result.losses[0])
    print("PC-ALM forward MSE after training: %.6f" % result.losses[-1])
    print("ANN forward MSE after training:    %.6f" % ann_metrics["losses"][-1])
    print("PC-ALM held-out forward MSE:       %.6f" % result.heldout_mse)
    print("ANN held-out forward MSE:          %.6f" % ann_metrics["heldout_mse"])
    print("inference energy L_rho:       %.6f" % result.energy)
    print("mean |residual| after inference: %.6e" % result.mean_abs_residual)
    print(
        "prediction[0]: %.4f  target[0]: %.4f"
        % (result.prediction0, problem.y_held[0, 0])
    )
    if not result.integrated_on_brian2 or not ann_backend.integrated_on_brian2:
        return 1
    if not np.isfinite(result.losses[-1]) or not np.isfinite(result.energy):
        return 1
    if not np.isfinite(ann_metrics["losses"][-1]) or not np.isfinite(ann_metrics["heldout_mse"]):
        return 1
    if not result.losses[-1] < result.losses[0]:
        return 1
    if not (
        result.losses[-1] < ann_metrics["losses"][-1]
        and result.heldout_mse < ann_metrics["heldout_mse"]
    ):
        return 1
    return 0
