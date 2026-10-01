"""Brian2 backend and a short PC-ALM run."""

import numpy as np

from lpc_s2.backend import PcalmBrian2Backend
from lpc_s2.demo import init_weights, teacher_predict
from lpc_s2.network import Problem, build_network
from spinnaker2 import brian2_sim


def test_backend_subclasses_brian2():
    assert issubclass(PcalmBrian2Backend, brian2_sim.Brian2Backend)


def _numpy_train(weights, x, y, steps, T, eta_h, alpha, rho, eta_theta):
    """Independent NumPy PC-ALM, identity chain, one hidden layer."""
    weights = [weight.copy() for weight in weights]

    def forward_mse():
        hidden = x @ weights[0].T
        prediction = hidden @ weights[1].T
        return float(np.mean((prediction - y) ** 2))

    losses = [forward_mse()]
    for _ in range(steps):
        hidden = x @ weights[0].T
        multiplier = np.zeros_like(hidden)
        for _cycle in range(T - 1):
            residual = hidden - x @ weights[0].T
            prediction = hidden @ weights[1].T
            supervised = y - prediction
            gradient = multiplier + rho * residual - supervised @ weights[1]
            hidden = hidden - eta_h * gradient
            residual = hidden - x @ weights[0].T
            multiplier = multiplier + alpha * residual
        residual = hidden - x @ weights[0].T
        prediction = hidden @ weights[1].T
        supervised = y - prediction
        gradient = multiplier + rho * residual - supervised @ weights[1]
        hidden = hidden - eta_h * gradient
        residual = hidden - x @ weights[0].T
        signal = multiplier + rho * residual
        batch = x.shape[0]
        weights[0] = weights[0] - eta_theta * (-(signal.T @ x) / batch)
        prediction = hidden @ weights[1].T
        weights[1] = weights[1] - eta_theta * (((prediction - y).T @ hidden) / batch)
        losses.append(forward_mse())
    return losses, weights


def test_short_run_matches_numpy_and_uses_brian2():
    rng = np.random.default_rng(0)
    sizes = (2, 3, 1)
    weights = init_weights(sizes, seed=2, scale=0.15)
    inputs = rng.normal(size=(4, 2))
    teacher = init_weights(sizes, seed=1, scale=0.6)
    targets = teacher_predict(teacher, inputs)
    held_x = rng.normal(size=(2, 2))
    held_y = teacher_predict(teacher, held_x)
    hyper = dict(steps=2, T=3, eta_h=0.1, alpha=1.0, rho=1.0, eta_theta=0.15)
    problem = Problem(
        sizes=sizes,
        weights=[weight.copy() for weight in weights],
        x_train=inputs,
        y_train=targets,
        x_held=held_x,
        y_held=held_y,
        activation="identity",
        **hyper,
    )
    expected, _expected_weights = _numpy_train(weights, inputs, targets, **hyper)
    backend = PcalmBrian2Backend()
    result = backend.run(build_network(problem), time_steps=5000)
    assert result.integrated_on_brian2
    assert result.backend_name == "brian2"
    assert result.affine_steps > 0
    assert result.timesteps >= result.affine_steps
    np.testing.assert_allclose(result.losses, expected, rtol=1e-8, atol=1e-8)
    assert result.losses[-1] < result.losses[0]
