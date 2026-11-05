"""Brian2 backend and a short PC-ALM run."""

import numpy as np

from lpc_s2.algorithm import forward_sweep, infer, layer_residuals
from lpc_s2.backend import PcalmBrian2Backend
from lpc_s2.demo import init_weights, teacher_predict
from lpc_s2.network import Problem, build_network
from lpc_s2.ops import Ops
from spinnaker2 import brian2_sim


def test_backend_subclasses_brian2():
    assert issubclass(PcalmBrian2Backend, brian2_sim.Brian2Backend)


def _drive(gen):
    """Run a schedule with NumPy affine steps, the same map Brian2 integrates."""
    try:
        spec = next(gen)
    except StopIteration as done:
        return done.value
    while True:
        offset, left, right = spec
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        value = left @ right if offset is None else np.asarray(offset, dtype=np.float64) + left @ right
        try:
            spec = gen.send(value)
        except StopIteration as done:
            return done.value


def _numpy_sweep(weights, x, y, hidden, multiplier, eta_h, alpha, rho, update_activities, update_multipliers):
    """One forward sweep for a single hidden layer. λ ← λ + α r while h is written."""
    activated = x @ weights[0].T
    if update_activities:
        residual = hidden - activated
        prediction = hidden @ weights[1].T
        supervised = y - prediction
        gradient = multiplier + rho * residual - supervised @ weights[1]
        hidden = hidden - eta_h * gradient
    else:
        hidden = activated
        multiplier = np.zeros_like(hidden)
    residual = hidden - activated
    if update_multipliers:
        multiplier = multiplier + alpha * residual
    return hidden, multiplier


def _numpy_train(weights, x, y, steps, T, eta_h, alpha, rho, eta_theta):
    """Independent NumPy PC-ALM, identity chain, one hidden layer.

    Multipliers are updated inside each forward sweep, then the weight step
    reads that λ. The last sweep of an inference is primal only.
    """
    weights = [weight.copy() for weight in weights]

    def forward_mse():
        hidden = x @ weights[0].T
        prediction = hidden @ weights[1].T
        return float(np.mean((prediction - y) ** 2))

    losses = [forward_mse()]
    for _ in range(steps):
        hidden, multiplier = _numpy_sweep(
            weights, x, y, None, None, eta_h, alpha, rho, False, True
        )
        for _cycle in range(T - 1):
            hidden, multiplier = _numpy_sweep(
                weights, x, y, hidden, multiplier, eta_h, alpha, rho, True, True
            )
        hidden, multiplier = _numpy_sweep(
            weights, x, y, hidden, multiplier, eta_h, alpha, rho, True, False
        )
        residual = hidden - x @ weights[0].T
        signal = multiplier + rho * residual
        batch = x.shape[0]
        weights[0] = weights[0] - eta_theta * (-(signal.T @ x) / batch)
        prediction = hidden @ weights[1].T
        weights[1] = weights[1] - eta_theta * (((prediction - y).T @ hidden) / batch)
        losses.append(forward_mse())
    return losses, weights


def test_forward_sweep_updates_lambda_with_the_residual():
    """Each forward sweep writes λ ← λ + α r as it commits that layer's activity."""
    weights = [np.array([[0.5]]), np.array([[1.5]])]
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    alpha = 0.4
    eta_h = 0.05
    rho = 1.0
    state = _drive(
        infer(Ops(), weights, x, y, "identity", T=2, eta_h=eta_h, alpha=alpha, rho=rho)
    )
    hidden, multiplier = _numpy_sweep(weights, x, y, None, None, eta_h, alpha, rho, False, True)
    hidden, multiplier = _numpy_sweep(
        weights, x, y, hidden, multiplier, eta_h, alpha, rho, True, True
    )
    np.testing.assert_allclose(state.multipliers[0], multiplier)
    np.testing.assert_allclose(multiplier, alpha * (hidden - x @ weights[0].T))

    rng = np.random.default_rng(0)
    chain = [
        rng.normal(scale=0.2, size=(3, 2)),
        rng.normal(scale=0.2, size=(4, 3)),
        rng.normal(scale=0.2, size=(2, 4)),
    ]
    batch_x = rng.normal(size=(3, 2))
    batch_y = rng.normal(size=(3, 2))
    hiddens, multipliers = _drive(
        forward_sweep(
            Ops(),
            chain,
            batch_x,
            "tanh",
            y=batch_y,
            alpha=alpha,
            rho=rho,
            update_multipliers=True,
        )
    )
    for layer in multipliers:
        np.testing.assert_allclose(layer, np.zeros_like(layer), atol=1e-12)
    hiddens, multipliers = _drive(
        forward_sweep(
            Ops(),
            chain,
            batch_x,
            "tanh",
            hiddens=hiddens,
            multipliers=multipliers,
            y=batch_y,
            eta_h=eta_h,
            alpha=alpha,
            rho=rho,
            update_activities=True,
            update_multipliers=True,
        )
    )
    residuals, _pres = _drive(layer_residuals(Ops(), chain, hiddens, batch_x, "tanh"))
    for multiplier, residual in zip(multipliers, residuals):
        np.testing.assert_allclose(multiplier, alpha * residual)


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
