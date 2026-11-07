"""Pooled Fashion-MNIST helpers and a short Brian2 minibatch. No download."""

import numpy as np

from lpc_s2.ann import affine_minibatch_train, predict, train_backprop_minibatches
from lpc_s2.backend import PcalmBrian2Backend
from lpc_s2.demo import init_weights
from lpc_s2.fashion import fashion_arrays, minibatch_order, pcalm_minibatches
from lpc_s2.fashion_mnist import one_hot
from lpc_s2.network import Problem, build_network
from tests.test_pcalm import _drive


def test_pooled_split_flattens_to_the_block_width():
    images = np.arange(4 * 8 * 8, dtype=np.float64).reshape(4, 8, 8) / 64.0
    dataset = {
        "train_images": images,
        "train_labels": np.array([0, 0, 1, 1]),
        "test_images": images,
        "test_labels": np.array([0, 0, 1, 1]),
    }
    split = fashion_arrays(dataset, train_per_class=2, test_per_class=2, train_seed=0, test_seed=1, pool=4)
    assert split["x_train"].shape == (4, 4)
    assert split["x_test"].shape == (4, 4)
    assert set(split["y_train"].tolist()) == {0, 1}


def test_accumulator_batch_caps_the_synapse_grid():
    weights = init_weights((6, 4, 2), seed=0, scale=0.1)
    x = np.zeros((40, 6))
    y = np.zeros((40, 2))
    problem = Problem(
        sizes=(6, 4, 2),
        weights=weights,
        x_train=x,
        y_train=y,
        x_held=x[:4],
        y_held=y[:4],
        steps=1,
        T=1,
        eta_h=0.1,
        alpha=1.0,
        rho=1.0,
        eta_theta=0.1,
        activation="tanh",
        accumulator_batch=8,
    )
    network = build_network(problem)
    assert network.accumulator_shape == (8, 8, 6)


def test_brian2_minibatch_backprop_matches_numpy():
    rng = np.random.default_rng(0)
    sizes = (4, 5, 3)
    weights = init_weights(sizes, seed=1, scale=0.2)
    x = rng.normal(size=(16, 4))
    labels = rng.integers(0, 3, size=16)
    y = one_hot(labels, 3)
    x_test = rng.normal(size=(6, 4))
    order = minibatch_order(16, seed=3)
    epochs = 2
    eta = 0.1
    batch = 8
    expected = train_backprop_minibatches(
        weights, x, y, order, epochs=epochs, eta=eta, activation="tanh", batch_size=batch
    )
    problem = Problem(
        sizes=sizes,
        weights=[weight.copy() for weight in weights],
        x_train=x,
        y_train=y,
        x_held=x_test,
        y_held=np.zeros((6, 3)),
        steps=epochs,
        T=1,
        eta_h=0.1,
        alpha=1.0,
        rho=1.0,
        eta_theta=eta,
        activation="tanh",
        accumulator_batch=batch,
    )
    backend = PcalmBrian2Backend()
    metrics = backend.integrate(
        build_network(problem),
        affine_minibatch_train(
            weights, x, y, x_test, order, epochs, eta, "tanh", batch
        ),
        5000,
    )
    assert backend.integrated_on_brian2
    np.testing.assert_allclose(metrics["train_prediction"], predict(expected, x, "tanh"))
    np.testing.assert_allclose(metrics["test_prediction"], predict(expected, x_test, "tanh"))
    for left, right in zip(metrics["weights"], expected):
        np.testing.assert_allclose(left, right)


def test_brian2_pcalm_minibatches_match_the_numpy_affine_map():
    rng = np.random.default_rng(1)
    sizes = (3, 4, 2)
    weights = init_weights(sizes, seed=2, scale=0.15)
    x = rng.normal(size=(8, 3))
    y = one_hot(rng.integers(0, 2, size=8), 2)
    x_test = rng.normal(size=(4, 3))
    order = np.arange(8)
    hyper = dict(
        epochs=1,
        batch_size=4,
        activation="tanh",
        T=2,
        eta_h=0.05,
        alpha=1.0,
        rho=1.0,
        eta=0.1,
    )
    expected = _drive(
        pcalm_minibatches(
            [weight.copy() for weight in weights], x, y, x_test, order, **hyper
        )
    )
    problem = Problem(
        sizes=sizes,
        weights=[weight.copy() for weight in weights],
        x_train=x,
        y_train=y,
        x_held=x_test,
        y_held=np.zeros_like(x_test[:, :2]),
        steps=1,
        T=hyper["T"],
        eta_h=hyper["eta_h"],
        alpha=hyper["alpha"],
        rho=hyper["rho"],
        eta_theta=hyper["eta"],
        activation="tanh",
        accumulator_batch=4,
    )
    backend = PcalmBrian2Backend()
    metrics = backend.integrate(
        build_network(problem),
        pcalm_minibatches(problem.weights, x, y, x_test, order, **hyper),
        5000,
    )
    assert backend.integrated_on_brian2
    np.testing.assert_allclose(metrics["train_prediction"], expected["train_prediction"])
    np.testing.assert_allclose(metrics["test_prediction"], expected["test_prediction"])
