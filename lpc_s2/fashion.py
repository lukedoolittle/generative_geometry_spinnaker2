"""Fashion-MNIST on the Brian2 accumulator, pooled so the synapse grid fits.

Usage, from the repository root:

    python -m lpc_s2.fashion

Targets are one-hot class vectors. The loss is the same feasible-forward
mean squared error as the identity-chain demo. Accuracy is the argmax of
the linear readout. Images are average-pooled from 28×28 to 7×7 because the
LIF synapse grid is one factor per column of each product.

The split is 32 training images and 8 test images from each class. Both
models see the same minibatches of 32. ``python -m lpc_s2`` is unchanged.
"""

import numpy as np

from lpc_s2.algorithm import forward_hiddens, train_step
from lpc_s2.ann import affine_minibatch_train, predict
from lpc_s2.backend import PcalmBrian2Backend
from lpc_s2.demo import init_weights
from lpc_s2.fashion_mnist import (
    accuracy,
    average_pool,
    load_fashion_mnist,
    one_hot,
    stratified_indices,
)
from lpc_s2.network import Problem, build_network
from lpc_s2.ops import Ops

TRAIN_PER_CLASS = 32
TEST_PER_CLASS = 8
POOL = 4
SIZES = (49, 24, 10)
ACTIVATION = "tanh"
EPOCHS = 8
BATCH = 32
T = 6
ETA_H = 0.1
ALPHA = 1.0
RHO = 1.0
ETA_THETA = 0.2
N_CLASSES = 10


def fashion_arrays(dataset, train_per_class, test_per_class, train_seed, test_seed, pool):
    """Stratified pooled pixels in ``[0, 1]``, flattened, with integer labels."""
    train_index = stratified_indices(
        dataset["train_labels"], train_per_class, np.random.default_rng(train_seed)
    )
    test_index = stratified_indices(
        dataset["test_labels"], test_per_class, np.random.default_rng(test_seed)
    )
    train_images = average_pool(dataset["train_images"][train_index], pool)
    test_images = average_pool(dataset["test_images"][test_index], pool)
    return {
        "x_train": train_images.reshape(train_images.shape[0], -1),
        "y_train": dataset["train_labels"][train_index],
        "x_test": test_images.reshape(test_images.shape[0], -1),
        "y_test": dataset["test_labels"][test_index],
    }


def minibatch_order(count, seed):
    """One permutation, reused every epoch."""
    order = np.arange(count)
    np.random.default_rng(seed).shuffle(order)
    return order


def _predict_chunks(ops, weights, x, activation, batch_size):
    pieces = []
    step = int(batch_size)
    for start in range(0, x.shape[0], step):
        chunk = x[start:start + step]
        hiddens = yield from forward_hiddens(ops, weights, chunk, activation)
        pieces.append((yield from ops.matmul(hiddens[-1], weights[-1].T)))
    return np.concatenate(pieces, axis=0)


def pcalm_minibatches(
    weights, x, y, x_test, order, epochs, batch_size, activation, T, eta_h, alpha, rho, eta
):
    """Minibatch Algorithm 1. Each ``train_step`` is one inference plus a weight step.

    Returns the feasible readout on the full train and test sets before the
    first update and after the last one. Predictions are chunked at
    ``batch_size``.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    ops = Ops()
    before = yield from _predict_chunks(ops, weights, x, activation, batch_size)
    for _ in range(epochs):
        for start in range(0, order.shape[0], int(batch_size)):
            index = order[start:start + int(batch_size)]
            yield from train_step(
                ops,
                weights,
                x[index],
                y[index],
                activation=activation,
                T=T,
                eta_h=eta_h,
                alpha=alpha,
                rho=rho,
                eta_theta=eta,
            )
    train_prediction = yield from _predict_chunks(ops, weights, x, activation, batch_size)
    test_prediction = yield from _predict_chunks(ops, weights, x_test, activation, batch_size)
    return {
        "before": before,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "weights": weights,
    }


def step_cap(n_train, n_test, epochs, batch_size, inference_budget):
    """Loose upper bound on Brian2 milliseconds. The generator stops earlier."""
    batches = (int(n_train) + int(batch_size) - 1) // int(batch_size)
    per_batch = 40 + 16 * int(inference_budget)
    eval_chunks = batches + (int(n_test) + int(batch_size) - 1) // int(batch_size)
    return int(epochs) * batches * per_batch + (int(epochs) + 2) * eval_chunks * 8 + 64


def _mse(prediction, targets):
    return float(np.mean((prediction - targets) ** 2))


def main():
    dataset = load_fashion_mnist()
    split = fashion_arrays(
        dataset, TRAIN_PER_CLASS, TEST_PER_CLASS, train_seed=0, test_seed=1, pool=POOL
    )
    if split["x_train"].shape[1] != SIZES[0]:
        raise RuntimeError("pooled width %s does not match %s" % (split["x_train"].shape[1], SIZES))
    targets = one_hot(split["y_train"], N_CLASSES)
    test_targets = one_hot(split["y_test"], N_CLASSES)
    order = minibatch_order(split["x_train"].shape[0], seed=3)
    weights = init_weights(SIZES, seed=2, scale=1.0 / np.sqrt(SIZES[0]))
    pc_weights = [weight.copy() for weight in weights]
    problem = Problem(
        sizes=SIZES,
        weights=pc_weights,
        x_train=split["x_train"],
        y_train=targets,
        x_held=split["x_test"],
        y_held=test_targets,
        steps=EPOCHS,
        T=T,
        eta_h=ETA_H,
        alpha=ALPHA,
        rho=RHO,
        eta_theta=ETA_THETA,
        activation=ACTIVATION,
        accumulator_batch=BATCH,
    )
    network = build_network(problem)
    cap = step_cap(split["x_train"].shape[0], split["x_test"].shape[0], EPOCHS, BATCH, T)
    pc_backend = PcalmBrian2Backend()
    pc = pc_backend.integrate(
        network,
        pcalm_minibatches(
            pc_weights,
            split["x_train"],
            targets,
            split["x_test"],
            order,
            EPOCHS,
            BATCH,
            ACTIVATION,
            T,
            ETA_H,
            ALPHA,
            RHO,
            ETA_THETA,
        ),
        cap,
    )
    ann_backend = PcalmBrian2Backend()
    ann = ann_backend.integrate(
        network,
        affine_minibatch_train(
            weights,
            split["x_train"],
            targets,
            split["x_test"],
            order,
            EPOCHS,
            ETA_THETA,
            ACTIVATION,
            BATCH,
        ),
        cap,
    )
    pc_train_mse = _mse(pc["train_prediction"], targets)
    pc_test_mse = _mse(pc["test_prediction"], test_targets)
    ann_train_mse = _mse(ann["train_prediction"], targets)
    ann_test_mse = _mse(ann["test_prediction"], test_targets)
    before_mse = _mse(pc["before"], targets)
    ann_before = predict(weights, split["x_train"], ACTIVATION)
    pc_train_acc = accuracy(pc["train_prediction"], split["y_train"])
    pc_test_acc = accuracy(pc["test_prediction"], split["y_test"])
    ann_train_acc = accuracy(ann["train_prediction"], split["y_train"])
    ann_test_acc = accuracy(ann["test_prediction"], split["y_test"])

    print("Fashion-MNIST classification on SpiNNaker2 (one-hot squared error, sizes %s)" % (SIZES,))
    print("backend: %s" % pc_backend.backend_name)
    print("pixels: 28x28 scaled to [0, 1], average-pooled by %d to 7x7" % POOL)
    print(
        "samples: %d train (%d per class), %d test (%d per class), minibatch %d, epochs %d"
        % (
            split["x_train"].shape[0],
            TRAIN_PER_CLASS,
            split["x_test"].shape[0],
            TEST_PER_CLASS,
            BATCH,
            EPOCHS,
        )
    )
    print("activation: %s    inference budget T: %d" % (ACTIVATION, T))
    print("brian2 timesteps, PC-ALM: %d" % pc_backend.timesteps)
    print("brian2 timesteps, ANN:    %d" % ann_backend.timesteps)
    print("PC-ALM affine steps: %d" % pc_backend.affine_steps)
    print("ANN affine steps:    %d" % ann_backend.affine_steps)
    print("PC-ALM train MSE:  %.6f -> %.6f" % (before_mse, pc_train_mse))
    print("ANN train MSE:     %.6f -> %.6f" % (_mse(ann_before, targets), ann_train_mse))
    print("PC-ALM test MSE:   %.6f" % pc_test_mse)
    print("ANN test MSE:      %.6f" % ann_test_mse)
    print("PC-ALM train accuracy: %.3f" % pc_train_acc)
    print("ANN train accuracy:    %.3f" % ann_train_acc)
    print("PC-ALM test accuracy:  %.3f" % pc_test_acc)
    print("ANN test accuracy:     %.3f" % ann_test_acc)

    if not pc_backend.integrated_on_brian2 or not ann_backend.integrated_on_brian2:
        return 1
    if not all(np.isfinite(value) for value in (pc_train_mse, pc_test_mse, ann_train_mse, ann_test_mse)):
        return 1
    if not (pc_train_mse < before_mse and ann_train_mse < _mse(ann_before, targets)):
        return 1
    if min(pc_train_acc, pc_test_acc, ann_train_acc, ann_test_acc) <= 0.5:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
