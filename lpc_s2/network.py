"""SpiNNaker2 populations for one PC-ALM chain.

The chain populations and projections are the model: input, hidden layers,
multipliers, the linear readout, and the initial weight matrices. The factor
and accumulator populations are the Brian2 machine that evaluates each
affine step. Data and hyperparameters live on ``network.problem`` because
LIF parameters cannot store a training batch.
"""

from dataclasses import dataclass

import numpy as np
from spinnaker2 import snn


LIF_PARAMS = {
    "threshold": 1.0e12,
    "alpha_decay": 0.0,
    "i_offset": 0.0,
    "v_reset": 0.0,
    "reset": "reset_to_v_reset",
}


@dataclass
class Problem:
    """Identity-chain regression and the Algorithm 1 hyperparameters."""

    sizes: tuple
    weights: list
    x_train: np.ndarray
    y_train: np.ndarray
    x_held: np.ndarray
    y_held: np.ndarray
    steps: int
    T: int
    eta_h: float
    alpha: float
    rho: float
    eta_theta: float
    activation: str = "identity"


def accumulator_limits(sizes, batch):
    """Grid large enough for every matmul and axpy in a chain of these sizes."""
    width = max(int(size) for size in sizes)
    batch = int(batch)
    return max(batch, width), max(batch, width), width


def _lif(size, name):
    return snn.Population(
        size=int(size),
        neuron_model="lif",
        params=dict(LIF_PARAMS),
        name=name,
        record=[],
    )


def _connections(weight):
    """Connection list for W shaped (post, pre)."""
    connections = []
    fan_out, fan_in = weight.shape
    for pre in range(fan_in):
        for post in range(fan_out):
            connections.append([pre, post, float(weight[post, pre]), 0])
    return connections


def build_network(problem):
    """Chain network plus the spike-list / LIF pair Brian2 integrates."""
    sizes = tuple(int(size) for size in problem.sizes)
    if len(sizes) < 3:
        raise ValueError("sizes must include input, one hidden layer, and output")
    batch = max(problem.x_train.shape[0], problem.x_held.shape[0])
    max_m, max_n, max_p = accumulator_limits(sizes, batch)
    network = snn.Network("pcalm")
    layers = [_lif(sizes[0], "input")]
    multipliers = []
    for index, width in enumerate(sizes[1:-1], start=1):
        layers.append(_lif(width, "hidden_%d" % index))
        multipliers.append(_lif(width, "multiplier_%d" % index))
    layers.append(_lif(sizes[-1], "readout"))
    projections = []
    for index, weight in enumerate(problem.weights, start=1):
        projections.append(
            snn.Projection(
                pre=layers[index - 1],
                post=layers[index],
                connections=_connections(weight),
                name="W_%d" % index,
            )
        )
    factors = snn.Population(
        size=max_n,
        neuron_model="spike_list",
        params={neuron: [0] for neuron in range(max_n)},
        name="factors",
    )
    accumulator = _lif(max_m * max_p, "accumulator")
    network.add(*layers, *multipliers, *projections, factors, accumulator)
    network.problem = problem
    network.accumulator_shape = (max_m, max_n, max_p)
    network.validate()
    return network
