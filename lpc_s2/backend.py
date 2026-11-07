"""Brian2 simulation of PC-ALM affine steps on SpiNNaker2 populations.

``PcalmBrian2Backend`` is a ``Brian2Backend``. It does not open a board
connection. Each algorithm step programs signed float synapses onto a LIF
population and advances Brian2 by one millisecond. With ``alpha_decay = 0``
the backend's LIF equation is an exact accumulate

    v ← i_offset + Σ synaptic weights

which is the affine map ``offset + left @ right``. The stock backend would
quantize those weights to on-chip synapse words. PC-ALM's Lagrangian is
real-valued, so this run keeps the Brian2 floats.
"""

import logging
from dataclasses import dataclass

import brian2
import numpy as np
from spinnaker2 import brian2_sim

from lpc_s2.algorithm import schedule

# SpiNNaker2 installs a diagnostic logger. The demo should print the PC-ALM
# numbers, not every Brian2 code object.
logging.getLogger("brian2").setLevel(logging.ERROR)
logging.getLogger("spinnaker2").setLevel(logging.ERROR)


@dataclass
class RunResult:
    """Numeric outcome plus the Brian2 clock that produced it."""

    losses: list
    heldout_mse: float
    energy: float
    mean_abs_residual: float
    prediction0: float
    timesteps: int
    affine_steps: int
    backend_name: str
    brian2_version: str
    integrated_on_brian2: bool


class PcalmBrian2Backend(brian2_sim.Brian2Backend):
    """Run a PC-ALM chain through Brian2's LIF update."""

    def __init__(self):
        super().__init__()
        self.backend_name = "brian2"
        self.integrated_on_brian2 = False
        self.timesteps = 0
        self.affine_steps = 0

    def run(self, network, time_steps, **kwargs):
        """Integrate ``schedule(network.problem)`` for up to ``time_steps`` ms.

        ``time_steps`` is a cap. The run stops when the schedule returns.
        Keyword arguments are accepted so callers can pass the same ``run``
        signature as ``Brian2Backend``; weights are not quantized.
        """
        del kwargs
        problem = getattr(network, "problem", None)
        if problem is None:
            raise ValueError("network.problem is required")
        metrics = self.integrate(network, schedule(problem), time_steps)
        return RunResult(
            losses=metrics["losses"],
            heldout_mse=metrics["heldout_mse"],
            energy=metrics["energy"],
            mean_abs_residual=metrics["mean_abs_residual"],
            prediction0=metrics["prediction0"],
            timesteps=self.timesteps,
            affine_steps=self.affine_steps,
            backend_name=self.backend_name,
            brian2_version=brian2.__version__,
            integrated_on_brian2=self.integrated_on_brian2,
        )

    def integrate(self, network, gen, time_steps):
        """Run one affine-step generator on this network's LIF accumulator.

        The clock counters are reset. ``gen`` is the same kind of generator
        ``schedule`` is: each yield is ``(offset, left, right)``.
        """
        self.integrated_on_brian2 = False
        self.timesteps = 0
        self.affine_steps = 0
        max_m, max_n, max_p = network.accumulator_shape
        factors = network.get("factors")
        accumulator = network.get("accumulator")
        if factors is None or accumulator is None:
            raise ValueError("network is missing the factor or accumulator population")
        return self._integrate(
            factors,
            accumulator,
            max_m,
            max_n,
            max_p,
            gen,
            int(time_steps),
        )

    def _integrate(self, factors, accumulator, max_m, max_n, max_p, gen, time_steps):
        try:
            job = next(gen)
        except StopIteration as done:
            return done.value

        clock = brian2.Clock(dt=1 * brian2.ms)
        post = self._lif_group(accumulator, clock)
        pre = self._factor_group(factors, clock)
        synapses = self._dense_synapses(pre, post, max_m, max_n, max_p, clock)
        state = {
            "job": job,
            "inflight": None,
            "value": None,
            "error": None,
            "finished": False,
        }

        def program(spec):
            offset, left, right = spec
            left = np.asarray(left, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            rows, inner = left.shape
            width = right.shape[1]
            if right.shape[0] != inner:
                raise ValueError("affine inner dimensions do not match")
            if rows > max_m or inner > max_n or width > max_p:
                raise ValueError(
                    "affine shape (%s, %s, %s) exceeds the accumulator %s"
                    % (rows, inner, width, (max_m, max_n, max_p))
                )
            if offset is None:
                bias = np.zeros((rows, width), dtype=np.float64)
            else:
                bias = np.asarray(offset, dtype=np.float64)
                if bias.shape != (rows, width):
                    raise ValueError("offset shape %s does not match the product" % (bias.shape,))
            bias_grid = np.zeros((max_m, max_p), dtype=np.float64)
            bias_grid[:rows, :width] = bias
            post.i_offset = bias_grid.reshape(-1)
            weights = np.zeros((max_m, max_n, max_p), dtype=np.float64)
            weights[:rows, :inner, :width] = left[:, :, None] * right[None, :, :]
            synapses.w[:] = weights.reshape(-1)
            self.affine_steps += 1
            return rows, width

        def read(shape):
            rows, width = shape
            voltage = np.array(post.v, dtype=np.float64).reshape(max_m, max_p)
            return voltage[:rows, :width].copy()

        @brian2.network_operation(clock=clock, when="start")
        def tick():
            # Mark only from inside Brian2's step, so a skipped run cannot pass.
            self.integrated_on_brian2 = True
            step = int(np.round(float(clock.t / clock.dt)))
            if step == 0:
                return
            if step == 1:
                state["inflight"] = program(state["job"])
                return
            result = read(state["inflight"])
            try:
                state["job"] = gen.send(result)
            except StopIteration as done:
                state["value"] = done.value
                state["finished"] = True
                brian2.stop()
                return
            except Exception as exc:  # surfaced after Network.run returns
                state["error"] = exc
                state["finished"] = True
                brian2.stop()
                return
            state["inflight"] = program(state["job"])

        net = brian2.Network(pre, post, synapses, tick)
        net.schedule = ["start", "groups", "synapses", "thresholds", "resets", "end"]
        net.run(int(time_steps) * clock.dt, report=None)
        self.timesteps = int(np.round(float(clock.t / clock.dt)))
        if state["error"] is not None:
            raise state["error"]
        if not state["finished"]:
            raise RuntimeError(
                "PC-ALM schedule did not finish within %d Brian2 steps" % int(time_steps)
            )
        return state["value"]

    def _lif_group(self, population, clock):
        """LIF group using the Brian2 backend's equations and parameter map."""
        if population.neuron_model != "lif":
            raise ValueError("accumulator population must use the lif neuron model")
        group = brian2.NeuronGroup(
            population.size,
            brian2_sim.eqs_lif,
            threshold="v > threshold",
            reset="v = v_reset",
            method="euler",
            clock=clock,
        )
        brian2_sim.apply_neuron_param_to_lif_neuron_group(group, population.params, clock.dt)
        return group

    def _factor_group(self, population, clock):
        """Spike source for ``spike_list``, one spike per factor every step."""
        if population.neuron_model != "spike_list":
            raise ValueError("factor population must use the spike_list neuron model")
        count = population.size
        indices = np.arange(count)
        times = np.zeros(count) * clock.dt
        return brian2.SpikeGeneratorGroup(
            count,
            indices=indices,
            times=times,
            period=clock.dt,
            clock=clock,
        )

    def _dense_synapses(self, pre, post, max_m, max_n, max_p, clock):
        """One signed synapse for every (row, factor, column) of the grid.

        Connection order matches C-order of an array shaped
        ``(max_m, max_n, max_p)`` so a later ``synapses.w[:] = weights.ravel()``
        lines up with ``left[row, k] * right[k, column]``.
        """
        rows = np.repeat(np.arange(max_m), max_n * max_p)
        factors = np.tile(np.repeat(np.arange(max_n), max_p), max_m)
        columns = np.tile(np.arange(max_p), max_m * max_n)
        synapses = brian2.Synapses(pre, post, model="w : 1", on_pre="v_post += w", clock=clock)
        synapses.connect(i=factors, j=rows * max_p + columns)
        synapses.w[:] = 0.0
        return synapses
