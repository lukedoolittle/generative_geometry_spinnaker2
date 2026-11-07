# Lagrangian predictive coding on SpiNNaker2

PC-ALM (augmented Lagrangian predictive coding) from Seely and Gould,
[arXiv:2605.31022](https://arxiv.org/pdf/2605.31022), defined with
[py-spinnaker2](https://gitlab.com/spinnaker2/py-spinnaker2) populations and
integrated by the Brian2 simulation backend. Training and inference use the
same identity-chain regression as the Python reference: sizes `(2, 8, 1)`,
identity activations, a frozen teacher, and one student trained for 40 steps
with inference budget `T = 20`. The dual rate is `alpha = 2`, so that within
that budget PC-ALM finishes below a backprop ANN trained for the same 40
steps from the same initialization.

## Backend

No SpiNNaker2 board is required. `PcalmBrian2Backend` subclasses
`spinnaker2.brian2_sim.Brian2Backend` and never opens a chip connection.

The stock Brian2 backend quantizes synaptic weights to on-chip synapse words.
PC-ALM's augmented Lagrangian is real-valued, so this run keeps float64
weights and still uses that backend's LIF equation. With `alpha_decay = 0`
one Brian2 millisecond is the accumulate

```text
v ← i_offset + Σ synaptic weights
```

which is the affine step `offset + left @ right`. Every primal update, dual
ascent, and weight step is one of those accumulates. The demo prints
`backend: brian2`.

## Install

Python 3.12 (py-spinnaker2 requires 3.10 or newer). From the repository root:

```bash
./scripts/install.sh
source .venv/bin/activate
```

The script creates `.venv`, clones py-spinnaker2, checks out the `snn-core`
commit its installer expects, and installs Brian2 with it. macOS `/bin/bash`
3.2 cannot run upstream `setup_libs.sh` (it uses a bash-4 array subscript),
which is why the checkout is pinned here first.

## Run

Training and inference are one command, from the repository root with the
virtualenv active:

```bash
python -m lpc_s2
```

That Brian2 run trains on 64 synthetic samples, then runs inference on 16
held-out samples. It then trains a backprop ANN on that same batch, from the
same initial weights, for the same 40 steps and the same step size, with no
biases. The ANN's matmuls and weight updates are the same LIF accumulator
steps as PC-ALM. It prints both training and held-out forward MSEs, the
augmented Lagrangian, the mean absolute constraint residual, and the first
inferred prediction against its target.

Inference does not change weights. Each iteration is a forward sweep from the
input toward the readout. In that sweep, layer i writes its activity on the
LIF accumulator and then updates its multiplier, `λ ← λ + α r`, before the
sweep moves on. The first sweep is the feasible initialization (the residual
is zero, so λ stays zero). The next `T - 1` sweeps are primal-dual. The last
sweep is a primal step only. Each training step does that inference and then
the weight step `W ← W − η_θ ∇_W L_ρ`.

## Tests

```bash
pytest
```

## Equation map

Logged training loss is the feasible readout. It does not carry multipliers.
During inference the quantities below are Brian2 affine steps inside the
forward sweep, except the weight step, which runs once the sweeps return.

| Quantity | Equation | Where it runs |
| --- | --- | --- |
| forward activity | h_i = σ(W_i h_{i-1}), then ŷ = W_L h_{L-1} | forward sweep, input toward the readout |
| residual | r_i = h_i − σ(W_i h_{i-1}) | that same layer, after h_i is written |
| combined signal | c_i = λ_i + ρ r_i | activity gradient for the sweep |
| dual step | λ_i ← λ_i + α r_i | that same layer, before the next layer |
| primal step | h_i ← h_i − η_h ∇_{h_i} L_ρ | primal sweeps; gradients from the state at the start of the sweep |
| weight step | W ← W − η_θ ∇_W L_ρ | after inference, using the λ the sweeps wrote |

`alpha = 0` freezes every multiplier at zero. The initialization sweep
applies the dual step to a zero residual, and a primal-dual sweep then adds
α r = 0. The demo uses `alpha = 2`.

## SpiNNaker2 / Brian2 pieces and the Python LPC ops

The chain (`input`, `hidden_1`, `multiplier_1`, `readout`, projections `W_1`
and `W_2`) is the model. `factors` is a `spike_list` population: one spike
per column of a product, every millisecond. `accumulator` is a `lif`
population whose voltage is the product. A dense signed synapse from factor
`k` into accumulator neuron `(row, column)` carries
`left[row, k] * right[k, column]`.

| SpiNNaker2 / Brian2 piece | Python LPC operation |
| --- | --- |
| `lif` equation in `brian2_sim`, `alpha_decay = 0` | one affine step `v ← i_offset + Σ w` |
| `i_offset` | bias term loaded for that step (activity, multiplier, or zero) |
| `spike_list` factors, one spike per step | columns of `left @ right` |
| signed float synapse `v += w` | one summand of a matmul or an axpy |
| primal step | `h ← h − η_h ∇_h L_ρ` inside the forward sweep, using `c = λ + ρ r` |
| dual step | `λ ← λ + α r` in that same layer, with `r = h − σ(W h_prev)` |
| weight step | `∇_W L_ρ` from `c = λ + ρ r`, then `W ← W − η_θ ∇_W L_ρ` |
| `PcalmBrian2Backend.run` | Algorithm 1: forward sweeps, then learn; Brian2 `Network.run` is the clock |

Hidden-layer weight updates use `c = λ + ρ r`. The readout is linear and is
not a constraint. For the identity demo, `σ` is the identity, so every state
update is a Brian2 LIF step. `tanh` and `relu` apply `σ` and `σ′` to the
Brian2 pre-activations before the next accumulate.

## Processing-element sketch

`firmware/pcalm_core.c` is a thought experiment, not a binary loaded onto a
chip. It follows the stock `lif_neuron` split. `packet_received_callback`
adds one term of `W h_prev`. `timer_callback` drains those packets and only
then writes `h` and applies `λ ← λ + α r`. A partial dot product does not
update `λ`.

```bash
cc -DPCALM_HOST_CHECK -O2 -o /tmp/pcalm_core firmware/pcalm_core.c -lm
/tmp/pcalm_core
```
