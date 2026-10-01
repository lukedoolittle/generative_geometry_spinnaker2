# Lagrangian predictive coding on SpiNNaker2

PC-ALM (augmented Lagrangian predictive coding) from Seely and Gould,
[arXiv:2605.31022](https://arxiv.org/pdf/2605.31022), defined with
[py-spinnaker2](https://gitlab.com/spinnaker2/py-spinnaker2) populations and
integrated by the Brian2 simulation backend. Training and inference use the
same identity-chain regression as the Python reference: sizes `(2, 8, 1)`,
identity activations, a frozen teacher, and one student trained for 40 steps
with inference budget `T = 20`.

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
held-out samples. It prints forward MSE before and after training, held-out
forward MSE, the augmented Lagrangian, the mean absolute constraint residual,
and the first inferred prediction against its target.

Inference does not change weights. It resets multipliers to zero, initializes
activities on the forward pass, runs `T - 1` primal-dual cycles
(`h ← h − η_h ∇_h L_ρ`, then `λ ← λ + α r`), and finishes with one more
primal step. Each training step does that inference and then the weight step
`W ← W − η_θ ∇_W L_ρ`.

## Tests

```bash
pytest
```

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
| primal step | `h ← h − η_h ∇_h L_ρ` |
| dual step | `λ ← λ + α r`, with `r = h − σ(W h_prev)` |
| weight step | `∇_W L_ρ` from `c = λ + ρ r`, then `W ← W − η_θ ∇_W L_ρ` |
| `PcalmBrian2Backend.run` | Algorithm 1: infer, then learn; Brian2 `Network.run` is the clock |

Hidden-layer weight updates use `c = λ + ρ r`. The readout is linear and is
not a constraint. `alpha = 0` would freeze the multipliers at zero; the demo
uses `alpha = 1`. For the identity demo, `σ` is the identity, so every state
update is a Brian2 LIF step. `tanh` and `relu` apply `σ` and `σ′` to the
Brian2 pre-activations before the next accumulate.
