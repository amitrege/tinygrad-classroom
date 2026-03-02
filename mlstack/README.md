# mlstack: How to Read This Folder

If you are browsing this repo on GitHub and mostly want to understand the code, this file is your guide.

`mlstack/` is the core teaching code behind the demo app. `app.py` imports these modules directly and uses them in different lesson stages.

## Big Picture Mental Model

Think of this folder as a small learning stack that grows from simple to more realistic:

1. Make small synthetic data.
2. Train one neuron manually.
3. Build tiny autograd primitives.
4. Build layers + optimizers on top of autograd.
5. Train tiny models end-to-end.
6. Check gradients and visualize behavior.

Text map:

`datasets.py` -> `manual_neuron.py` -> `autograd.py` -> `nn.py` -> `train.py`  
`gradcheck.py` and `visuals.py` help validate and explain what is happening.

## File-by-File Map

### `datasets.py`
- What it does: creates the toy datasets used in lessons (`linearly separable`, `two moons`).
- Why it matters: gives controlled data so model behavior is easy to reason about.
- Read first: `make_linearly_separable`, `make_two_moons`.
- Seen in app stages: early demos + end-to-end comparisons.

### `manual_neuron.py`
- What it does: manual logistic neuron (forward pass + hand-written gradients + training loop).
- Why it matters: this is the baseline before autograd.
- Read first: `sigmoid`, `manual_gradients`, `train_single_neuron`.
- Seen in app stages: Stage 1 and Stage 2.

### `autograd.py`
- What it does: tiny reverse-mode autodiff engine (`Tensor` + operator overloads + backward pass).
- Why it matters: this is the core idea behind modern DL frameworks.
- Read first: `Tensor`, `_unbroadcast`, operator methods (`__add__`, `__mul__`, `__matmul__`), `backward`.
- Seen in app stages: Stage 3 and Stage 4, then reused later.

### `nn.py`
- What it does: minimal neural-net building blocks on top of `Tensor` (`Linear`, `ReLU`, `Sequential`) plus optimizers (`SGD`, `Adam`) and BCE helpers.
- Why it matters: turns autograd primitives into trainable models.
- Read first: `Module` interface, `Linear`, `Sequential`, `SGD.step`, `Adam.step`.
- Seen in app stages: used in later training comparisons and tiny framework demos.

### `train.py`
- What it does: higher-level training utilities for binary MLP experiments.
- Why it matters: connects model + optimizer + metrics into one loop.
- Read first: `build_mlp`, `train_binary_mlp`, `predict_logits`.
- Seen in app stages: end-to-end model comparison stages.

### `gradcheck.py`
- What it does: finite-difference gradient checks vs autograd gradients.
- Why it matters: catches silent math bugs in backward logic.
- Read first: `check_linear_layer_grad`.
- Seen in app stages: Stage 4 validation/debugging.

### `vectorization.py`
- What it does: loop vs vectorized forward pass + simple benchmark.
- Why it matters: helps the scalar-to-matrix transition click.
- Read first: `loop_forward`, `vectorized_forward`, `benchmark_forward`.
- Seen in app stages: Stage 6.

### `visuals.py`
- What it does: plotting utilities for curves and decision boundaries.
- Why it matters: makes model behavior visible, not just numeric.
- Read first: `grid_for_points`, `plot_decision_surface`, `plot_curve`.
- Seen in app stages: used across many interactive visual tabs.

## Reading Paths (Pick One)

### Path A: New to this topic (intuition first)
1. `datasets.py`
2. `manual_neuron.py`
3. `vectorization.py`
4. `autograd.py`
5. `nn.py`
6. `train.py`
7. `gradcheck.py`

### Path B: I only care about autograd internals
1. `autograd.py`
2. `gradcheck.py`
3. `nn.py` (to see how autograd gets used in modules)

### Path C: I want end-to-end training quickly
1. `nn.py`
2. `train.py`
3. `datasets.py`
4. `visuals.py`

## How to Build a Mental Model While Reading

Use this checklist:

1. Track shapes at every step (`(B, D)`, `(D, H)`, `(B, 1)`).
2. Separate logits vs probabilities in your head.
3. In backward logic, always ask: “where is this gradient accumulated?”
4. When confused, jump to tests and verify expected behavior.

Useful cross-check files:

- `tests/test_autograd.py`
- `tests/test_gradcheck.py`

## Common Confusion Points

- Broadcasting in backward pass: forward can broadcast silently, backward cannot.
- `zero_grad()` timing: gradients should be reset before each optimizer step.
- BCE terms: logits and sigmoid probabilities are not interchangeable.
- Scalar-to-batch jump: same math, just with matrix shapes.

## 10-Minute Quick Skim

If you only have a few minutes:

1. `manual_neuron.py` (`manual_gradients`, `train_single_neuron`)
2. `autograd.py` (`Tensor`, `backward`, `_unbroadcast`)
3. `nn.py` (`Linear`, `SGD`/`Adam`)
4. `train.py` (`train_binary_mlp`)
5. `gradcheck.py` (`check_linear_layer_grad`)
