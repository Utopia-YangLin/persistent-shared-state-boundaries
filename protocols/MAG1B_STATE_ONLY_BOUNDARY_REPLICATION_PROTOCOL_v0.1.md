# MAG1b State-Only Boundary Replication Protocol v0.1

**Status:** PROSPECTIVELY FROZEN BEFORE MAG1b OPTIMIZER STEP OR MAG1b TRAINING RUNNER IMPLEMENTATION.

**Authority:** MAG1a scientific review `docs/analysis/MAG1A_MEMORY_ACCESS_GEOMETRY_SCIENTIFIC_REVIEW_v0.1.md` at commit `fef276544a0d68936ca8efd7528ee23cbc4cbe7c`.

## 1. Question

> Does the qualified past-only scope signature remain learnable from scratch when previous processor hidden state is prevented from influencing input-attention selection, while ordinary recurrent h/c carry is preserved?

This is a nuisance-robustness test of the dual-access concern, not a new scientific axis.

## 2. Frozen geometries

Use the already frozen MAG intervention semantics:

- `CANONICAL = Q1_R1`: input attention receives real previous `h`; recurrent LSTM receives real previous `h,c`.
- `STATE_ONLY = Q0_R1`: input attention receives exact-zero hidden reference; recurrent LSTM receives real previous `h,c`.

No other geometry is trained in MAG1b. Persistent workspace, Reset semantics, Local communication, active masks, task tokens, cycle-0 ownership firewall, and readout are unchanged.

## 3. Frozen task scope

Use PX1 binding `B0` only. PX1 already established that the past-only signature transports across X=p5,p1,p3 physical realizations, so MAG1b does not repeat the role-binding factor.

Run only:

- `PRIVATE_PAST`;
- `SHARED_PAST`.

Task computation and token generator are exactly PX1 v0.1.

## 4. Fresh paired seeds

Exactly:

- `20276001`;
- `20276002`.

No replacement, restart, extension, best-of-R, or seed substitution.

## 5. Matrix

`2 geometries × 2 cells × 2 seeds × 4 architectures = 32 scientific units / 8 paired geometry-cell-seed groups`.

Architectures remain:

- Persistent Innovation-SGW;
- Reset Innovation-SGW;
- RCL340 Local;
- NoComm.

Within each geometry/cell/seed group, data and dropout streams are paired across architectures.

## 6. Training recipe

Identical to PX1 qualification:

- Adam lr `1e-3`, weight decay 0;
- batch 64;
- 2500 optimizer steps;
- validation every 250 steps;
- deterministic validation `N=4096`;
- gradient clip 1.0;
- checkpoint selected by minimum validation CE after full exposure;
- deterministic arm64, torch 2.2.2, one torch thread;
- at most four group workers.

No hyperparameter tuning is authorized.

## 7. Q0 gate

Before optimizer step 1 verify:

1. protocol/source authority and Test `CLOSED/0`;
2. task batch/oracle/anti-leakage checks inherited from PX1;
3. `CANONICAL` MAG wrapper logits equal PX1 wrapper logits at identical initialization for a finite smoke batch;
4. `STATE_ONLY` differs only by the input-attention hidden reference being exact zero;
5. recurrent LSTM receives real h/c in both geometries;
6. wrapper parameter names/shapes/initialized tensors equal canonical clean model exactly;
7. no new trainable parameter;
8. fresh seed namespace collision-free.

Q0 failure => zero training.

## 8. Predeclared boundary gates

For every geometry and both seeds individually:

### PRIVATE_PAST sufficiency

- Persistent >= 0.90;
- Reset >= 0.90;
- Local >= 0.90;
- NoComm <= 0.35.

### SHARED_PAST persistent-global signature

- Persistent >= 0.80;
- Persistent - Reset >= 0.15;
- Persistent - Local >= 0.15;
- NoComm <= 0.35.

### State-only preservation

For both seeds:

- `Local_PRIVATE_STATE_ONLY >= 0.90`;
- `Persistent_SHARED_STATE_ONLY >= 0.80`.

The primary MAG1b gate is:

`MAG1B_PASS = CANONICAL signature passes && STATE_ONLY signature passes`.

No requirement is imposed that STATE_ONLY exactly match CANONICAL accuracy; preservation of the qualitative architecture-ordering signature is the estimand.

## 9. Interpretation

PASS supports:

> the current scope×availability positive diagonal is not contingent on the canonical hidden-to-input-attention query path; it remains learnable when processor memory is accessed only through recurrent h/c carry while persistent workspace remains available.

FAIL means the parent result remains valid for the canonical host but the task boundary is host-memory-access-geometry dependent; continuous boundary mapping is blocked until that dependence is understood.

## 10. Stop rule

Stop after the 32-unit MAG1b qualification for scientific review. No Q-only retraining, no Q/R ratio sweep, no continuous task boundary scan, no architecture transportability expansion, and no Test access is automatically authorized by this protocol.
