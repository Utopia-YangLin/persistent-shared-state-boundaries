# PX1 Consumer Role-Binding Invariance Protocol v0.1

**Status:** PROSPECTIVELY FROZEN BEFORE SOURCE IMPLEMENTATION, MODEL FORWARD, OR OPTIMIZER STEP.

**Parent evidence:** Scope × Availability qualification PASS and causal decomposition remain immutable. PX0 established that physical p5 should be treated as an instantiation of abstract consumer role X.

## 1. Question

> Does the past-only scope signature depend on the abstract context–consumer geometry, or on physical processor 5 specifically?

PX1 is a nuisance-robustness qualification, not a new task axis and not publication-level inference.

## 2. Frozen task cells

Run only the two informative past-only endpoints:

- `PRIVATE_PAST`: `C=X`; selector appears only at cycle 0 in X;
- `SHARED_PAST`: `C!=X`; selector appears only at cycle 0 in remote context owner C.

The selector/evidence/target computation is unchanged from `SGW_NATIVE_SCOPE_AVAILABILITY_QUALIFICATION_PROTOCOL_v0.1`:

- selector `s=(s0,s1)`;
- cycle-1 evidence `u=(u0,u1)`;
- cycle-2 evidence `v=(v0,v1)`;
- `y_j = u_j if s_j=0 else v_j`;
- four-class target `2*y0+y1`.

No associative-memory semantics are introduced.

## 3. Abstract roles and three frozen physical bindings

Roles:

- `C`: remote context owner used only in SHARED;
- `E`: evidence producer;
- `X`: designated consumer/readout;
- `F0,F1,F2`: fillers.

Three full permutations are frozen:

- `B0`: `C=p0, E=p1, X=p5, F0=p2, F1=p3, F2=p4` — historical physical realization;
- `B1`: `C=p2, E=p3, X=p1, F0=p4, F1=p5, F2=p0`;
- `B2`: `C=p4, E=p5, X=p3, F0=p0, F1=p1, F2=p2`.

Within every binding, C/E/X are physically distinct. Across bindings, X occupies p5, p1, and p3; C/E/filler roles are permuted with it rather than holding a privileged physical pathway fixed.

## 4. Frozen active geometry in role coordinates

### PRIVATE_PAST (`C=X` semantically)

The remote-C physical processor is a spare/non-context processor in this cell.

- cycle 0: `{X,F0,F1,F2}`; selector only at X;
- cycle 1: `{E,X,F0,F1}`; u only at E;
- cycle 2: `{E,X,F0,F1}`; v only at E;
- cycle 3: `{X,F0,F1,F2}`; no selector/evidence externally present.

### SHARED_PAST (`C!=X`)

- cycle 0: `{C,F0,F1,F2}`; selector only at C;
- cycle 1: `{E,X,F0,F1}`; u only at E; C absent;
- cycle 2: `{E,X,F0,F1}`; v only at E; C absent;
- cycle 3: `{X,F0,F1,F2}`; no selector/evidence externally present.

Exactly four processors are active each cycle.

## 5. Ownership firewall

Immediately after cycle-0 communication, identically for all architectures:

- PRIVATE_PAST: retain only X processor h/c; zero all other processor h/c;
- SHARED_PAST: retain only C processor h/c; zero all other processor h/c;
- Persistent workspace state is preserved;
- Reset follows canonical reset-each-cycle semantics;
- no later recurrent-state firewall is applied.

This is the same abstract ownership rule as the parent instrument, rebound through rho.

## 6. Readout abstraction

Use exactly the same shared readout module as the canonical blind-receiver head, but apply it to the abstract consumer state:

`logits = readout(h[:, rho(X)])`.

No physical-identity-specific readout modules, additional parameters, per-binding heads, or parameter remapping are allowed.

Physical processor-id one-hot remains the true physical index. Role one-hot is generated from the abstract role binding. This intentionally permits the model to represent both physical identity and task role while testing whether the qualitative architecture signature transports across bindings.

## 7. Q0 static/identity gate

Before optimizer step 1 verify for all 6 cell×binding combinations:

1. token shape `[B,4,6,46]` and exactly four active processors/cycle;
2. role-specific active sets equal the frozen schedule;
3. selector appears only at X (PRIVATE) or C (SHARED), cycle0 only;
4. evidence appears only at E in cycles1/2;
5. cycle3 has no task content;
6. exact selector×target cross-balance;
7. symbolic oracle = 1.0;
8. X external-token-only target lookup = 0.25 exactly;
9. value-scrambled oracle accuracy in `[0.23,0.27]`;
10. ownership firewall retains exactly the abstract owner;
11. wrapper has no trainable parameter absent from the canonical clean model;
12. initialized parameter tensors are exactly identical to canonical model for each architecture/seed;
13. readout module object/state is identical across bindings and only its selected processor input changes;
14. canonical clean-family source blob unchanged;
15. fresh PX1 seed namespace collision-free;
16. Test remains `CLOSED/0`.

Q0 failure => no training.

## 8. Frozen small qualification

Fresh paired seeds exactly:

- `20275901`
- `20275902`

Matrix:

`2 cells × 3 bindings × 2 seeds × 4 architectures = 48 scientific units / 12 paired groups`.

Training recipe is identical to the parent small qualification:

- Adam lr `1e-3`, weight decay 0;
- batch 64;
- 2500 optimizer steps;
- validation every 250 steps;
- deterministic validation `N=4096`;
- gradient clip 1.0;
- minimum validation CE checkpoint after full exposure;
- deterministic arm64 torch2.2.2, one torch thread;
- at most four independent group workers;
- paired data/dropout streams across architectures;
- no restart, replacement, best-of-R, extension, tuning or architecture patch.

## 9. Predeclared signature gates

For **every binding and both seeds individually**:

### PRIVATE_PAST sufficiency

- Persistent >= 0.90;
- Reset >= 0.90;
- Local >= 0.90;
- NoComm <= 0.35.

No winner requirement among communicating models.

### SHARED_PAST persistent-global signature

- Persistent >= 0.80;
- `Persistent - Reset >= 0.15`;
- `Persistent - Local >= 0.15`;
- NoComm <= 0.35.

### Overall

`PX1_PASS = all PRIVATE_PAST sufficiency checks && all SHARED_PAST signature checks`.

A failure at one physical binding is retained as evidence against full role-binding invariance; it is not permission to replace that binding or seed.

## 10. Interpretation

PASS supports:

> the qualified past-only scope signature transports across several physical realizations of consumer role X and is not obviously specific to processor 5.

PASS does not prove full permutation invariance over all 6! role assignments.

FAIL narrows the parent result and blocks larger Consumer Geometry expansion until the source of physical-identity dependence is understood.

Stop after the 48-unit development qualification for human review. No consumer uncertainty, multiplicity, continuous boundary scan, or Test access is automatically authorized.
