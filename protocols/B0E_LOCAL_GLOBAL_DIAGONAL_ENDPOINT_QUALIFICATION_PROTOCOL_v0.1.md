# B0E Local–Global Diagonal Endpoint Qualification Protocol v0.1

**Status:** `PROSPECTIVELY_FROZEN__BEFORE_B0E_SOURCE_IMPLEMENTATION_MODEL_FORWARD_OR_OPTIMIZER_STEP`

**Purpose:** admit or reject a compact signed Local↔Global regime map before any boundary scan.

**Parent endpoint audit:** `docs/analysis/B0_LOCAL_GLOBAL_ENDPOINT_CANDIDATE_AUDIT_v0.1.md` (commit `f6bcec0ad417a2cac9302eaaee03f27518912c8e`)

**Unified task design:** `docs/analysis/B0_UNIFIED_LOCAL_GLOBAL_COMPARATIVE_TASK_DESIGN_v0.1.md` (commit `b4f20cdf3c5b11627b0428a195c8aae0fdecdb84`)

**Canonical architecture source:** `src/workspace_broadcast_boundaries/architectures/task_v2_2_clean_family_v0_1.py`, blob `1d192f8786132d48188125b9bc371598ae60c497`.

**Frozen CGAO rule source:** `src/workspace_broadcast_boundaries/task/rules_v2_2.py`, blob `84574651c2d94a6d6520df67793b74f87cf16839`.

This protocol authorizes no Test access. Scientific task or gate changes after any B0E model forward require a new version and may not be used to rescue this qualification.

## 1. Scientific question

> Can one common relation-computation family produce two prospectively defined, opposite architecture-dominance endpoints: a remote-persistence regime strongly favoring Persistent-SGW and a current distributed-composition regime favoring Local?

B0E is an **endpoint admission test**, not the boundary map itself.

If either endpoint fails its frozen gate, the signed Local↔Global map is not admitted. No B1/B2/B3 performance scan follows from a B0E FAIL.

## 2. Fixed target computation

For every example and both endpoints, sample three two-bit variables:

- `c=(c0,c1)` — context;
- `a=(a0,a1)` — operand A;
- `b=(b0,b1)` — operand B.

For each bit `j`:

- `yj = aj AND bj` when `cj=0`;
- `yj = aj OR bj` when `cj=1`.

Target class:

`y = 2*y0 + y1`.

This is exactly the frozen Task-v2.2 CGAO relation. No endpoint-specific target rule exists.

## 3. Exact B0E endpoints

Only two diagonal endpoints are exposed in B0E.

### G — Global/Persistent endpoint

Coordinate identity: `(P=1, D=0)` = `REMOTE_PAST_ONLY × CONCENTRATED_OPERANDS`.

- cycle 0: `c` at remote context owner `p0` only;
- no later context refresh;
- `p0` absent at integration;
- cycle 2: both `a` and `b` at evidence source `p1`;
- final consumer `p5` has no external task values;
- cycle-0 ownership firewall prevents proxy local copies while preserving Persistent workspace memory.

Intended pressure: remotely originated context must persist across time; contemporaneous operand composition is low-distribution.

### L — Local endpoint

Coordinate identity: `(P=0, D=1)` = `CURRENT × DISTRIBUTED_OPERANDS`.

- cycle 0: `c` at `p0`;
- cycle 2: the same `c` is externally refreshed at `p0`;
- cycle 2: `a` at `p1`, `b` at `p2`;
- `p0,p1,p2,p5` are simultaneously active at integration;
- final consumer `p5` has no external task values.

Intended pressure: no shared cross-cycle persistence is required at integration, while relation-sensitive contemporaneous information is distributed across distinct sender identities.

## 4. Four-cycle active geometry

Exactly four processors are active every cycle.

### Cycle 0 — context

Active: `{p0,p2,p3,p4}`.

Task content: `c` at `p0` only.

Immediately after cycle-0 communication, identically across all architectures:

- retain only `p0` recurrent h/c;
- zero p1–p5 h/c exactly;
- preserve Persistent workspace memory;
- Reset follows its canonical reset-each-cycle workspace equation;
- Local/NoComm have no shared workspace.

### Cycle 1 — delay

Active: `{p5,p2,p3,p4}`.

No `c/a/b` task content.

### Cycle 2 — integration

G active: `{p1,p2,p3,p5}`.

- `p1`: both `a,b`;
- `p2,p3`: task-content-free fillers;
- no `c`; `p0` absent.

L active: `{p0,p1,p2,p5}`.

- `p0`: refreshed `c`;
- `p1`: `a`;
- `p2`: `b`;
- `p5`: no task content.

### Cycle 3 — decision

Active: `{p5,p2,p3,p4}`.

No `c/a/b` task content. Blind four-class readout from p5.

## 5. Frozen 46D deterministic token layout

No trainable task encoder.

Per processor/cycle token:

- context-present flag: 1D;
- context bit0 one-hot: 2D;
- context bit1 one-hot: 2D;
- operand-A-present flag: 1D;
- operand A bit0 one-hot: 2D;
- operand A bit1 one-hot: 2D;
- operand-B-present flag: 1D;
- operand B bit0 one-hot: 2D;
- operand B bit1 one-hot: 2D;
- cycle one-hot: 4D;
- processor-id one-hot: 6D;
- fixed role one-hot (`context`,`operand_a`,`operand_b`,`consumer`,`filler`): 5D;
- exact-zero padding: 16D.

Total = `46D`.

Processor role identity is fixed by physical role and is not a target label. `p2` retains its fixed operand-B role identity even when it is a task-content-free filler at G; presence flags distinguish content from absence.

No token includes the target class, P/D coordinate label, architecture identity, future information, or model output.

## 6. Frozen data balance

Training batch size is 64.

The generator must maintain exact four-class target balance in every training and validation batch: 16 examples per target class.

Within each endpoint, `(c,a,b)` values must be sampled independently of architecture. The same paired examples and ordering are used across the four architectures for a seed.

The implementation may use deterministic stratified sampling or exact truth-table-derived sampling, but Q0 must prove:

- target balance exact;
- the frozen CGAO oracle is 1.0;
- a target-scrambled / value-scrambled control is near chance;
- geometry assignment contains no target leakage.

No endpoint-specific value distribution is permitted except the frozen placement/availability geometry.

## 7. Information-theoretic anchor

For the frozen uniform CGAO relation, the Task-v2.2 static audit establishes that observing any pair of the three input symbols yields Bayes-optimal target accuracy `0.5625`.

At G, after the cycle-0 firewall and with p0 absent later, a system that has no legal retained access to `c` but observes `a,b` should therefore be bounded by the missing-context regime rather than full-information performance.

Q0 must independently exhaustively reproduce the `0.5625` pair-source Bayes ceiling from the frozen CGAO truth table. This number is a task property, not a performance gate fitted to B0E outcomes.

## 8. Architectures

Exactly:

1. `sgw_persistent_innovation_clean` — Persistent-SGW / Global primary;
2. `rcl340_local_clean` — Local primary;
3. `sgw_reset_innovation_clean` — nonpersistent Global control;
4. `nocomm_clean` — negative communication control.

No architecture source edit is authorized.

## 9. Fresh seeds and execution matrix

Fresh paired development seeds exactly:

- `20276301`
- `20276302`

Repository search before freeze found no collision for either seed.

Matrix:

`2 endpoints × 2 seeds × 4 architectures = 16 scientific units / 4 paired groups`.

No seed replacement, best-of-R, restarts for scientific outcome, or additional endpoint is allowed.

## 10. Frozen optimization

Use the same small-qualification recipe as the parent SGW-native endpoint programme:

- Adam `lr=1e-3`, `weight_decay=0`;
- batch size 64;
- exactly 2500 optimizer steps per scientific unit;
- validation at step 0 and every 250 steps through 2500;
- deterministic validation `N=4096`;
- gradient clip 1.0;
- checkpoint = minimum validation cross-entropy after full exposure, earliest exact tie;
- deterministic arm64 torch 2.2.2 semantics;
- one torch thread per group worker;
- at most four independent paired groups concurrently;
- paired batch identities/order and paired dropout RNG streams across architectures;
- no scheduler, optimizer change, LR change, curriculum, auxiliary loss, budget extension, cell-specific tuning, architecture patch, seed replacement, or result-dependent stopping.

## 11. Mandatory Q0 — zero optimizer steps

Before optimizer step 1, Q0 must pass all checks:

1. exact tensor geometry `[B,4,6,46]` and four active processors/cycle;
2. exact G/L active sets from this protocol;
3. exact task-content placement and absence by cycle/processor;
4. cycle-0 firewall zeros p1–p5 recurrent h/c and retains only p0 h/c after communication for all four architectures;
5. Persistent workspace is not erased by the firewall;
6. L refreshes full c at p0 in cycle2; G has no c after cycle0 and p0 is absent at cycle2/3;
7. G co-locates both operands at p1 while keeping total unique task values unchanged;
8. L places a at p1 and b at p2;
9. target class balance is exactly uniform;
10. frozen CGAO oracle accuracy = 1.0;
11. exhaustive pair-source Bayes ceiling = 0.5625;
12. p5 external-token-only best lookup = 0.25 exactly on an exact-balance audit batch;
13. value-scrambled oracle is in `[0.23,0.27]` on a sufficiently large deterministic audit;
14. zero padding is exact zero;
15. canonical clean-family parameter names/shapes/initial tensors are unchanged and no trainable task encoder exists;
16. architecture source blob remains `1d192f8786132d48188125b9bc371598ae60c497`;
17. frozen CGAO rule blob remains `84574651c2d94a6d6520df67793b74f87cf16839`;
18. B0E seeds are collision-free outside this instrument;
19. Test remains `CLOSED/0`.

Any Q0 scientific/semantic failure => `B0E_Q0_FAIL__NO_TRAINING`. Only a semantics-identical engineering repair may rerun Q0. Any task semantic change requires v0.2 before performance exposure.

## 12. Predeclared endpoint gates

All accuracies below are selected-checkpoint deterministic validation joint accuracies.

### Gate G — strong Global dominance

For **each seed individually**:

- Persistent-SGW `>= 0.90`;
- Persistent-SGW minus Local `>= +0.25`;
- Persistent-SGW minus Reset-SGW `>= +0.25`;
- Local `<= 0.65`;
- Reset-SGW `<= 0.65`;
- NoComm `<= 0.35`.

This is intentionally stronger than merely detecting a positive difference.

### Gate L — Local-dominant endpoint admission

For **each seed individually**:

- Local `>= 0.95`;
- Local minus Persistent-SGW `>= +0.03`;
- NoComm `<= 0.35`.

Across the two seeds:

- mean(Local minus Persistent-SGW) `>= +0.05`;
- Persistent-SGW may reach `>=0.95` on at most `1/2` seeds.

Reset-SGW is reported as a mechanistic/global-control outcome but is not required to lose to Local for Gate L; the primary signed map is Persistent-SGW vs Local.

### Overall B0E admission

`B0E_PASS = Gate_G(seed1) && Gate_G(seed2) && Gate_L(seed1) && Gate_L(seed2) && Gate_L_aggregate`.

A FAIL is scientifically informative and terminates this comparative map route. It does not authorize a different threshold, new seed, endpoint tweak, architecture patch, or training extension.

## 13. What a PASS authorizes

A B0E PASS authorizes only human scientific review of whether to freeze a compact P×D boundary-scan protocol.

It does **not** automatically authorize:

- a 25-cell grid;
- B1 edge performance scans;
- B2 interior points;
- B3 adaptive matrix;
- Test access;
- CTM or another architecture family;
- architecture changes.

The intended later coordinate vocabulary is `P,D in {0,.25,.5,.75,1}`, but map point selection must be separately frozen after B0E review and before any map performance exposure.

## 14. Test and governance

Test remains `CLOSED` with model-forward count `0`.

B0E uses fresh development-only data. No Test-derived statistic may inform endpoint admission, map-point selection, thresholds, seeds, architecture changes, or manuscript claims.
