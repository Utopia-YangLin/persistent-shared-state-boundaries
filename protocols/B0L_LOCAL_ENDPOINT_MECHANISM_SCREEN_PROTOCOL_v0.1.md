# B0L Local Endpoint Mechanism Screen Protocol v0.1

**Status:** `PROSPECTIVELY_FROZEN__BEFORE_B0L_IMPLEMENTATION_MODEL_FORWARD_OR_OPTIMIZER_STEP`

**Parent scientific review:** `docs/analysis/B0E_LOCAL_GLOBAL_ENDPOINT_SCIENTIFIC_REVIEW_v0.1.md`

**Candidate design:** `docs/analysis/B0L_LOCAL_ENDPOINT_MECHANISM_SCREEN_DESIGN_v0.1.md`

**Purpose:** efficiently screen a finite, literature/architecture-grounded library of replacement Local-side pressures while retaining the already qualified Global persistence mechanism.

This is a **developmental mechanism screen**, not endpoint confirmation and not a boundary scan.

## 1. Scientific question

> Among four prospectively frozen environmental pressures, does any one produce a reproducible Local-over-Persistent directional signal large enough to justify a fresh confirmatory endpoint qualification?

No candidate may be added, modified, dropped, or rerun for scientific outcome after B0L performance exposure begins.

## 2. Fixed architecture family

Screen only the two primary architectures:

1. `sgw_persistent_innovation_clean` — Persistent-SGW / Global primary;
2. `rcl340_local_clean` — Local primary.

Canonical architecture source remains:

`src/workspace_broadcast_boundaries/architectures/task_v2_2_clean_family_v0_1.py`

No architecture equation, hidden width, workspace size, communication width, readout, input-attention equation, optimizer, or checkpoint rule may change.

Reset-SGW and NoComm are deliberately omitted from this exploratory screen to spend the same scientific-unit budget across more mechanistic hypotheses. Any selected candidate must later be re-qualified on **fresh seeds with Reset and NoComm restored** before it may become a map endpoint.

## 3. Fixed computational primitive

Every candidate uses frozen CGAO as its relation primitive:

for two-bit `c,a,b`, independently per bit `j`:

- if `cj=0`: `yj = aj AND bj`;
- if `cj=1`: `yj = aj OR bj`.

No candidate introduces a different relation rule.

Candidates may change only temporal/source/binding geometry as prospectively specified below.

## 4. Frozen candidate library

Exactly four candidates, in this fixed ID order:

1. `Q1_OBSOLETE_WRITE_TURNOVER`
2. `Q2_SEGREGATED_LOCAL_RETENTION`
3. `Q3_RECEIVER_LOCAL_ROLE_BINDING`
4. `Q4_TWO_EVENT_SPLIT_CGAO`

### Q1 — obsolete-write turnover

- cycles 0 and 1 contain independent target-like CGAO triples that are explicitly obsolete;
- cycle 2 contains the sole target-defining fully current CGAO triple;
- cycle 3 is blind decision;
- target = cycle-2 CGAO only;
- obsolete triples are independently sampled and exactly target-independent.

No remote/past information is required for the target.

### Q2 — segregated owner-local retention

- cycle 0 contains target `c,a,b` at three distinct persistent owners;
- p5 is absent during early storage;
- cycles 1 and 2 keep the three owners active under task-neutral continuation with no target refresh;
- cycle 3 activates the three owners plus p5 for late integration/readout;
- target = CGAO of the original cycle-0 `c,a,b`;
- no cycle-0 ownership firewall is applied because legal owner-local recurrent retention is the estimand.

### Q3 — receiver-local role binding

- a current integration event presents three unlabeled 2-bit symbols at p0,p1,p2;
- a uniformly balanced six-way permutation determines which physical source is semantic `(context,A,B)`;
- p5 receives only that permutation cue;
- source tokens do not reveal semantic roles beyond physical source identity;
- target = CGAO under the p5-supplied role permutation;
- p5 cue alone must remain exactly target-independent under the audit distribution.

No delayed persistence is required.

### Q4 — two-event split CGAO

- cycle 0 contains an independent fully current CGAO event with proposal `u0`;
- cycle 1 is target-neutral hold/continuation;
- cycle 2 contains an independent fully current CGAO event with proposal `u2`;
- cycle 3 is blind decision;
- final target = `2*high_bit(u0) + low_bit(u2)`;
- p5 is a receiver/readout but not a direct task-value source at either writing event;
- no shared-only persistence firewall is imposed.

This candidate intentionally recovers the temporal partial-answer structure that has the strongest historical receiver-blind Local-positive prior.

## 5. Common 46D host contract

All candidates must produce `[B,4,6,46]` float32 tokens for the unchanged canonical host.

Candidate-specific field semantics are allowed only inside a single prospectively frozen B0L task module and must satisfy:

- no trainable task encoder;
- exact-zero unused dimensions;
- processor identity and cycle identity may be encoded;
- any role/permutation cue is architecture-neutral;
- no target class, architecture ID, future event, model output, or outcome-derived field is encoded;
- exactly four processors are active each cycle;
- both architectures receive byte/tensor-identical paired examples for a seed/candidate.

## 6. Fresh screen seeds

Exactly:

- `20276401`
- `20276402`

Repository search before freeze found no occurrence of either seed.

The same two paired development seeds are used for every candidate.

No replacement seeds, best-of-R, retries for scientific outcome, or candidate-specific seeds are allowed.

## 7. Execution matrix

`4 candidates × 2 seeds × 2 architectures = 16 scientific units`

= `8 paired candidate/seed groups`.

At most four paired groups may execute concurrently. Within a paired group, the two architectures use paired batch identities/order and paired initial stochastic streams under the established deterministic arm64 execution semantics.

## 8. Frozen optimization

Use the B0E/Phase-A small-study recipe unchanged:

- Adam `lr=1e-3`, `weight_decay=0`;
- batch size `64`;
- exactly `2500` optimizer steps per scientific unit;
- validation at step `0` and every `250` steps through `2500`;
- deterministic validation `N=4096`;
- gradient clip `1.0`;
- selected checkpoint = minimum validation cross entropy after full exposure, earliest exact tie;
- deterministic arm64 torch `2.2.2` semantics;
- one torch thread per group worker;
- no scheduler, LR change, optimizer change, curriculum, auxiliary loss, budget extension, candidate-specific training recipe, seed replacement, result-dependent stopping, or architecture patch.

## 9. Mandatory Q0 before optimizer step 1

A separate static/data audit must pass all of the following with zero optimizer steps:

### Common

1. tensor geometry exactly `[B,4,6,46]`;
2. exactly four active processors every cycle;
3. exact four-class target balance on deterministic audit batches;
4. frozen CGAO oracle reproduces targets exactly;
5. p5 external-token-only lookup/Bayes accuracy is `0.25` on the exact-balance audit distribution;
6. no token contains target, architecture ID, future-event values, or outcome-derived metadata;
7. exact-zero unused padding;
8. both primary architectures retain canonical parameter names/shapes/initial tensors;
9. canonical architecture source identity is unchanged;
10. screen seeds are collision-free outside B0L;
11. Test remains `CLOSED/0`.

### Q1-specific

12. obsolete cycle-0/1 triples are independent of cycle-2 target and each other under deterministic generation;
13. target is a function only of the cycle-2 current triple;
14. all target-relevant information is current by cycle 2.

### Q2-specific

15. p5 is absent during the early storage cycles specified by the task;
16. the original three target symbols are present only at their owners at cycle 0 and are not externally refreshed;
17. owner recurrent paths remain legal; no ownership firewall or architecture-specific carry operation is introduced;
18. cycle-3 late integration exposes no target value directly to p5.

### Q3-specific

19. all six role permutations are exactly balanced and target independent;
20. p5 cue alone has target lookup/Bayes accuracy `0.25`;
21. source-token fields do not leak semantic role assignment;
22. applying the cue permutation followed by frozen CGAO reproduces the target exactly.

### Q4-specific

23. cycle-0 and cycle-2 CGAO events are independently sampled and individually balanced;
24. final target is exactly `2*high_bit(u0)+low_bit(u2)`;
25. p5 is not a direct source of `c/a/b` at either event;
26. no special shared-only persistence operation is introduced.

Any semantic Q0 failure closes B0L v0.1. Only a semantics-identical engineering repair may rerun Q0.

## 10. Frozen screen estimand

For candidate `q` and seed `r`:

`Delta_LP(q,r) = joint_accuracy(Local;q,r) - joint_accuracy(Persistent;q,r)`.

Report for each candidate:

- both seed accuracies for Local and Persistent;
- both seed `Delta_LP` values;
- mean `Delta_LP`;
- minimum seedwise `Delta_LP`;
- mean selected-checkpoint cross entropy as descriptive information;
- selected step and training trajectory for both arms.

No significance test is used to turn this two-seed screen into confirmatory evidence.

## 11. Predeclared screen-positive rule

A candidate is `SCREEN_POSITIVE` only if all conditions hold:

for **each seed individually**:

- Local joint accuracy `>= 0.90`;
- `Delta_LP >= +0.02`;

and across both seeds:

- mean `Delta_LP >= +0.04`.

These thresholds are discovery filters, not manuscript effect thresholds.

A candidate that merely produces `Local ≈ Persistent`, or makes both models fail, is not positive.

## 12. Predeclared selection rule if multiple candidates are positive

If exactly one candidate is positive, select it for confirmatory-design review.

If multiple candidates are positive, rank them lexicographically by:

1. larger `min_seed_Delta_LP`;
2. larger `mean_Delta_LP`;
3. larger `mean_Local_accuracy`;
4. fixed candidate ID order `Q1 < Q2 < Q3 < Q4` only for an exact numerical tie.

Select exactly one candidate.

The non-selected positive candidates remain exploratory observations only and are not combined into a higher-dimensional map.

## 13. Decision after the screen

### No positive candidate

Status:

`B0L_SCREEN_NEGATIVE__SIGNED_LOCAL_GLOBAL_MAP_ROUTE_CLOSED`

Then:

- do not invent additional candidates in v0.1;
- do not tune the four candidates;
- return to the already supported `tie/local-sufficient ↔ Global-advantage` onset-boundary programme or manuscript synthesis.

### One selected positive candidate

Status:

`B0L_SCREEN_POSITIVE__FRESH_CONFIRMATORY_ENDPOINT_PROTOCOL_REQUIRED`

This authorizes only scientific design/review of a **new fresh-seed confirmatory endpoint qualification** containing:

- selected Local candidate;
- retained Global endpoint;
- Persistent-SGW;
- Local;
- Reset-SGW;
- NoComm;
- fresh confirmatory development seeds;
- preregistered dominance gates.

It does **not** authorize B1/B2/B3 or allow the screen seeds to serve as confirmatory evidence.

## 14. Governance

- this protocol freezes all four candidates before B0L performance exposure;
- no post-exposure candidate rescue;
- no architecture tuning;
- no Test access;
- B1/B2/B3 remain closed regardless of screen outcome until separate confirmatory endpoint qualification passes.
