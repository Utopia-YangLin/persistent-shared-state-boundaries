# B0L Local Endpoint Mechanism Screen Protocol v0.2

**Status:** `PROSPECTIVELY_FROZEN__SEED_ONLY_REPAIR__BEFORE_V0_2_MODEL_FORWARD_OR_OPTIMIZER_STEP`

**Parent v0.1 protocol:** `docs/contracts/B0L_LOCAL_ENDPOINT_MECHANISM_SCREEN_PROTOCOL_v0.1.md`, blob `7575b336a0091edb33a1c0073c6f068c6d97fb35`.

**Pre-exposure repair record:** `docs/analysis/B0L_V0_1_PRE_EXPOSURE_ABORT_AND_V0_2_SEED_REFREEZE.md`.

**Purpose:** execute exactly the finite Local-side mechanism screen frozen in v0.1 after conservatively replacing an ambiguous seed pair discovered by mandatory zero-optimizer Q0.

## 1. Inheritance rule

Except where explicitly overridden in this document, **every scientific clause of B0L v0.1 is inherited unchanged and binding**, including:

- scientific question;
- four-candidate library and exact candidate semantics;
- fixed CGAO computational primitive;
- canonical architecture family;
- two screen arms only (`sgw_persistent_innovation_clean`, `rcl340_local_clean`);
- 46D common-host requirements;
- exactly one extreme corner per candidate;
- execution matrix shape;
- optimization recipe and checkpoint selection;
- mandatory Q0 semantic/data checks;
- estimand `Delta_LP = Local - Persistent`;
- screen-positive thresholds;
- multi-positive ranking rule;
- fresh confirmatory endpoint requirement before any map;
- B1/B2/B3 closure;
- Test `CLOSED/0`.

No candidate or gate is added, deleted, weakened, strengthened, or reinterpreted by v0.2.

## 2. Exact candidate library — unchanged

Fixed order remains exactly:

1. `Q1_OBSOLETE_WRITE_TURNOVER`
2. `Q2_SEGREGATED_LOCAL_RETENTION`
3. `Q3_RECEIVER_LOCAL_ROLE_BINDING`
4. `Q4_TWO_EVENT_SPLIT_CGAO`

Their task geometries are exactly those frozen in v0.1 Sections 4–5.

## 3. Seed-only override

v0.1 reserved `20276401/02`, but mandatory Q0 found those identifiers already present in a pre-existing B0E scientific-review artifact. No B0L performance model forward or optimizer step had occurred.

v0.2 replaces only that pair with fresh paired development seeds exactly:

- `20276501`
- `20276502`

Repository search immediately before this freeze returned no occurrence for either seed.

No replacement, best-of-R, candidate-specific seed, or further seed change is allowed after v0.2 performance exposure begins.

## 4. Execution matrix — unchanged except seed identity

`4 candidates × 2 seeds × 2 architectures = 16 scientific units`

= `8 paired candidate/seed groups`.

At most four paired groups execute concurrently. Paired examples, ordering, initialization policy, and stochastic streams follow v0.1 exactly.

## 5. Optimization — unchanged

Exactly:

- Adam `lr=1e-3`, `weight_decay=0`;
- batch `64`;
- `2500` optimizer steps per unit;
- validation at `0,250,...,2500`;
- deterministic validation `N=4096`;
- gradient clip `1.0`;
- selected checkpoint = minimum validation cross entropy after full exposure, earliest exact tie;
- deterministic arm64 torch `2.2.2` semantics;
- one torch thread per group worker;
- no scheduler, LR/optimizer change, curriculum, auxiliary loss, budget extension, restart for scientific outcome, result-dependent stopping, or architecture patch.

## 6. Mandatory v0.2 Q0

Before optimizer step 1, v0.2 must independently reproduce all v0.1 common and candidate-specific Q0 requirements using the v0.2 source identity and fresh seeds.

In addition, Q0 must verify:

1. the v0.2 task module is a semantics-preserving wrapper over frozen v0.1 task blob `9edf2089ad99246c6e1f5153d3f920c4898efdd7`;
2. the only intended task-module scientific difference is `SCREEN_SEEDS=(20276501,20276502)`;
3. neither new seed occurs outside the v0.2 B0L instrument/support artifacts at Q0 time;
4. optimizer steps remain `0`;
5. Test remains `CLOSED/0`.

Any semantic/data Q0 failure closes v0.2. Semantics-identical audit/workflow engineering repairs may rerun Q0 before performance exposure.

## 7. Frozen screen-positive rule — unchanged

For candidate `q` and each seed `r`:

`Delta_LP(q,r) = joint_accuracy(Local;q,r) - joint_accuracy(Persistent;q,r)`.

A candidate is `SCREEN_POSITIVE` only if:

for **each seed individually**:

- Local joint accuracy `>= 0.90`;
- `Delta_LP >= +0.02`;

and across both seeds:

- mean `Delta_LP >= +0.04`.

A tie, a one-seed effect, or joint failure of both architectures is not positive.

## 8. Frozen multi-positive selection rule — unchanged

If multiple candidates are positive, rank lexicographically by:

1. larger `min_seed_Delta_LP`;
2. larger `mean_Delta_LP`;
3. larger `mean_Local_accuracy`;
4. fixed candidate order `Q1 < Q2 < Q3 < Q4` only for an exact numerical tie.

Select exactly one candidate for fresh confirmatory-design review.

## 9. What the screen can authorize

A positive screen is developmental construct discovery only. It authorizes **design of** a fresh confirmatory endpoint protocol containing the selected Local candidate, retained Global endpoint, Persistent, Local, Reset, NoComm, and new confirmatory development seeds.

It does not itself authorize endpoint confirmation, B1/B2/B3, continuous mapping, architecture changes, or Test.

If no candidate is positive, the signed Local↔Global route closes for this programme and the scientifically supported fallback is the roadmap-native tie/local-sufficient ↔ Global-advantage onset boundary.

## 10. Governance

- v0.1: `ABORTED_PRE_EXPOSURE__SEED_FRESHNESS_AMBIGUITY`;
- v0.2 candidate semantics/gates/budget: `FROZEN`;
- v0.2 model/optimizer performance exposure: `CLOSED UNTIL Q0 PASS`;
- B1/B2/B3: `CLOSED`;
- Test: `CLOSED/0`.
