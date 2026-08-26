# B0L v0.1 Pre-Exposure Abort and v0.2 Seed Re-freeze

**Status:** `B0L_V0_1_ABORTED_BEFORE_MODEL_PERFORMANCE_EXPOSURE__V0_2_SEED_ONLY_REFREEZE_ALLOWED`

## 1. What happened

B0L v0.1 prospectively froze a four-candidate Local-endpoint mechanism screen with seeds `20276401` and `20276402`. Mandatory Q0 ran with `optimizer_steps=0` and no screen model-performance exposure.

The Q0 semantic/data checks for all four candidates passed, including target balance, exact symbolic oracle, receiver external-token lookup at chance, Q1 obsolete-event independence, Q2 owner-local-retention geometry, Q3 receiver-cue target independence, and Q4 split-target construction.

Q0 nevertheless correctly refused training because repository seed-collision auditing found `20276401/02` in an existing B0E scientific-review artifact.

## 2. Repository inconsistency discovered by Q0

The active branch contains two conflicting B0E records:

- durable `reports/post_scope_b0e/B0E_ENDPOINT_QUALIFICATION_RESULTS_v0.1.json` identifies the actual B0E performance seeds as `20276301/02`, with source identities tied to protocol blob `ea4fca6cd873e56fe3f392b5466274143867c44f`;
- `docs/analysis/B0E_LOCAL_GLOBAL_DIAGONAL_ENDPOINT_SCIENTIFIC_REVIEW_v0.1.md` text labels its endpoint rows with `20276401/02` and cites a B0E result commit that is not present in the repository history queried during this review.

This artifact conflict is not resolved by weakening the seed-collision audit. For conservative governance, any seed named as previously exposed in a scientific-review artifact is treated as unavailable for the new exploratory screen even when another durable result indicates a different actual seed identity.

## 3. Scientific consequence

No B0L v0.1 optimizer step was authorized or executed. No Persistent-vs-Local screen accuracy was exposed for Q1-Q4. Therefore changing only the reserved screen seeds before first performance exposure does **not** constitute result-dependent seed replacement or PASS chasing.

B0L v0.1 is closed as:

`ABORTED_PRE_EXPOSURE__SEED_FRESHNESS_AMBIGUITY`

No candidate definition, architecture arm, optimization rule, screen-positive threshold, candidate ranking rule, or scientific interpretation may change in v0.2.

## 4. v0.2 repair

Repository search before v0.2 freeze found no occurrence of:

- `20276501`
- `20276502`

B0L v0.2 therefore re-freezes the identical v0.1 screen with exactly these two fresh paired development seeds.

Allowed differences from v0.1:

1. protocol/source/state version identity;
2. screen seeds `20276501/02` replacing ambiguous `20276401/02`;
3. audit/workflow engineering needed to make the zero-optimizer Q0 observable and deterministic.

No other scientific change is authorized.

## 5. Authority

- B0L v0.1 performance screen: `CLOSED / NEVER OPENED`;
- B0L v0.2 protocol freeze with seed-only repair: `AUTHORIZED`;
- B0L v0.2 implementation/Q0 after freeze: `AUTHORIZED`;
- B0L v0.2 optimizer steps: `CLOSED UNTIL Q0 PASS`;
- B1/B2/B3: `CLOSED`;
- Test: `CLOSED/0`.
