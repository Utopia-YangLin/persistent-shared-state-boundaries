# SGW-Native Scope × Availability Qualification Protocol v0.1

**Status:** PROSPECTIVELY FROZEN BEFORE SOURCE IMPLEMENTATION, MODEL FORWARD, OR OPTIMIZER STEP.  
**Prior-art gate:** `docs/analysis/SGW_NATIVE_AXIS_PRIOR_ART_AUDIT_v0.1.md` completed before this freeze.  
**Parent canonical family:** exact Task-v2.2 clean Persistent / Reset / RCL340 Local / NoComm family, unchanged.  
**Purpose:** qualify a task-level SGW-native resource distinction without reusing published/internal axis crosses and without requiring associative-memory addressability.

## 1. Scientific question

> When a low-dimensional task-set determines how later evidence should be selected, does it need persistent global availability only when that task-state originates outside the final consumer, while private task-state can remain in local recurrent state and be combined with later communicated evidence?

The task is designed around SGW-native **persistent shared context + later filtering/integration**, not late key-value table lookup.

## 2. Frozen 2×2 factors

### S — task-state scope / ownership

- `PRIVATE`: the two-bit selector/task-set is owned by final consumer p5.
- `SHARED`: the identical selector/task-set is owned by separate context source p0.

This factor changes where task-state originates in the environment. It does not change model parameters or memory architecture.

### A — temporal availability

- `CURRENT`: the selector is externally re-presented by its owner in cycles 1 and 2 while evidence arrives.
- `PAST_ONLY`: the selector is presented only in cycle 0.

Cells:

1. `PRIVATE_CURRENT`
2. `PRIVATE_PAST`
3. `SHARED_CURRENT`
4. `SHARED_PAST`

Primary mechanistic diagonal: `PRIVATE_PAST <-> SHARED_PAST`. Temporal demand is identical; only whether the retained task-state is private to the final consumer or remotely originated changes.

## 3. Frozen computation

Each example contains:

- selector `s=(s0,s1)`, each bit in `{0,1}`;
- cycle-1 evidence `u=(u0,u1)`;
- cycle-2 evidence `v=(v0,v1)`;
- target bits
  - `y0 = u0 if s0=0 else v0`
  - `y1 = u1 if s1=0 else v1`;
- four-class target `2*y0 + y1`.

Thus the selector is a compact task-set that determines **which later evidence is behaviorally relevant**. No key→slot lookup, version query, or addressable record semantics exist.

Selector and target classes are exactly cross-balanced on audit/validation batches. Non-selected evidence bits are independently sampled.

## 4. Frozen processor roles and activity

- p0: SHARED context owner;
- p1: evidence source;
- p5: final consumer/readout and PRIVATE context owner;
- p2/p3/p4: task-irrelevant fillers.

Exactly four processors are active every cycle.

### Cycle 0

- `PRIVATE`: active `{p5,p2,p3,p4}`; selector at p5.
- `SHARED`: active `{p0,p2,p3,p4}`; selector at p0.

No evidence appears at cycle 0.

### Cycle 1

Evidence `u` appears only at p1.

- `PRIVATE_*`: active `{p1,p5,p2,p3}`.
- `SHARED_CURRENT`: active `{p0,p1,p5,p2}`; selector re-presented at p0.
- `SHARED_PAST`: active `{p1,p5,p2,p3}`; p0 absent.

For `PRIVATE_CURRENT`, selector is re-presented at p5. For `PRIVATE_PAST`, no selector is externally re-presented.

### Cycle 2

Same active geometry as cycle 1. Evidence `v` appears only at p1. CURRENT cells re-present selector at its owner; PAST_ONLY cells do not.

### Cycle 3

All cells use active `{p5,p2,p3,p4}`. No selector or evidence is externally presented. This is a common final decision/readout cycle.

## 5. Single architecture-neutral ownership firewall

Same-cycle communication at cycle 0 could otherwise copy the selector into proxy processor recurrent states and destroy the task-state ownership manipulation.

Therefore, immediately **after cycle-0 communication**, identically for all four architectures:

- `PRIVATE`: retain only p5 h/c; zero p0–p4 h/c exactly;
- `SHARED`: retain only p0 h/c; zero p1–p5 h/c exactly;
- Persistent workspace state is preserved exactly;
- Reset workspace follows its canonical reset-each-cycle equation on the next cycle;
- Local and NoComm have no shared workspace;
- no later recurrent-state firewall is applied.

This firewall enforces environmental ownership; it does not alter any communication operator, workspace equation, readout, parameter, or optimizer.

## 6. Frozen 46D deterministic token interface

No trainable task encoder.

- selector-present flag: 1D;
- selector bit0 one-hot: 2D;
- selector bit1 one-hot: 2D;
- evidence-present flag: 1D;
- evidence bit0 one-hot: 2D;
- evidence bit1 one-hot: 2D;
- evidence-phase one-hot (`NONE`,`U`,`V`): 3D;
- cycle one-hot: 4D;
- processor-id one-hot: 6D;
- role one-hot (`context`,`evidence`,`consumer`,`filler`): 4D;
- exact-zero padding: 19D.

Total: 46D.

No token directly contains the target class. p5 never externally receives `u` or `v`; p1 never externally receives selector bits. In SHARED cells p5 never externally receives selector bits.

## 7. Mandatory Q0 — zero optimizer steps

Before training, all four cells must pass:

1. `[B,4,6,46]` and exactly four active processors/cycle;
2. correct cell-specific active sets;
3. selector appears only at the frozen owner and at exactly the availability-defined cycles;
4. evidence `u/v` appears only at p1 in cycles 1/2;
5. cycle 3 contains no task-state/evidence values;
6. exact selector×target 4×4 cross-balance on the audit batch;
7. symbolic solver accuracy = 1.0;
8. p5 external-token-only target lookup = 0.25 exactly on the audit batch;
9. value-scrambled symbolic accuracy in `[0.23,0.27]`;
10. exact-zero padding;
11. ownership firewall leaves exactly the frozen owner h/c and zeros every non-owner h/c after cycle0 in finite smoke forwards for every architecture;
12. wrapper and canonical family have identical parameter names/shapes/initial tensors and no new trainable parameters;
13. canonical clean-family source Git blob remains unchanged;
14. qualification seeds are collision-free outside this new instrument;
15. prior-art audit file is present and unchanged;
16. Test remains `CLOSED/0`.

Q0 failure => no optimizer step 1. Only semantics-identical engineering repair is allowed. Any task-semantic change requires a separately versioned instrument and a renewed prior-art audit.

## 8. Frozen small qualification

Fresh paired development seeds exactly:

- `20275801`
- `20275802`

`4 cells × 2 seeds × 4 canonical architectures = 32 scientific units / 8 paired groups`.

Training recipe unchanged from recent small qualification instruments:

- Adam `lr=1e-3`, `weight_decay=0`;
- batch 64;
- 2500 optimizer steps;
- validation every 250 steps, deterministic `N=4096`;
- gradient clip 1.0;
- minimum validation cross-entropy checkpoint after full exposure;
- deterministic arm64 torch2.2.2, one torch thread;
- at most four independent group workers;
- paired data and paired dropout RNG streams across architectures;
- no restart, replacement seed, best-of-R, budget extension, scheduler/optimizer change, architecture patch, or cell-specific tuning.

## 9. Predeclared qualification gates

Two seeds are instrument qualification only, not publication inference.

### Gate N — `PRIVATE_CURRENT`: common communication sanity

For both seeds individually:

- Persistent >= 0.85;
- Reset >= 0.85;
- Local >= 0.85;
- NoComm <= 0.35.

Purpose: current private selector + remote evidence is learnable by all communicating systems.

### Gate L — `PRIVATE_PAST`: local recurrent sufficiency

For both seeds individually:

- Local >= 0.90;
- NoComm <= 0.35.

No winner condition is imposed on Persistent or Reset. The gate asks only whether task-state that is privately owned by p5 can legally remain in local recurrence and later combine with communicated evidence.

### Gate C — `SHARED_CURRENT`: communication control without persistence necessity

For both seeds individually:

- Persistent >= 0.85;
- Reset >= 0.85;
- Local >= 0.85;
- NoComm <= 0.35;
- `abs(Persistent - Reset) <= 0.10`.

Purpose: when remote task-state is currently available, communication should suffice and persistent workspace should not be uniquely necessary.

### Gate P — `SHARED_PAST`: persistent shared-context positive control

For both seeds individually:

- Persistent >= 0.80;
- `Persistent - Reset >= +0.15`;
- `Persistent - Local >= +0.15`;
- NoComm <= 0.35.

This is the decisive SGW-native persistence gate: the task-set originates remotely, disappears before later evidence arrives, and must remain globally available to determine later evidence relevance.

### Overall

`PASS = N && L && C && P`.

A PASS establishes a **mechanistic scope boundary**, not a general performance crossover and not permission to claim Local is globally better in PRIVATE cells.

## 10. Governance

- Q0 fail => no training.
- Qualification scientific FAIL => stop; no post-exposure semantic patch and no larger map.
- PASS => stop for human scientific review; no automatic atlas.
- Any future boundary map requires fresh seeds and a separately frozen protocol.
- Rejected prior-art-overlapping axes may not be substituted after performance exposure.
- Canonical architectures remain unchanged.
- Test remains `CLOSED/0`.
