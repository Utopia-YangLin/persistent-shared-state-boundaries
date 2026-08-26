# SGW-Native Causal Decomposition Protocol v0.1

**Status:** PROSPECTIVELY FROZEN BEFORE ANY NEW INTERVENTION FORWARD. CHECKPOINT-ONLY; ZERO OPTIMIZER STEPS.  
**Parent positive result:** `agent/sgw-native-scope-availability-v0.1` terminal qualification commit `4ef59d67a4aca4fc35c0a9b562d5080c885ad9d1`, workflow run `32852451741`, status `QUALIFICATION_PASS__HUMAN_REVIEW_REQUIRED`.  
**Purpose:** causally decompose the qualified `Task-state scope × Temporal availability` effect into workspace persistence, workspace-to-processor broadcast, their interaction, and receiver breadth without changing task or architecture.

## 1. Governance and scope

Use only already-selected checkpoints from qualification run `32852451741` and the exact deterministic validation generator (`tag="qualification:validation"`, `N=4096`) for seeds exactly `20275801, 20275802`.

No training, gradient, optimizer, learned probe, checkpoint reselection, new seed, architecture parameter edit, task semantic edit, threshold tuning, or Test access is permitted.

The interventions are forward-only causal ablations of retained trained solutions. They do not define candidate architectures and cannot be promoted automatically.

Test remains `CLOSED/0`.

## 2. Frozen checkpoint families

### Main persistence/broadcast interaction

- cell: `SHARED_PAST`
- architecture: `sgw_persistent_innovation_clean`
- seeds: `20275801, 20275802`

### Persistence-specificity controls

Use the same Persistent architecture/checkpoint type in:

- `PRIVATE_PAST`
- `SHARED_CURRENT`
- `SHARED_PAST`

### Broadcast-only/current-context control

- cell: `SHARED_CURRENT`
- architecture: `sgw_reset_innovation_clean`
- seeds: both frozen seeds

The qualification result must be replayed exactly before interventions are accepted.

## 3. Exact host invariants

All forwards preserve the frozen SGW-native task generator and cycle-0 ownership firewall:

- immediately after cycle-0 communication, only the context owner's processor h/c is retained;
- all non-owner processor h/c is zeroed;
- Persistent workspace memory survives the firewall;
- no later recurrent-state firewall exists;
- canonical input attention, recurrent LSTMs, workspace write/update equations, readout, active masks, and token stream are unchanged.

Interventions may alter only the two causal channels explicitly defined below.

## 4. Factor R — workspace persistence / reverberation

For a Persistent checkpoint:

- `R_ON`: canonical Persistent semantics (`reset_each_cycle=False`) on all cycles.
- `R_OFF`: cycle 0 remains canonical; on cycles 1,2,3 the workspace call uses reset semantics so the previous cross-cycle workspace state is replaced by canonical initial memory before the normal current-cycle write/update/broadcast.

Thus `R_OFF` removes cross-cycle workspace persistence while preserving identical trained weights, fresh write source, current-cycle workspace computation, and broadcast machinery.

## 5. Factor B — intermediate workspace broadcast

`B` refers only to workspace-to-processor messages during evidence-processing cycles 1 and 2.

- `B_ON`: canonical workspace `received` is added to processor hbar.
- `B_OFF`: the workspace is still written/updated normally and its memory state is preserved according to R, but `received` is replaced by exact zero for all processors on cycles 1 and 2 before `h = hbar + received`.

For the primary `R × B` factorial, cycle-3 final broadcast is always ON. Cycle-0 broadcast is left canonical; the ownership firewall immediately erases it from every non-owner processor, while the owner is later absent in the decisive `SHARED_PAST` cell.

## 6. Primary 2×2 causal factorial on SHARED_PAST Persistent checkpoints

Evaluate:

- `R1_B1`: canonical persistence + intermediate broadcast;
- `R1_B0`: persistence ON + intermediate broadcast OFF;
- `R0_B1`: persistence OFF + intermediate broadcast ON;
- `R0_B0`: persistence OFF + intermediate broadcast OFF.

Cycle-3 final broadcast remains ON for all four.

For each seed report accuracy and CE plus:

- persistence effect when B on: `PE_B1 = Y11 - Y01`;
- persistence effect when B off: `PE_B0 = Y10 - Y00`;
- intermediate-broadcast effect when R on: `BE_R1 = Y11 - Y10`;
- intermediate-broadcast effect when R off: `BE_R0 = Y01 - Y00`;
- interaction: `I_RxB = (Y11 - Y01) - (Y10 - Y00)` (equivalently `BE_R1 - BE_R0`).

Predeclared descriptive flags, requiring the condition on both seeds:

- `PERSISTENCE_CAUSAL_MAIN`: `PE_B1 >= 0.15`.
- `INTERMEDIATE_BROADCAST_CAUSAL_MAIN`: `BE_R1 >= 0.10`.
- `PERSISTENCE_X_BROADCAST_SYNERGY`: `I_RxB >= 0.10`.

Failure of a flag is a mechanistic result, not permission to change the intervention.

## 7. Persistence-specificity profile

For Persistent checkpoints in `PRIVATE_PAST`, `SHARED_CURRENT`, and `SHARED_PAST`, compare canonical baseline with `R_OFF`, with all broadcasts canonical.

Report per-seed drops.

Predeclared flag:

`PERSISTENCE_SCOPE_SPECIFIC` is true only if on both seeds:

- `SHARED_PAST` drop >= 0.15;
- `PRIVATE_PAST` drop <= 0.10;
- `SHARED_CURRENT` drop <= 0.10.

This asks whether persistence is causally necessary specifically when task state is both remotely originated and past-only.

## 8. Broadcast timing decomposition

Evaluate both:

1. `SHARED_PAST` Persistent checkpoints with R_ON;
2. `SHARED_CURRENT` Reset checkpoints (nonpersistent broadcast-only/current-context control).

Broadcast timing conditions after cycle 0:

- `ALL`: cycles 1,2,3 canonical broadcast;
- `INTERMEDIATE_ONLY`: cycles 1,2 canonical; cycle 3 received is zero for all processors;
- `FINAL_ONLY`: cycles 1,2 received is zero; cycle 3 canonical;
- `NONE`: cycles 1,2,3 received is zero.

Workspace write/update continues normally in every condition.

Report:

- intermediate contribution with final present: `ALL - FINAL_ONLY`;
- final contribution with intermediate present: `ALL - INTERMEDIATE_ONLY`;
- total post-cycle0 broadcast contribution: `ALL - NONE`.

Flags (both seeds):

- `CURRENT_CONTEXT_BROADCAST_CAUSAL`: on `SHARED_CURRENT` Reset, `ALL - NONE >= 0.15`.
- `PAST_CONTEXT_BROADCAST_CAUSAL`: on `SHARED_PAST` Persistent, `ALL - NONE >= 0.15`.
- `PAST_DYNAMIC_INTERMEDIATE_BROADCAST`: on `SHARED_PAST` Persistent, `ALL - FINAL_ONLY >= 0.10`.

If only final broadcast is necessary, describe the learned solution as persistent shared-state storage with final read access rather than strong evidence for recurrent dynamic broadcast.

## 9. Intermediate receiver-breadth decomposition

Run on the same two control families above, with their canonical persistence semantics and cycle-3 final broadcast ON. On cycles 1 and 2, compute canonical broadcast but retain messages only for the frozen receiver set:

- `ALL_SIX`: p0..p5;
- `TASK_RELEVANT`: `{p1, p5}` (evidence specialist + final consumer);
- `EVIDENCE_ONLY`: `{p1}`;
- `CONSUMER_ONLY`: `{p5}`;
- `NONE`: no intermediate receiver.

The workspace memory update is identical across conditions.

Report per-seed accuracies and contrasts:

- evidence-specialist value: `TASK_RELEVANT - CONSUMER_ONLY`;
- consumer value: `TASK_RELEVANT - EVIDENCE_ONLY`;
- irrelevant/global-extra receiver value: `ALL_SIX - TASK_RELEVANT`.

Flags (both seeds):

- `EVIDENCE_SPECIALIST_BROADCAST_VALUE`: `TASK_RELEVANT - CONSUMER_ONLY >= 0.10`.
- `CONSUMER_INTERMEDIATE_BROADCAST_VALUE`: `TASK_RELEVANT - EVIDENCE_ONLY >= 0.10`.
- `EXTRA_GLOBAL_RECEIVER_VALUE`: `ALL_SIX - TASK_RELEVANT >= 0.05`.

These flags characterize the learned routing need; no flag is required for the parent positive result to remain valid.

## 10. Mandatory replay / engineering gate

Before reporting any intervention:

1. all required checkpoint files from run `32852451741` must exist;
2. checkpoint metadata must match cell, architecture, seed and selected validation values in the durable qualification report;
3. exact baseline replay on `qualification:validation` must match the stored selected checkpoint accuracy to numerical tolerance `<=1e-7`;
4. all model parameters remain unchanged across every intervention forward;
5. zero optimizer steps and zero learned probes;
6. Test remains `CLOSED/0`.

Any failure is an engineering stop and not a mechanistic result.

## 11. Interpretation boundaries

This development-checkpoint audit can establish causal dependence of the trained solutions, not population-level confirmation. It does not authorize a new task, architecture repair, larger atlas, or publication-level inferential claim.

The scientifically relevant decomposition is:

- **Persistence**: cross-cycle workspace state itself is necessary;
- **Broadcast**: workspace-to-processor availability is necessary even without a persistence requirement;
- **Joint mechanism**: persistence gains behavioral value specifically through intermediate broadcast/re-entry (`R×B` positive interaction);
- **Receiver breadth**: whether the learned solution needs evidence-specialist access, consumer access, or broader global availability.

Stop after durable report for human scientific review.
