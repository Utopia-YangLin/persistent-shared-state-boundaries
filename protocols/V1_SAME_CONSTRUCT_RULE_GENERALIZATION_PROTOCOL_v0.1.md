# V1 Same-Construct Rule Generalization Protocol v0.1

**Status:** PROSPECTIVELY FROZEN BEFORE V1 TASK SOURCE, MODEL FORWARD, OR OPTIMIZER STEP.

**Parent authority:** MAG1b development PASS, workflow `32869511711`, durable result commit `ff19f374220ee0a4906d451c6a2294588ede07a9`, scientific review `docs/analysis/MAG1B_STATE_ONLY_BOUNDARY_SCIENTIFIC_REVIEW_v0.1.md`.

## 1. Question

> Does the past-only scope signature generalize to a second selector-to-evidence binding rule that preserves the same task-state ownership, temporal availability, information amount, no-selector Bayes difficulty, processor roles, and architecture family?

This is a small construct-generalization qualification, not a new main axis and not a continuous-boundary study.

## 2. Clean host

Use the prospectively selected cleaned boundary host:

- processor memory geometry `STATE_ONLY = Q0_R1`;
- previous processor `h` is NOT provided to input-attention selection (exact-zero query reference);
- ordinary recurrent `h/c` carry remains ON;
- Persistent/Reset workspace semantics remain unchanged;
- Local/NoComm semantics remain unchanged;
- no trainable parameter is added or removed.

The historical canonical Q1_R1 evidence remains immutable and is not reclassified.

## 3. Frozen task geometry

Use PX1 physical binding `B0` only:

- `C=p0`, `E=p1`, `X=p5`, fillers p2-p4;
- `PRIVATE_PAST`: selector cycle0 at X, retain only X processor h/c after cycle0;
- `SHARED_PAST`: selector cycle0 at C, retain only C processor h/c after cycle0;
- cycle1 evidence vector `u=(u0,u1)` only at E;
- cycle2 evidence vector `v=(v0,v1)` only at E;
- cycle3 no task content;
- exactly four active processors per cycle;
- blind shared readout from abstract consumer X;
- no selector refresh after cycle0.

## 4. Two frozen rule families

Let `s=(s0,s1)`.

### R0 AXIS_ALIGNED — parent rule

`y0 = u0 if s0=0 else v0`  
`y1 = u1 if s1=0 else v1`

### R1 CROSS_V — phase-specific binding generalization

`y0 = u0 if s0=0 else v1`  
`y1 = u1 if s1=0 else v0`

Thus both rules retain the same semantics "task-state selects which temporally separated evidence is behaviorally relevant". R1 changes which V-component is bound to each selector-controlled output component without changing evidence dimensionality, timing, owner/consumer scope, or active geometry.

R1 is not a global target-label permutation of R0 across all selector/evidence states. It changes the selector-conditioned binding relation specifically for the V branch.

## 5. Difficulty/information matching gate

Before training, exhaustive symbolic enumeration over all 4 selectors × 4 u vectors × 4 v vectors must verify for BOTH rules:

1. targets are exactly balanced over four classes;
2. selector×target table is exactly balanced under the generated qualification stream;
3. symbolic oracle accuracy = 1.0;
4. when u and v are observed but selector is hidden, exact Bayes joint accuracy = `0.5625`;
5. X external-token-only target lookup = 0.25 under the frozen generated stream;
6. value-scrambled oracle accuracy is within `[0.23,0.27]`;
7. token width, active schedule, owner firewall and zero padding are identical across R0/R1;
8. Test remains `CLOSED/0`.

Any failure => no training.

## 6. Fresh seeds and matrix

Fresh seeds exactly:

- `20276101`
- `20276102`

Matrix:

`2 rules × 2 past-only cells × 2 seeds × 4 architectures = 32 scientific units / 8 paired groups`.

Architectures:

- Persistent Innovation-SGW;
- Reset Innovation-SGW;
- RCL340 Local;
- NoComm.

No role-binding expansion is repeated because PX1 already established physical X robustness over three complete bindings.

## 7. Training recipe

Use the exact MAG1b/PX1 recipe:

- Adam lr `1e-3`, weight decay 0;
- batch 64;
- 2500 optimizer steps;
- validation every 250;
- deterministic validation N=4096;
- gradient clip 1.0;
- checkpoint selected by minimum validation CE;
- deterministic arm64 torch2.2.2, one torch thread;
- at most four independent paired groups concurrently;
- paired data/dropout streams across architectures within each rule/cell/seed;
- no restart, replacement, extension, tuning, curriculum or architecture patch.

## 8. Predeclared gates

For EACH rule and BOTH seeds individually:

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

### Overall

`V1_PASS = all R0 endpoint gates && all R1 endpoint gates`.

No requirement is imposed that R1 numerical accuracies exactly equal R0. The estimand is preservation of the qualitative architecture-ordering signature under matched information geometry.

## 9. Interpretation

PASS supports a bounded claim that the past-only scope signature is not tied to the exact axis-aligned selector/evidence truth rule.

FAIL preserves the parent result but narrows task generalization and blocks architecture transportability/boundary mapping until the failure is understood. No post-exposure rule patch, seed replacement, threshold change or budget extension is allowed.

## 10. Stop rule

Stop after the 32-unit V1 development qualification for scientific review. No source-faithful workspace transportability, CTM test, continuous edge scan, sparse interior scan, dense/adaptive matrix, or Test access is automatically authorized.
