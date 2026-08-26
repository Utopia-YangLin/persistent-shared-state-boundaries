# T1 Source-Faithful Shared Workspace Transportability Protocol v0.1

**Status:** `PROSPECTIVELY_FROZEN__BEFORE_T1_SOURCE_IMPLEMENTATION_MODEL_FORWARD_OR_OPTIMIZER_STEP`  
**Parent scientific review:** `docs/analysis/V1_VALUE_GENERALIZATION_SCIENTIFIC_REVIEW_v0.1.md`, commit `cb754764f9bc35cb02e8f9645a9cba43d017eca2`  
**Parent V1 durable result:** commit `94a8c8808be7cbe29401b4a50971b05629cf30f2`, workflow `32875423900`  
**Purpose:** smallest second-family architecture transportability qualification of the already established past-only scope signature.

## 1. Question

> Does the `PRIVATE_PAST` versus `SHARED_PAST` persistence signature survive when the Innovation-SGW communication mechanism is replaced by the independently developed, accepted-paper/reference-derived Shared Workspace communication operator, while retaining the cleaned V1 outer task/recurrent host?

This is an architecture-family transportability check. It is **not** a new task axis, not a raw full-model benchmark against Phase0Ref, and not a continuous boundary study.

## 2. Frozen source identities

The following Git blobs are frozen inputs to T1:

- V1 protocol: `2529dd5e3c909d92f4c72fe4c76c21a723283338`;
- V1 task source: `bed6b8110803041ecc394df6034bfc02a4b7f05b`;
- MAG1a STATE_ONLY host: `a213b0a8aae873dcfba84e1b32ed7f403c772fdf`;
- canonical Task-v2.2 clean family: `1d192f8786132d48188125b9bc371598ae60c497`;
- source-complete Shared Workspace operator: `228d0837d21736f4982f9d278d15a4600c76f696` (`reference_communications_v0_3.py`);
- reference RIM direct communication operator: `fdacc5ef61e7ac61bf927b3956db59e77ed11b55` (`reference_communications_v0_2.py`).

The T1 implementation may import those operators but may not edit or copy-modify their internal scientific equations.

## 3. Full-chassis audit and estimand protection

### 3.1 Rejected: historical Phase0Ref full chassis

The historical source-complete Phase0Ref model uses a common 32D Task-v2.1 processor/recurrent/readout chassis. V1 uses an 85D canonical Task-v2.2 clean recurrent host, 46D deterministic task tokens, the B0 cycle-0 ownership firewall, blind consumer readout, and STATE_ONLY (`Q0_R1`) processor-memory access.

Running the old full Phase0Ref chassis on the T1 task would alter both the communication family and the outer recurrent/readout host. Such raw accuracy is outside the T1 estimand and is prohibited.

### 3.2 Rejected: Phase0L communication

`phase0l_port_communications_v0_1.py` implements a custom fixed-port/additive-delta global mechanism. It is not the accepted-paper/reference-derived Shared Workspace operator frozen by the Phase0Ref source-completion programme and may not stand in for the T1 source-faithful candidate.

### 3.3 Selected: common linear communication bridge

The source-complete SGW and reference RIM operators require `[B,6,32]` states and return `[B,6,32]`; the clean host recurrent state is `[B,6,85]`. T1 therefore freezes the minimum architecture-neutral bridge:

- `sender_bridge`: one `Linear(85,32,bias=False)`;
- `receiver_bridge`: one `Linear(32,85,bias=False)`;
- both bridges are common modules with architecture-independent tagged initialization and identical initial tensors across all four T1 arms;
- no activation, gate, normalization, second channel, task label, exact relevance label, target, or future information is added by the bridge.

At every cycle:

1. execute the exact cleaned STATE_ONLY input/recurrent step: prior `h` is zeroed only for input-attention query; real prior `h/c` enter the recurrent transition;
2. obtain common precommunication `hbar [B,6,85]`;
3. multiply `hbar` by the frozen current-cycle **active processor mask** before the sender bridge, so an inactive processor's retained recurrent state cannot become an undeclared communication path;
4. compute `z = sender_bridge(masked_hbar) [B,6,32]`;
5. pass `z` to the frozen 32D communication arm with `current_relevance_mask=None`;
6. compute `received85 = receiver_bridge(received32)`;
7. use the exact common residual receiver `h = hbar + received85`;
8. after cycle 0, apply the exact V1 B0 ownership firewall to `h/c`; Shared Workspace memory is not erased by the firewall;
9. use the unchanged blind readout from consumer `X=p5` after cycle 3.

For NoComm, `received32` is exact zero; because the receiver bridge has no bias, `received85` must be exact zero.

The active mask is an architecture-neutral environmental availability mask already frozen by the task. It is **not** a three-way relevance/route label. Inactive states are zeroed before the common sender bridge; no attention-logit mask or other internal modification is inserted into the source operators.

## 4. Frozen task subset

Use the exact V1 task generator without modification:

- binding `B0`: `C=p0`, `E=p1`, `X=p5`;
- rule: `AXIS_ALIGNED` only;
- cells: `PRIVATE_PAST`, `SHARED_PAST` only;
- exact four active processors per cycle;
- selector appears only at cycle 0 at the frozen owner;
- `u` at p1 cycle 1, `v` at p1 cycle 2;
- cycle 3 no task content;
- no selector refresh;
- same balanced target construction and information geometry.

`CROSS_V` is deliberately not crossed with T1. V1 has already established same-construct rule generalization; adding rule × architecture here would double the first transport matrix without answering a new primary question.

## 5. Frozen T1 arms

Exactly four arms:

1. `SOURCE_SGW_PERSISTENT`: `SourceCompleteSharedWorkspaceCommunicationV03(reset_each_cycle=False)`;
2. `SOURCE_SGW_RESET`: the same source operator and parameters with `reset_each_cycle=True`;
3. `RIM_DIRECT_REF`: `RIMDirectCommunicationReferenceV02`, the reference-derived nonpersistent direct communication comparator;
4. `NOCOMM_REF`: exact-zero 32D communication.

Primary causal persistence contrast: `SOURCE_SGW_PERSISTENT - SOURCE_SGW_RESET`.

`RIM_DIRECT_REF` is a second nonpersistent communication reference; it is not interpreted as a pure one-factor match to SGW. No raw numerical T1-vs-V1 architecture comparison is a primary estimand.

## 6. Fresh seeds and matrix

Fresh development seeds exactly:

- `20276201`
- `20276202`

Matrix:

`2 cells × 2 seeds × 4 arms = 16 scientific units / 4 paired groups`.

No replacement or additional seed may be introduced after any T1 performance exposure.

## 7. Mandatory Q0 — zero optimizer steps

Before optimizer step 1, T1 must durably verify all of the following:

1. every source blob in section 2 matches exactly;
2. the T1 wrapper/runner was created only after this protocol freeze;
3. task batches are byte/semantic-equivalent to V1 `AXIS_ALIGNED` for the same seed/tag and both cells;
4. symbolic oracle accuracy `1.0`, selector-hidden Bayes `0.5625`, X external-token-only lookup `0.25`, selector×target balance PASS, value-scrambled accuracy within `[0.23,0.27]`;
5. exact STATE_ONLY behavior: input-attention query reference is zero; recurrent `h/c` carry is ON;
6. B0 ownership firewall is identical to V1;
7. sender/receiver bridge names, shapes and initial tensors are identical across all four arms at paired seed;
8. sender bridge is exactly `85->32`, bias-free; receiver bridge exactly `32->85`, bias-free; no additional adapter module exists;
9. before `sender_bridge`, every inactive processor state is exact zero and every active processor state equals the common `hbar` value for the same arm-independent host smoke input;
10. all source arms receive `current_relevance_mask=None`; no exact current relevance or target-derived routing enters a source operator;
11. source-complete Persistent and Reset source-operator state dictionaries are byte-identical at paired seed;
12. SGW write/broadcast scientific geometry remains exactly the source-complete v0.3 operator: 4×32 memory, write source `[M_prev; processor_states]`, one scaled 32D write head, three repeated shared attention-MLP applications, source input/forget gated carry, distinct four-head broadcast, 128->32 output;
13. RIM direct operator source blob and scientific geometry remain unchanged;
14. NoComm produces exact-zero 32D and exact-zero post-bridge 85D receive;
15. finite smoke forwards for all arms/cells without any optimizer step;
16. fresh seeds occur nowhere outside the T1 authority/implementation/report namespace;
17. no Phase0Ref full-chassis validation/performance is consulted as a T1 selection signal;
18. Test remains `CLOSED/0`.

Any Q0 failure blocks training. Only a semantics-identical engineering repair is allowed. A bridge/host/operator scientific change requires a separately versioned protocol and must not use failed T1 performance as a tuning signal.

## 8. Frozen optimization

All four arms use the same prospective recipe:

- Adam `lr=1e-3`, weight decay `0`;
- batch `64`;
- `5000` optimizer steps;
- validation every `250` steps including step 0 and 5000;
- deterministic validation `N=4096`;
- gradient clip `1.0`;
- checkpoint = minimum validation CE, earliest exact tie;
- deterministic arm64 torch `2.2.2`, one torch thread;
- at most four independent paired groups concurrently;
- paired task batches and paired stochastic streams across arms within each cell/seed;
- no scheduler, restart, replacement seed, best-of-R, curriculum, auxiliary loss, budget extension, architecture patch, or result-dependent stopping.

The 5000-step budget is chosen prospectively from the frozen source-complete SGW temporal microqualification budget, not from any T1 performance exposure.

## 9. Predeclared transportability gates

For **both seeds individually**:

### PRIVATE_PAST — communication/local-state sufficiency

- `SOURCE_SGW_PERSISTENT >= 0.90`;
- `SOURCE_SGW_RESET >= 0.90`;
- `RIM_DIRECT_REF >= 0.90`;
- `NOCOMM_REF <= 0.35`.

### SHARED_PAST — persistent-global signature

- `SOURCE_SGW_PERSISTENT >= 0.80`;
- `SOURCE_SGW_PERSISTENT - SOURCE_SGW_RESET >= 0.15`;
- `SOURCE_SGW_PERSISTENT - RIM_DIRECT_REF >= 0.15`;
- `NOCOMM_REF <= 0.35`.

### Overall

`T1_PASS = all PRIVATE_PAST gates && all SHARED_PAST gates`.

The gate mirrors the already frozen qualitative endpoint signature, but **does not require numerical equality with V1**.

## 10. Interpretation

PASS supports the bounded statement that the past-only scope/persistence signature transports from the Innovation-SGW family to a separately developed source-complete Shared Workspace communication family under one retained cleaned outer host and a common minimal 85↔32 communication bridge.

FAIL preserves V1 as valid evidence but blocks any claim of architecture-family transportability. A failure may reflect the second communication family, the common bridge, or their interaction; no post-exposure bridge width, nonlinearity, seed, threshold, budget, task rule, source operator, or host patch is permitted within T1 v0.1.

## 11. Stop rule

After the 16-unit qualification, stop for scientific review.

T1 v0.1 does **not** authorize:

- continuous edge/boundary scans;
- sparse interior, dense or adaptive matrices;
- CTM or another architecture family;
- PX/MAG as new main axes;
- Q/R ratio sweeps;
- CROSS_V × architecture expansion;
- task revision;
- Test access.
