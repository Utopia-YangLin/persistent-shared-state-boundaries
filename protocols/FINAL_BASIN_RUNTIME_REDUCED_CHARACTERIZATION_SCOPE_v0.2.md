# Final Basin Runtime-Reduced Characterization Scope v0.2

**Status: PROSPECTIVELY FROZEN BEFORE v0.2 OPTIMIZER EXPOSURE**

This study supersedes the oversized 384-unit Adam+SGD characterization and the full-size warm-restart companion. Cancellation of those superseded workflows is an execution/runtime decision only; no partial result from either superseded run may be used scientifically or to choose this v0.2 design.

The purpose is to preserve the statistically important basin-incidence estimate while bounding total wall time and retaining targeted optimizer/escape/engineering stress tests.

## 1. Scientific questions

1. What is the basin incidence of the final clean family under the original Adam protocol?
2. On a smaller prospectively fixed paired subset, is the SGW basin strongly Adam-specific or does it also appear under SGD+momentum?
3. For a bounded prospectively selected subset of actual SGW basin trajectories, does simply training longer or applying cosine warm restarts permit escape?
4. On the same fixed small paired subset, can a strong conventional non-architectural recipe (SGD+momentum + warmup + cosine warm restarts) materially suppress the SGW basin?

No result may reopen architecture/task/loss/threshold search, X2, Learned-Relevance L2, Phase A, or Test.

## 2. Frozen architectures and task

Task: frozen `DSTAR_TEMPORAL` only.

Final clean architectures:
- `sgw_persistent_innovation_clean` — Persistent SGW MLP1;
- `sgw_reset_innovation_clean` — Reset SGW MLP1;
- `rcl340_local_clean` — Local RCL340 MLP0.

Test remains `CLOSED/0`.

## 3. Seed namespaces and matrix

### A. Primary Adam incidence — NOT reduced

Use all 64 fresh paired seeds `20274201..20274264` for all three clean architectures.

Total: `64 x 3 = 192` primary units.

This is the only study component used for the principal basin-incidence estimate with Wilson 95% intervals.

### B. SGD+momentum robustness — reduced

Use the prospectively fixed first 24 seeds `20274201..20274224` for Persistent and Reset only.

Total: `24 x 2 = 48` primary units.

RCL340 is omitted here because this arm asks whether the SGW-family basin is optimizer-specific; Local stability is already estimated under the complete Adam incidence arm.

### C. Full conventional optimization recipe — reduced

Use the exact same 24 seeds `20274201..20274224` for Persistent and Reset only.

Total: `24 x 2 = 48` primary units.

### Total primary budget

`192 + 48 + 48 = 288` primary 2500-step units.

No seed replacement, best-of-R, or post-result seed extension is allowed.

## 4. Frozen primary protocols

Common:
- native arm64 CPU;
- PyTorch 2.2.2;
- float32 deterministic;
- one PyTorch thread per process;
- maximum four independent workers;
- batch 64;
- gradient clip 1;
- final-only 4-class CE;
- validation every 250 steps;
- validation N=4096;
- primary budget 2500 optimizer steps;
- primary selected checkpoint = minimum validation CE after full exposure, earliest exact tie;
- capability threshold 0.95 for joint, early-bit, late-bit.

### Adam reference
`torch.optim.Adam(lr=1e-3, weight_decay=0)`, no scheduler.

### SGD+momentum
`torch.optim.SGD(lr=1e-2, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)`, no scheduler.

### Full conventional recipe
Optimizer: the same SGD+momentum definition.

Learning-rate trajectory is frozen, with no tuning:
- updates 0..249: linear warmup from `1e-3` to `1e-2`;
- from update 250 onward: cosine warm restarts with `eta_max=1e-2`, `eta_min=0`, cycle length `T_0=500` optimizer updates, `T_mult=1`;
- no scheduler/LR search and no validation-triggered changes.

The full recipe is a stress test of whether conventional optimizer/scheduling machinery can suppress the basin; it is not a replacement primary protocol.

## 5. Primary labels at 2500

From the frozen selected checkpoint:
- `CAPABILITY_PASS`: finite and joint/early/late all >=0.95;
- `LOW_CURRENT_LEVERAGE_BASIN`: finite, early >=0.95, joint <0.95, late <0.95;
- `OTHER_FAIL`: all other outcomes.

Primary labels are immutable.

## 6. Runtime-bounded escape subset

Longer-step and warm-restart characterization is limited to Adam and plain SGD SGW cells only:
- Adam/Persistent;
- Adam/Reset;
- SGD/Persistent;
- SGD/Reset.

Eligibility requires BOTH:
- selected primary label = `LOW_CURRENT_LEVERAGE_BASIN`;
- actual step-2500 endpoint label = `LOW_CURRENT_LEVERAGE_BASIN`.

Within each cell, sort eligible seeds ascending and select at most the first **8**. This deterministic cap is frozen before v0.2 exposure. It is not selected by severity, score, or ease of recovery.

Thus at most 32 escape trajectories are opened.

The full-recipe arm receives no step extension.

## 7. Paired escape interventions

Each selected escape trajectory must persist its exact step-2500 model and optimizer state during the primary run. From that identical state create two deterministic branches.

### Constant-LR continuation
Continue the exact original optimizer with unchanged LR to fixed step 5000. If all capability metrics pass, label `CONST_RECOVERED_AT_5000` and stop. If still basin, continue unchanged to fixed step 7500 and label `CONST_RECOVERED_AT_7500`, `CONST_STILL_BASIN_AT_7500`, or `CONST_OTHER_FAIL_AT_7500`.

### Cosine warm-restart continuation
From the same step-2500 model+optimizer state, attach a cosine warm-restart schedule:
- inherited base LR (`1e-3` Adam or `1e-2` SGD);
- `eta_min=0`;
- `T_0=500` optimizer updates;
- `T_mult=1`;
- first post-2500 update uses the inherited base LR;
- no parameter reset, optimizer-state reset, adaptive restart timing, or scheduler tuning.

Evaluate fixed step 5000, and only if still basin continue to fixed step 7500. Labels mirror the constant branch with prefix `WARM_`.

No best-of-extension checkpoint is allowed.

## 8. Required reporting

### Adam incidence
For each of Persistent/Reset/RCL340, report pass/basin/other counts out of 64 and Wilson 95% CI for basin proportion.

### SGD paired robustness
For Persistent and Reset on the fixed 24 seeds, report pass/basin/other counts plus Adam-vs-SGD paired basin contingency.

### Full recipe stress test
For Persistent and Reset on the exact same 24 seeds, report pass/basin/other counts and paired contingency versus Adam and versus plain SGD.

### Escape subset
For each of four optimizer x SGW cells, report:
- total eligible basin count;
- selected-for-escape count (max 8);
- selected seed IDs;
- constant-LR outcomes;
- warm-restart outcomes;
- paired counts: both recover, constant-only, warm-only, neither by 7500, other transitions.

The escape subset estimates mechanism/escapability only; it is not used to re-estimate population basin incidence.

## 9. Runtime and governance

The study is intentionally sized for approximately 6–7 hours end-to-end on the existing four-worker M1 self-hosted setup, including the already-spent time on superseded runs. Runtime is an engineering budget, not a scientific stopping rule: no unit may be dropped after observing its result.

After v0.2 completes:
- no optimizer or scheduler sweep;
- no AdamW/RMSProp/SAM/OneCycle expansion;
- no additional seed extension;
- no architecture/task/loss/threshold rescue;
- X2 locked;
- Learned-Relevance L2 locked;
- Phase A unauthorized;
- Test `CLOSED/0`;
- next activity = manuscript integration.
