# Supplementary Material — When Does Persistent Shared State Matter?

**Status:** `CURRENT_SUBMISSION_SUPPLEMENT_SOURCE__A_TO_G_ONLY__READER_FACING__NO_DEVELOPMENT_PROVENANCE_SECTION`

**Purpose:** retain only supplementary evidence that directly supports a current main-text claim. Historical development provenance remains outside this public submission-facing supplement.

---

# Supplement A — Primary scope × temporal-availability qualification

The primary qualification compares Persistent, Reset, Local, and NoComm across consumer-local/cross-scope × CURRENT/PAST-ONLY conditions under two paired scientific seeds.

| Condition | Persistent | Reset | Local | NoComm |
|---|---:|---:|---:|---:|
| consumer-local + CURRENT | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.2500 / 0.2500 |
| consumer-local + PAST-ONLY | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.2500 / 0.2500 |
| cross-scope + CURRENT | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.2500 / 0.2500 |
| cross-scope + PAST-ONLY | 1.0000 / 1.0000 | 0.5811 / 0.5701 | 0.5825 / 0.5725 | 0.2500 / 0.2500 |

Critical paired contrasts:

| Contrast | seed 1 | seed 2 | mean |
|---|---:|---:|---:|
| Persistent − Reset | +0.4189 | +0.4299 | +0.4244 |
| Persistent − Local | +0.4175 | +0.4275 | +0.4225 |

Both contrasts are exactly zero in each of the other three cells. The primary two-seed qualification is interpreted descriptively and does not locate a continuous onset surface.

---

# Supplement B — Final-clean solution-accessibility characterization

The accessibility study uses a separate final-clean DSTAR host and is not another primary factorial cell. The final runtime-reduced characterization contains 288 primary units: 192 in the complete 64-seed Adam incidence arm and 48 units in each of two matched 24-seed alternative-recipe arms.

| Architecture | PASS | BASIN | OTHER FAIL | Basin proportion | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Local | 63 | 1 | 0 | 0.0156 | [0.0028, 0.0833] |
| Reset | 60 | 4 | 0 | 0.0625 | [0.0246, 0.1500] |
| Persistent | 58 | 6 | 0 | 0.0938 | [0.0437, 0.1898] |

Matched 24-seed recipe outcomes:

| Architecture | Adam | frozen full recipe | SGD+momentum |
|---|---|---|---|
| Persistent | 23 PASS + 1 BASIN | 24 OTHER FAIL | 24 OTHER FAIL |
| Reset | 23 PASS + 1 BASIN | 24 OTHER FAIL | 24 OTHER FAIL |

Selected basin-checkpoint continuations:

| Architecture | n | both recovered | constant only | warm only | neither by 7.5k |
|---|---:|---:|---:|---:|---:|
| Persistent | 6 | 2 | 1 | 1 | 2 |
| Reset | 4 | 3 | 0 | 0 | 1 |

Across the ten selected basin checkpoints, 7/10 recovered under at least one frozen continuation by 7.5k. Disappearance of the named basin under the alternative recipes is therefore not equivalent to improved accessibility.

---

# Supplement C — Causal decomposition

Broadcast timing:

| Condition | ALL | INTERMEDIATE ONLY | FINAL ONLY | NONE |
|---|---:|---:|---:|---:|
| cross-scope + CURRENT, Reset | 1.0000 | 1.0000 | 0.2500 | 0.2500 |
| cross-scope + PAST-ONLY, Persistent | 1.0000 | 1.0000 | 1.0000 | 0.2500 |

Persistent-carry removal:

| Condition | seed 1 drop | seed 2 drop | mean drop |
|---|---:|---:|---:|
| consumer-local + PAST-ONLY | 0.5100 | 0.5334 | 0.5217 |
| cross-scope + CURRENT | 0.2981 | 0.4399 | 0.3690 |
| cross-scope + PAST-ONLY | 0.6284 | 0.4263 | 0.5273 |

The frozen causal interpretation supports current-context broadcast dependence, past-context broadcast dependence, and a persistence causal main effect, but not a universal scope-specific persistence interaction.

---

# Supplement D — Consumer and host validation

Consumer rebinding:

| Consumer binding | seed 1 P−Reset | seed 2 P−Reset | mean |
|---|---:|---:|---:|
| map 1 | 0.4314 | 0.4365 | 0.4340 |
| map 2 | 0.4248 | 0.4399 | 0.4324 |
| map 3 | 0.4268 | 0.4309 | 0.4288 |

Memory-access geometry:

| Geometry | seed 1 P−Reset | seed 2 P−Reset | mean |
|---|---:|---:|---:|
| canonical | 0.4216 | 0.4290 | 0.4253 |
| state-only | 0.4309 | 0.4316 | 0.4313 |

The state-only geometry removes the previous-hidden-state contribution to input-attention selection while preserving ordinary recurrent h/c carry.

---

# Supplement E — Same-construct rule generalization

| Rule | seed 1 P−Reset | seed 2 P−Reset | mean |
|---|---:|---:|---:|
| original | 0.4314 | 0.4297 | 0.4305 |
| altered same-construct rule | 0.4358 | 0.4338 | 0.4348 |

Before training, both rules retained oracle accuracy 1.0, selector-hidden Bayes joint accuracy 0.5625, direct consumer-token chance accuracy 0.25, and matched information geometry.

---

# Supplement F — Source-faithful second-family transportability

| Condition | Persistent | Reset | Local/direct | NoComm |
|---|---:|---:|---:|---:|
| consumer-local + PAST-ONLY | 1.0000 | 1.0000 | 1.0000 | 0.2500 |
| cross-scope + PAST-ONLY | 1.0000 | 0.5637 | 0.5651 | 0.2500 |

| Seed | Persistent − Reset | Persistent − Local/direct |
|---|---:|---:|
| 20276201 | +0.4343 | +0.4338 |
| 20276202 | +0.4382 | +0.4360 |
| mean | +0.4363 | +0.4349 |

All predeclared cell-by-seed gates passed across 16 scientific units with zero engineering failures. The result supports qualitative transportability under the common frozen host/bridge, not numerical or bridge-independent universality.

---

# Supplement G — Prospective theory-boundary tests

| Endpoint | Persistent | Reset | Local | NoComm | Decision |
|---|---:|---:|---:|---:|---|
| remote/past Persistent-favoring | 1.0000 | 0.5625 | 0.5625 | 0.2500 | Persistent − Local = +0.4375 on both seeds |
| proposed Local-favoring | 1.0000 | 1.0000 | 1.0000 | 0.2500 | Local − Persistent = 0.0000; gate failed |

A subsequent finite four-candidate Local-side mechanism screen produced Persistent = Local = 1.0000 for every candidate and both paired seeds. Across 8 paired groups / 16 scientific units, Local − Persistent was 0.0000 throughout, no positive candidate was admitted, and the route was closed by the frozen stop rule rather than extended with a fifth post-exposure candidate.

Supported theory: **local/communication sufficiency → Persistent shared-state advantage under joint cross-scope and temporal-unavailability demand.** A finite negative screen does not prove Local-dominant regimes impossible.

---

# Reproducibility and data availability

Publication-facing code, frozen protocols, and machine-readable manuscript-facing result summaries are available in this repository. `results/ORIGINAL_ARTIFACT_MANIFEST.json` records the source paths, Git blob identities, and sizes of the original frozen result artifacts. Machine-local checkpoint paths and private research-orchestration state are intentionally not republished. The post-scope Test gate remained `CLOSED/0`.
