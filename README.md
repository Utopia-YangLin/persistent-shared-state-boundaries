# Persistent Shared State Boundaries

Publication-facing archive for the manuscript:

**When Does Persistent Shared State Matter? Functional Boundaries, Solution Accessibility, and Causal Mechanisms in Modular Recurrent Systems**

Author: **Qiyang Lin**  
Affiliation: **University of Melbourne**

## Scientific claim

Persistent shared state is selectively useful rather than uniformly superior. In the controlled task construct reported here, its clearest behavioral advantage appears when task-relevant state must both cross consumer scope and become unavailable from the environment before it is used. Complementary optimization, causal-intervention, transportability, and prospective negative-result analyses characterize when this effect is reachable, how it is used, how far it transports, and which stronger theory is not supported.

This repository is a **clean publication archive**, not the original research-development repository. It intentionally excludes orchestration infrastructure, obsolete architectures, exploratory failure history, and unrelated engineering state.

## What is archived

- `manuscript/` — submission-facing manuscript, supplement, tables, references, and metadata.
- `protocols/` — frozen scientific protocols governing the reported primary, causal, optimization, validation, transportability, and theory-boundary studies.
- `results/` — machine-readable result artifacts and relevant static/Q0 audit records.
- `src/` — the final source modules needed by the reported scientific paths.
- `tools/` — frozen experiment runners used for the reported studies.
- `reproduce/` — publication-facing verification utilities and a map from manuscript claims to archived artifacts.
- `SOURCE_SNAPSHOT.md` — provenance of the publication snapshot.

## Two levels of reproducibility

### 1. Publication-result verification

The lightweight verification path recomputes the headline quantities reported in the manuscript directly from the archived machine-readable results. It does not train new models and therefore does not consume new experimental evidence.

```bash
python reproduce/verify_publication_results.py
```

### 2. Training-code audit / semantic rerun

The repository also preserves the frozen training runners and the source modules on which the reported experiments depended. The original prospective Q0 guards were intentionally tied to the source repository's pre-exposure state, exact Git blob identities, and directory allow-lists. Those guards are archived as evidence; they are **not relabeled as a fresh prospective preregistration in this publication repository**. See `REPRODUCIBILITY.md` before attempting training reruns.

## Environment used for the frozen runs

The final reported qualification families were run deterministically in float32 on arm64 with PyTorch 2.2.2, one Torch thread, and `OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The minimal package dependencies are listed in `requirements.txt`. Exact study-specific settings are frozen in `protocols/` and in the corresponding runners.

## Evidence boundary

The main two-seed qualification studies are descriptive, seedwise gate-based comparisons rather than population-level inferential estimates. The 64-seed Adam basin characterization is the only study in which Wilson confidence intervals are used. Validation families are not pooled into a meta-effect. The prospective Local-side screen is finite negative evidence and is not an impossibility proof. No held-out Test set was opened for the post-scope programme.

## Data and code availability

All machine-readable artifacts needed to audit the numerical statements in the manuscript are archived in this repository. The publication-facing snapshot is frozen on the `submission-v0.1` branch once assembly is complete.

## License status

No open-source license is granted by this initial publication snapshot. The repository is public for scientific inspection and verification, but reuse rights beyond applicable law and GitHub's platform terms are not implied. A separate open-source license may be added by the author later.

## Citation

Citation metadata is provided in `CITATION.cff`. Until a DOI or final archival identifier is assigned, cite the manuscript and this GitHub repository together.
