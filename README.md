# Persistent Shared State Boundaries

Publication-facing archive for the manuscript:

**When Does Persistent Shared State Matter? Functional Boundaries, Solution Accessibility, and Causal Mechanisms in Modular Recurrent Systems**

Author: **Qiyang Lin**  
Affiliation: **University of Melbourne**

## Scientific claim

Persistent shared state is selectively useful rather than uniformly superior. In the controlled task construct reported here, its clearest behavioral advantage appears when task-relevant state must both cross consumer scope and become unavailable from the environment before it is used. Complementary optimization, causal-intervention, transportability, and prospective negative-result analyses characterize when this effect is reachable, how it is used, how far it transports, and which stronger theory is not supported.

This repository is a **clean publication archive**, not the original research-development repository. It intentionally excludes orchestration infrastructure, obsolete architectures, exploratory failure history, and unrelated engineering state.

## What is archived

- `manuscript/` — submission-facing supplement, verified bibliography, and finalized submission metadata.
- `protocols/` — frozen scientific protocols governing the reported primary, causal, optimization, validation, transportability, and theory-boundary studies.
- `results/` — sanitized machine-readable manuscript-facing result summaries plus exact source-artifact Git blob identities and byte sizes.
- `src/` and `tools/` — SHA-verified exact mirrors of the core primary scientific implementation; `src/SOURCE_CODE_MANIFEST.json` separately records additional frozen source identities that are not duplicated into this initial clean archive.
- `reproduce/` — publication-facing numerical verification utilities and a map from manuscript claims to archived artifacts.
- `SOURCE_SNAPSHOT.md` — provenance of the publication snapshot.

The final Figure 1–5 PDFs are submission attachments. They are not reconstructed from lossy binary transfers through this repository interface; their scientific source map is represented by the manuscript/supplement and archived result identities.

## Two levels of reproducibility

### 1. Publication-result verification

The lightweight verification path recomputes the headline quantities reported in the manuscript directly from the archived machine-readable summary. It does not train new models and therefore does not consume new experimental evidence.

```bash
python reproduce/verify_publication_results.py
```

### 2. Training-code audit / semantic rerun

The exact-mirrored core primary source and runner are provided for source audit and semantic reruns. Additional study runners remain bound by their exact frozen source Git blob identities in `src/SOURCE_CODE_MANIFEST.json`; they are not silently replaced with cleaned reimplementations. The original prospective Q0 guards were intentionally tied to the source repository's pre-exposure state, exact Git blob identities, and directory allow-lists. Their protocols and audit identities are archived as historical evidence; rerunning code from this publication repository is **not relabeled as a fresh prospective preregistration**. See `REPRODUCIBILITY.md`.

## Environment used for the frozen runs

The final reported qualification families were run deterministically in float32 on arm64 with PyTorch 2.2.2, one Torch thread, and `OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The convenience environment is listed in `requirements-publication.txt`; exact study-specific settings are frozen in `protocols/`.

## Evidence boundary

The main two-seed qualification studies are descriptive, seedwise gate-based comparisons rather than population-level inferential estimates. The 64-seed Adam basin characterization is the only study in which Wilson confidence intervals are used. Validation families are not pooled into a meta-effect. The prospective Local-side screen is finite negative evidence and is not an impossibility proof. No held-out Test set was opened for the post-scope programme.

## Data and code availability

The machine-readable quantities needed to audit the numerical statements in the manuscript are archived here. `results/ORIGINAL_ARTIFACT_MANIFEST.json` identifies the original frozen result artifacts by source path, Git blob SHA, and byte size. Raw internal result files that contain machine-local checkpoint paths and private research-orchestration state are intentionally not republished; their manuscript-facing numerical content is preserved in sanitized summaries.

The submission-facing snapshot is frozen on branch `submission-v0.1` after final audit.

## License status

No open-source license is granted by this initial publication snapshot. The repository is public for scientific inspection and verification, but reuse rights beyond applicable law and GitHub's platform terms are not implied. A separate open-source license may be added by the author later.

## Citation

Citation metadata is provided in `CITATION.cff`. Until a DOI or final archival identifier is assigned, cite the manuscript and this GitHub repository together.
