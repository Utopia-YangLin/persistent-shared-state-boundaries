# Reproducibility and audit guide

This archive separates **result verification** from **training reruns** so that post hoc execution cannot be confused with the prospectively frozen comparisons reported in the manuscript.

## Result verification

The preferred publication-audit path is:

```bash
python reproduce/verify_publication_results.py
```

This script reads the sanitized machine-readable publication summary and checks the manuscript-facing headline quantities and evidence boundaries. It performs no optimization and creates no new scientific units.

## Training-code snapshot

The core primary task, canonical clean architecture family, scope/ownership host, causal-intervention host, and primary qualification runner are mirrored in `src/` and `tools/`. Their public Git blob SHA values are checked against the frozen source identities and recorded in `src/SOURCE_CODE_MANIFEST.json`.

Additional validation, basin, transportability, and B0 runner identities are recorded by exact frozen source path and Git blob SHA rather than replaced with post hoc cleaned implementations. Their scientific settings remain fully specified by `protocols/`, and their manuscript-facing outputs are preserved in `results/publication_claims_v0.1.json` plus `results/ORIGINAL_ARTIFACT_MANIFEST.json`.

The frozen qualification environment used deterministic float32 execution on arm64, PyTorch 2.2.2, one Torch thread, and `OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The convenience environment is listed in `requirements-publication.txt`; `requirements.txt` and `pyproject.toml` are copied from the source snapshot.

## Important distinction: semantic rerun versus prospective replication

A rerun from this publication repository can test the mirrored implementation under a compatible environment. It cannot reconstruct the original pre-exposure state of source-tree Q0 guards. The original Q0/static guards checked source-tree state, exact file identities, seed namespaces, or other pre-exposure conditions in the source repository. Their frozen protocols and result identities are historical audit evidence; rerunning them now would not constitute a new prospective preregistration.

## Statistical/reporting boundary

- Primary scope × availability and validation families use two fixed seeds and are reported descriptively/seedwise; they are not population-level estimates.
- The final-clean Adam basin characterization uses 64 seeds per architecture and Wilson 95% confidence intervals for basin incidence.
- Alternative optimizer/recipe comparisons do not establish a general optimizer ranking.
- Validation families are not pooled into a meta-effect.
- The Local-side B0 screen is a finite prospective negative screen, not evidence that a Local-dominant regime is impossible.
- No post-scope held-out Test set was opened (`Test CLOSED/0`).

## Result-artifact sanitization

The original machine-readable training reports contain machine-local checkpoint paths and private research-orchestration locations. Those raw files are therefore not republished verbatim. `results/ORIGINAL_ARTIFACT_MANIFEST.json` records each frozen artifact's source path, Git blob SHA, and byte size, while `results/publication_claims_v0.1.json` preserves the manuscript-facing numerical content without private machine paths.

The complete final basin characterization is substantially larger than the other artifacts; its publication-facing counts, continuations, and confidence-interval inputs are preserved in the same sanitized summary, with the original 1,158,240-byte artifact identified by Git blob SHA.

## Figures

The accepted Figure 1–5 PDFs are submission attachments. A direct binary transfer through the available repository connector did not preserve the local Git blob hash, so no corrupted or lossy PDF was attached to the repository tree. Their numerical content remains auditable through the public summary, supplement, and frozen artifact map.
