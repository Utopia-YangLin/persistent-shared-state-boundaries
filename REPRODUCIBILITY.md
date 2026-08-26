# Reproducibility and audit guide

This archive separates **result verification** from **training reruns** so that post hoc execution cannot be confused with the prospectively frozen comparisons reported in the manuscript.

## Result verification

The preferred publication-audit path is:

```bash
python reproduce/verify_publication_results.py
```

This script reads archived machine-readable artifacts and checks the manuscript-facing headline quantities and evidence boundaries. It performs no optimization and creates no new scientific units.

## Training-code snapshot

`tools/` and `src/` preserve the final scientific runners and their necessary source modules. Exact study-specific seeds, gates, schedules, and training hyperparameters are defined by the corresponding files in `protocols/` and the runners themselves.

The frozen qualification environment used deterministic float32 execution on arm64, PyTorch 2.2.2, one Torch thread, and `OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The convenience environment is listed in `requirements-publication.txt`; `requirements.txt` and `pyproject.toml` are copied from the source snapshot.

## Important distinction: semantic rerun versus prospective replication

A rerun from this publication repository can test whether the archived implementation still reproduces the same qualitative behavior under a compatible environment. It cannot reconstruct the original pre-exposure state of source-tree Q0 guards. The archived Q0 JSON records document those checks as they occurred in the source repository.

## Statistical/reporting boundary

- Primary scope × availability and validation families use two fixed seeds and are reported descriptively/seedwise; they are not population-level estimates.
- The final-clean Adam basin characterization uses 64 seeds per architecture and Wilson 95% confidence intervals for basin incidence.
- Alternative optimizer/recipe comparisons do not establish a general optimizer ranking.
- Validation families are not pooled into a meta-effect.
- The Local-side B0 screen is a finite prospective negative screen, not evidence that a Local-dominant regime is impossible.
- No post-scope held-out Test set was opened (`Test CLOSED/0`).

## Large artifact note

The complete final basin characterization is substantially larger than the other JSON artifacts. If the full JSON cannot be mirrored through a particular GitHub API client, the publication repository retains a derived manuscript-facing basin summary together with the original source path, Git blob SHA, file size, and frozen scope document. Such a summary is clearly labeled derived and must not be mistaken for the original raw characterization.
