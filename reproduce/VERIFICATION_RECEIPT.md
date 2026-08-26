# Publication archive verification receipt

**Date:** 2026-08-26  
**Status:** `PASS`

This receipt records a publication-facing consistency check only. It adds no model forward, optimization step, seed, checkpoint selection, or scientific observation.

## Inputs

- `results/publication_claims_v0.1.json` — Git blob `d91905245b798105023229e7ccff17050f6b958c`
- `reproduce/verify_publication_results.py` — Git blob `e220bdfd0cd04d7e244aa97cd786191825b0f89f`

The public JSON parsed successfully. The verification arithmetic reproduced the frozen manuscript-facing invariants, including:

- critical cross-scope + past-only Persistent−Reset mean = `0.4244384765625`;
- critical cross-scope + past-only Persistent−Local mean = `0.4224853515625`;
- Adam basin counts = Persistent `58 PASS / 6 BASIN / 0 OTHER_FAIL`, Reset `60 / 4 / 0`, Local `63 / 1 / 0`;
- Wilson 95% basin-incidence intervals:
  - Persistent: `0.043678..0.189829`;
  - Reset: `0.024571..0.149975`;
  - Local: `0.002764..0.083341`;
- selected continuation recovery = `7/10` under at least one frozen continuation by 7.5k;
- causal broadcast-timing patterns and Boolean flags match the frozen manuscript summary;
- source-faithful transportability mean gaps reproduce as approximately `0.436279` versus Reset and `0.4349365` versus the direct comparator;
- B0E Persistent−Local remote/past gap = `0.4375` on both seeds;
- proposed Local endpoint Local−Persistent gap = `0.0` on both seeds;
- B0L has no positive candidate, no selected candidate, and no fifth post-exposure candidate;
- post-scope Test status remains `CLOSED/0`.

## Source-fidelity checks

The following public files have Git blob SHA values identical to the frozen private-source blobs recorded in `src/SOURCE_CODE_MANIFEST.json`:

- canonical clean architecture family;
- primary scope×availability task generator;
- ownership-firewall host;
- causal-intervention host;
- primary qualification runner.

`manuscript/REFERENCES.bib` likewise retains the verified bibliography blob identity `8fcdda57d7d6b13ba5da4aae63178a013af2e258`.

## Public-surface hygiene

A repository-content search found no published `/Users/...` machine path, `.wbb-orchestrator` path, manuscript correspondence email, Gmail address, or `sk-` API-key-like string. Original result artifacts containing machine-local checkpoint paths are represented instead by sanitized numerical summaries plus exact original source path/blob/size identities.

Git commit metadata is governed by the GitHub account's configured commit identity and is separate from repository file contents.
