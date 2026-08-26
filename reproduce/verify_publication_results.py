#!/usr/bin/env python3
"""Verify manuscript-facing numerical invariants from the sanitized publication summary.

This script performs no model forward, optimization, checkpoint selection, or new
scientific analysis. It checks arithmetic and frozen reporting boundaries only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "results" / "publication_claims_v0.1.json"


def mean(xs):
    return sum(xs) / len(xs)


def close(a, b, tol=5e-6):
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{a} != {b} (tol={tol})")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main():
    d = json.loads(CLAIMS.read_text())
    assert d["test_status"] == {"gate": "CLOSED", "model_forward_count": 0}
    assert d["chance_accuracy"] == 0.25

    primary = d["primary_scope_availability"]["cells"]
    for name in ("consumer_local_current", "consumer_local_past_only", "cross_scope_current"):
        cell = primary[name]
        assert cell["persistent"] == [1.0, 1.0]
        assert cell["reset"] == [1.0, 1.0]
        assert cell["local"] == [1.0, 1.0]
        assert cell["nocomm"] == [0.25, 0.25]

    critical = primary["cross_scope_past_only"]
    close(mean(critical["reset"]), 0.5755615234375, 1e-12)
    close(mean(critical["local"]), 0.5775146484375, 1e-12)
    p_minus_r = [p-r for p, r in zip(critical["persistent"], critical["reset"])]
    p_minus_l = [p-l for p, l in zip(critical["persistent"], critical["local"])]
    close(mean(p_minus_r), 0.4244384765625, 1e-12)
    close(mean(p_minus_l), 0.4224853515625, 1e-12)
    assert p_minus_r == [0.4189453125, 0.429931640625]
    assert p_minus_l == [0.41748046875, 0.427490234375]

    basin = d["solution_accessibility"]
    assert basin["primary_units"] == 288
    for arch, expected in {"persistent": (58,6,0), "reset": (60,4,0), "local": (63,1,0)}.items():
        c = basin["adam_counts"][arch]
        assert (c["pass"], c["basin"], c["other_fail"]) == expected
        assert sum(expected) == 64
    ci_p = wilson(6, 64)
    ci_r = wilson(4, 64)
    ci_l = wilson(1, 64)
    close(ci_p[0], 0.04367825258519838, 1e-15); close(ci_p[1], 0.18982879599645597, 1e-15)
    close(ci_r[0], 0.024571201396618017, 1e-15); close(ci_r[1], 0.14997485092208662, 1e-15)
    close(ci_l[0], 0.002763541923337505, 1e-15); close(ci_l[1], 0.08334101600094265, 1e-15)
    assert basin["selected_continuations"]["recover_at_least_one_by_7500"] == {"count": 7, "n": 10}

    causal = d["causal_decomposition"]
    assert causal["broadcast_timing"]["cross_scope_current_reset"] == {"all":1.0,"intermediate_only":1.0,"final_only":0.25,"none":0.25}
    assert causal["broadcast_timing"]["cross_scope_past_only_persistent"] == {"all":1.0,"intermediate_only":1.0,"final_only":1.0,"none":0.25}
    assert causal["flags"] == {
        "CURRENT_CONTEXT_BROADCAST_CAUSAL": True,
        "PAST_CONTEXT_BROADCAST_CAUSAL": True,
        "PERSISTENCE_CAUSAL_MAIN": True,
        "PERSISTENCE_SCOPE_SPECIFIC": False,
        "PERSISTENCE_X_BROADCAST_SYNERGY": False,
    }

    val = d["validation_envelope"]
    close(mean(val["source_faithful_transportability"]["p_minus_reset_seed"]), 0.436279, 5e-7)
    close(mean(val["source_faithful_transportability"]["p_minus_direct_seed"]), 0.4349365, 5e-7)
    assert val["pooling"] == "none"

    b0 = d["theory_boundary"]
    for i in range(2):
        close(b0["b0e_remote_past"]["persistent"][i] - b0["b0e_remote_past"]["local"][i], 0.4375, 0.0)
        close(b0["b0e_proposed_local_endpoint"]["local"][i] - b0["b0e_proposed_local_endpoint"]["persistent"][i], 0.0, 0.0)
    assert b0["b0l"]["positive_candidates"] == []
    assert b0["b0l"]["selected_candidate"] is None
    assert b0["b0l"]["fifth_candidate_opened"] is False

    print("PASS: publication-facing arithmetic and evidence-boundary checks are consistent.")
    print("Adam basin Wilson 95% CIs:")
    print(f"  Persistent: {ci_p[0]:.6f}..{ci_p[1]:.6f}")
    print(f"  Reset:      {ci_r[0]:.6f}..{ci_r[1]:.6f}")
    print(f"  Local:      {ci_l[0]:.6f}..{ci_l[1]:.6f}")


if __name__ == "__main__":
    main()
