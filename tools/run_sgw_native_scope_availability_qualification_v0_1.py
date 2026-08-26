#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from workspace_broadcast_boundaries.architectures.sgw_native_scope_availability_family_v0_1 import (
    SGWNativeScopeAvailabilityModel,
)
from workspace_broadcast_boundaries.architectures.task_v2_2_clean_family_v0_1 import (
    ARCHS,
    NOCOMM,
    PERSISTENT,
    RCL340,
    RESET,
)
from workspace_broadcast_boundaries.sgw_native_scope_availability_v0_1 import (
    CURRENT,
    PAST_ONLY,
    PRIVATE,
    QUALIFICATION_SEEDS,
    SHARED,
    build_batch,
    cell_id,
    context_owner,
)

STATE = "research_workflow/sgw_native_scope_availability/STATE_v0.1.json"
Q0_REPORT = "reports/sgw_native_scope_availability/Q0_STATIC_AUDIT_v0.1.json"
REPORT = "reports/sgw_native_scope_availability/QUALIFICATION_RESULTS_v0.1.json"
PROGRESS = "reports/sgw_native_scope_availability/PROGRESS_v0.1.json"
TRAIN_STEPS = 2500
VAL_EVERY = 250
VAL_N = 4096
BATCH = 64
LR = 1e-3
CLIP = 1.0
MAX_WORKERS = 4
CELL_SPECS = (
    (PRIVATE, CURRENT),
    (PRIVATE, PAST_ONLY),
    (SHARED, CURRENT),
    (SHARED, PAST_ONLY),
)


def derive(seed: int, tag: str) -> int:
    payload = f"sgw-native-scope-availability-training-v0.1:{int(seed)}:{tag}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def runtime_assertions() -> None:
    if platform.machine() != "arm64":
        raise RuntimeError(f"arm64 required, got {platform.machine()}")
    if not torch.__version__.startswith("2.2.2"):
        raise RuntimeError(f"torch2.2.2 required, got {torch.__version__}")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise RuntimeError("OMP_NUM_THREADS=MKL_NUM_THREADS=1 required")


def metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(-1)
    return {
        "cross_entropy": float(F.cross_entropy(logits, y)),
        "joint_accuracy": float(pred.eq(y).float().mean()),
    }


@torch.no_grad()
def evaluate(model: SGWNativeScopeAvailabilityModel, batch) -> dict[str, float]:
    model.eval()
    out = model(
        batch.tokens,
        forced_active_mask=batch.forced_active_mask,
        context_owner=context_owner(batch.scope),
    )
    return metrics(out.logits, batch.targets)


def train_group(*, scope: str, availability: str, seed: int, run_root: Path) -> dict[str, Any]:
    runtime_assertions()
    cid = cell_id(scope, availability)
    val = build_batch(
        scope=scope,
        availability=availability,
        experimental_seed=seed,
        count=VAL_N,
        tag="qualification:validation",
    )

    models: dict[str, SGWNativeScopeAvailabilityModel] = {}
    opts: dict[str, torch.optim.Optimizer] = {}
    best_state: dict[str, dict[str, torch.Tensor] | None] = {a: None for a in ARCHS}
    best_metrics: dict[str, dict[str, float] | None] = {a: None for a in ARCHS}
    best_step: dict[str, int | None] = {a: None for a in ARCHS}
    histories: dict[str, list[dict[str, Any]]] = {a: [] for a in ARCHS}

    for arch in ARCHS:
        models[arch] = SGWNativeScopeAvailabilityModel(
            arch, experimental_seed=seed, blind_receiver_head=True
        )
        opts[arch] = torch.optim.Adam(models[arch].parameters(), lr=LR, weight_decay=0.0)

    torch.manual_seed(derive(seed, f"paired-dropout:{cid}"))
    initial_rng = torch.random.get_rng_state().clone()
    rng_states = {a: initial_rng.clone() for a in ARCHS}

    started = time.time()
    for step in range(TRAIN_STEPS + 1):
        if step % VAL_EVERY == 0:
            for arch in ARCHS:
                m = evaluate(models[arch], val)
                histories[arch].append({"step": step, **m})
                cur = best_metrics[arch]
                if cur is None or m["cross_entropy"] < cur["cross_entropy"]:
                    best_metrics[arch] = copy.deepcopy(m)
                    best_step[arch] = step
                    best_state[arch] = {
                        n: p.detach().cpu().clone()
                        for n, p in models[arch].state_dict().items()
                    }
        if step == TRAIN_STEPS:
            break

        train = build_batch(
            scope=scope,
            availability=availability,
            experimental_seed=seed,
            count=BATCH,
            tag=f"qualification:train:{step}",
        )
        owner = context_owner(scope)
        for arch in ARCHS:
            model = models[arch]
            opt = opts[arch]
            torch.random.set_rng_state(rng_states[arch])
            model.train()
            logits = model(
                train.tokens,
                forced_active_mask=train.forced_active_mask,
                context_owner=owner,
            ).logits
            loss = F.cross_entropy(logits, train.targets)
            rng_states[arch] = torch.random.get_rng_state().clone()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss {cid}/{seed}/{arch}/step{step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise FloatingPointError(f"nonfinite gradient {cid}/{seed}/{arch}/step{step}")
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not bool(torch.isfinite(torch.as_tensor(gn))):
                raise FloatingPointError(f"nonfinite grad norm {cid}/{seed}/{arch}/step{step}")
            opt.step()

    rows: list[dict[str, Any]] = []
    for arch in ARCHS:
        if best_state[arch] is None or best_metrics[arch] is None or best_step[arch] is None:
            raise RuntimeError(f"missing selected checkpoint {cid}/{seed}/{arch}")
        models[arch].load_state_dict(best_state[arch], strict=True)
        selected = evaluate(models[arch], val)
        ckpt = run_root / "checkpoints" / cid / arch / f"seed_{seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "wbb-sgw-native-scope-availability-checkpoint-v0.1",
                "cell": cid,
                "scope": scope,
                "availability": availability,
                "architecture": arch,
                "seed": int(seed),
                "selected_step": int(best_step[arch]),
                "selected_validation": selected,
                "state_dict": best_state[arch],
            },
            ckpt,
        )
        rows.append(
            {
                "cell": cid,
                "scope": scope,
                "availability": availability,
                "architecture": arch,
                "seed": int(seed),
                "finite": True,
                "completed_optimizer_steps": TRAIN_STEPS,
                "selected_step": int(best_step[arch]),
                "selected_validation": selected,
                "history": histories[arch],
                "checkpoint_path": str(ckpt),
            }
        )

    return {
        "cell": cid,
        "scope": scope,
        "availability": availability,
        "seed": int(seed),
        "rows": rows,
        "group_wall_seconds": time.time() - started,
    }


def launch_group(run_root: Path, scope: str, availability: str, seed: int) -> dict[str, Any]:
    cid = cell_id(scope, availability)
    out = run_root / "group_results" / f"{cid}__seed_{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    group_root = run_root / "groups" / f"{cid}__seed_{seed}"
    group_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--group-unit",
        "--run-root", str(group_root),
        "--unit-output", str(out),
        "--scope", scope,
        "--availability", availability,
        "--seed", str(seed),
    ]
    p = subprocess.run(cmd, cwd=Path.cwd(), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"group failed {cid}/{seed} rc={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-8000:]}"
        )
    return json.loads(out.read_text(encoding="utf-8"))


def aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for group in groups for row in group["rows"]]
    idx = {
        (row["cell"], int(row["seed"]), row["architecture"]): row
        for row in rows
    }

    def acc(cid: str, seed: int, arch: str) -> float:
        return float(idx[(cid, int(seed), arch)]["selected_validation"]["joint_accuracy"])

    summaries: dict[str, Any] = {}
    for scope, availability in CELL_SPECS:
        cid = cell_id(scope, availability)
        summaries[cid] = {}
        for arch in ARCHS:
            vals = {str(seed): acc(cid, seed, arch) for seed in QUALIFICATION_SEEDS}
            summaries[cid][arch] = {
                "seed_joint_accuracy": vals,
                "mean_joint_accuracy": sum(vals.values()) / len(vals),
            }
        summaries[cid]["paired"] = {
            "P_minus_L": {
                str(seed): acc(cid, seed, PERSISTENT) - acc(cid, seed, RCL340)
                for seed in QUALIFICATION_SEEDS
            },
            "P_minus_R": {
                str(seed): acc(cid, seed, PERSISTENT) - acc(cid, seed, RESET)
                for seed in QUALIFICATION_SEEDS
            },
            "L_minus_P": {
                str(seed): acc(cid, seed, RCL340) - acc(cid, seed, PERSISTENT)
                for seed in QUALIFICATION_SEEDS
            },
        }

    n_cid = cell_id(PRIVATE, CURRENT)
    l_cid = cell_id(PRIVATE, PAST_ONLY)
    c_cid = cell_id(SHARED, CURRENT)
    p_cid = cell_id(SHARED, PAST_ONLY)

    n_checks: dict[str, bool] = {}
    l_checks: dict[str, bool] = {}
    c_checks: dict[str, bool] = {}
    p_checks: dict[str, bool] = {}
    for seed in QUALIFICATION_SEEDS:
        n_checks[str(seed)] = bool(
            acc(n_cid, seed, PERSISTENT) >= 0.85
            and acc(n_cid, seed, RESET) >= 0.85
            and acc(n_cid, seed, RCL340) >= 0.85
            and acc(n_cid, seed, NOCOMM) <= 0.35
        )
        l_checks[str(seed)] = bool(
            acc(l_cid, seed, RCL340) >= 0.90
            and acc(l_cid, seed, NOCOMM) <= 0.35
        )
        pp = acc(c_cid, seed, PERSISTENT)
        rr = acc(c_cid, seed, RESET)
        c_checks[str(seed)] = bool(
            pp >= 0.85
            and rr >= 0.85
            and acc(c_cid, seed, RCL340) >= 0.85
            and acc(c_cid, seed, NOCOMM) <= 0.35
            and abs(pp - rr) <= 0.10
        )
        pp = acc(p_cid, seed, PERSISTENT)
        rr = acc(p_cid, seed, RESET)
        ll = acc(p_cid, seed, RCL340)
        p_checks[str(seed)] = bool(
            pp >= 0.80
            and (pp - rr) >= 0.15
            and (pp - ll) >= 0.15
            and acc(p_cid, seed, NOCOMM) <= 0.35
        )

    gates = {
        "N_PRIVATE_CURRENT": {"pass": all(n_checks.values()), "seed_checks": n_checks},
        "L_PRIVATE_PAST": {"pass": all(l_checks.values()), "seed_checks": l_checks},
        "C_SHARED_CURRENT": {"pass": all(c_checks.values()), "seed_checks": c_checks},
        "P_SHARED_PAST": {"pass": all(p_checks.values()), "seed_checks": p_checks},
    }
    gates["overall_pass"] = bool(all(g["pass"] for g in gates.values()))
    return {"rows": rows, "summaries": summaries, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--group-unit", action="store_true")
    ap.add_argument("--unit-output")
    ap.add_argument("--scope")
    ap.add_argument("--availability")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    runtime_assertions()
    run_root = Path(args.run_root).resolve()

    if args.group_unit:
        if args.scope is None or args.availability is None or args.seed is None or args.unit_output is None:
            raise RuntimeError("group-unit arguments missing")
        result = train_group(
            scope=args.scope,
            availability=args.availability,
            seed=int(args.seed),
            run_root=run_root,
        )
        atomic_json(Path(args.unit_output), result)
        return

    if not args.repo_root:
        raise RuntimeError("--repo-root required for aggregate run")
    repo_root = Path(args.repo_root).resolve()
    state_path = repo_root / STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    q0 = json.loads((repo_root / Q0_REPORT).read_text(encoding="utf-8"))
    if state["status"] != "Q0_PASS__32_UNIT_QUALIFICATION_AUTHORIZED":
        raise RuntimeError(f"unexpected state: {state['status']}")
    if not q0["decision"]["gate_pass"] or not state["training_authorized"]:
        raise RuntimeError("Q0 did not authorize training")
    if state["test_gate"] != "CLOSED" or state["test_model_forward_count"] != 0:
        raise RuntimeError("Test governance changed")

    jobs = [
        (scope, availability, seed)
        for scope, availability in CELL_SPECS
        for seed in QUALIFICATION_SEEDS
    ]
    groups: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(launch_group, run_root, scope, availability, seed): (scope, availability, seed)
            for scope, availability, seed in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            groups.append(fut.result())

    result = aggregate(groups)
    expected_groups = 8
    expected_units = 32
    if len(groups) != expected_groups or len(result["rows"]) != expected_units:
        raise RuntimeError("scientific unit count mismatch")

    status = (
        "QUALIFICATION_PASS__HUMAN_REVIEW_REQUIRED"
        if result["gates"]["overall_pass"]
        else "QUALIFICATION_FAIL__LARGER_MAP_NOT_AUTHORIZED"
    )
    report = {
        "schema_version": "wbb-sgw-native-scope-availability-qualification-v0.1",
        "status": status,
        "development_only": True,
        "cells": [cell_id(s, a) for s, a in CELL_SPECS],
        "seeds": list(QUALIFICATION_SEEDS),
        "execution": {
            "groups": len(groups),
            "scientific_units": len(result["rows"]),
            "expected_groups": expected_groups,
            "expected_scientific_units": expected_units,
            "engineering_failures": 0,
            "wall_seconds": time.time() - started,
        },
        "gates": result["gates"],
        "summaries": result["summaries"],
        "rows": result["rows"],
        "decision": {
            "larger_map_authorized": False,
            "automatic_task_revision_authorized": False,
            "architecture_change_authorized": False,
            "test_gate": "CLOSED",
            "test_model_forward_count": 0,
        },
    }
    atomic_json(repo_root / REPORT, report)
    atomic_json(
        repo_root / PROGRESS,
        {
            "schema_version": "wbb-sgw-native-scope-availability-progress-v0.1",
            "status": "COMPLETE",
            "groups": len(groups),
            "scientific_units": len(result["rows"]),
            "overall_pass": bool(result["gates"]["overall_pass"]),
        },
    )
    state["qualification_complete"] = True
    state["qualification_pass"] = bool(result["gates"]["overall_pass"])
    state["qualification_report"] = REPORT
    state["training_authorized"] = False
    state["larger_map_authorized"] = False
    state["status"] = status
    atomic_json(state_path, state)
    print(json.dumps({"status": status, "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
