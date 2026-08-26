"""SGW-native task-state scope × temporal availability qualification task.

The task is frozen in
`docs/contracts/SGW_NATIVE_SCOPE_AVAILABILITY_QUALIFICATION_PROTOCOL_v0.1.md`.
It changes only environmental information geometry and never edits the canonical
Task-v2.2 clean architecture family.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import torch

NUM_PROCESSORS = 6
NUM_CYCLES = 4
TOKEN_WIDTH = 46
ACTIVE_COUNT = 4
CONTEXT_SOURCE = 0
EVIDENCE_SOURCE = 1
READOUT_PROCESSOR = 5
FILLERS = (2, 3, 4)

PRIVATE = "PRIVATE"
SHARED = "SHARED"
CURRENT = "CURRENT"
PAST_ONLY = "PAST_ONLY"
SCOPES = (PRIVATE, SHARED)
AVAILABILITIES = (CURRENT, PAST_ONLY)
QUALIFICATION_SEEDS = (20275801, 20275802)

# 46D deterministic layout.
SELECTOR_PRESENT = 0
SELECTOR_BIT0 = slice(1, 3)
SELECTOR_BIT1 = slice(3, 5)
EVIDENCE_PRESENT = 5
EVIDENCE_BIT0 = slice(6, 8)
EVIDENCE_BIT1 = slice(8, 10)
EVIDENCE_PHASE = slice(10, 13)  # NONE, U, V
CYCLE_ID = slice(13, 17)
PROCESSOR_ID = slice(17, 23)
ROLE_ID = slice(23, 27)  # context, evidence, consumer, filler
ZERO_PADDING = slice(27, 46)


@dataclass(frozen=True)
class ScopeAvailabilityBatch:
    tokens: torch.Tensor                 # [B,4,6,46]
    forced_active_mask: torch.Tensor     # [B,4,6]
    targets: torch.Tensor                # [B]
    selectors: torch.Tensor              # [B,2]
    u_bits: torch.Tensor                 # [B,2]
    v_bits: torch.Tensor                 # [B,2]
    scope: str
    availability: str
    metadata: dict[str, Any]


def _derive(seed: int, tag: str) -> int:
    payload = f"sgw-native-scope-availability-v0.1:{int(seed)}:{tag}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _validate(scope: str, availability: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"invalid scope: {scope}")
    if availability not in AVAILABILITIES:
        raise ValueError(f"invalid availability: {availability}")


def cell_id(scope: str, availability: str) -> str:
    _validate(scope, availability)
    return f"{scope}_{'CURRENT' if availability == CURRENT else 'PAST'}"


def context_owner(scope: str) -> int:
    if scope == PRIVATE:
        return READOUT_PROCESSOR
    if scope == SHARED:
        return CONTEXT_SOURCE
    raise ValueError(scope)


def _bits_from_class(value: int) -> tuple[int, int]:
    return ((int(value) >> 1) & 1, int(value) & 1)


def _set_common_context(token: torch.Tensor, *, cycle: int, processor: int) -> None:
    token[CYCLE_ID.start + int(cycle)] = 1.0
    token[PROCESSOR_ID.start + int(processor)] = 1.0
    if processor == CONTEXT_SOURCE:
        role = 0
    elif processor == EVIDENCE_SOURCE:
        role = 1
    elif processor == READOUT_PROCESSOR:
        role = 2
    else:
        role = 3
    token[ROLE_ID.start + role] = 1.0

    phase = 0 if cycle in (0, 3) else (1 if cycle == 1 else 2)
    token[EVIDENCE_PHASE.start + phase] = 1.0


def _write_selector(token: torch.Tensor, selector: tuple[int, int]) -> None:
    token[SELECTOR_PRESENT] = 1.0
    token[SELECTOR_BIT0.start + int(selector[0])] = 1.0
    token[SELECTOR_BIT1.start + int(selector[1])] = 1.0


def _write_evidence(token: torch.Tensor, bits: tuple[int, int]) -> None:
    token[EVIDENCE_PRESENT] = 1.0
    token[EVIDENCE_BIT0.start + int(bits[0])] = 1.0
    token[EVIDENCE_BIT1.start + int(bits[1])] = 1.0


def _balanced_selector_target_pairs(count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count % 16 != 0:
        raise ValueError("count must be a positive multiple of 16 for exact selector×target balance")
    pairs = [(s, y) for s in range(4) for y in range(4)] * (count // 16)
    rng.shuffle(pairs)
    return pairs


def build_batch(
    *,
    scope: str,
    availability: str,
    experimental_seed: int,
    count: int,
    tag: str,
) -> ScopeAvailabilityBatch:
    scope = str(scope)
    availability = str(availability)
    _validate(scope, availability)
    if count <= 0:
        raise ValueError("count must be positive")

    cid = cell_id(scope, availability)
    schedule_seed = _derive(experimental_seed, f"{cid}:{tag}:schedule")
    value_seed = _derive(experimental_seed, f"{cid}:{tag}:values")
    schedule_rng = random.Random(schedule_seed)
    value_rng = random.Random(value_seed)
    selector_target = _balanced_selector_target_pairs(count, schedule_rng)

    tokens = torch.zeros((count, NUM_CYCLES, NUM_PROCESSORS, TOKEN_WIDTH), dtype=torch.float32)
    active = torch.zeros((count, NUM_CYCLES, NUM_PROCESSORS), dtype=torch.bool)
    targets = torch.empty(count, dtype=torch.long)
    selectors = torch.empty((count, 2), dtype=torch.long)
    u_tensor = torch.empty((count, 2), dtype=torch.long)
    v_tensor = torch.empty((count, 2), dtype=torch.long)

    owner = context_owner(scope)

    for row, (selector_class, target_class) in enumerate(selector_target):
        selector = _bits_from_class(selector_class)
        target_bits = _bits_from_class(target_class)

        u = [value_rng.randrange(2), value_rng.randrange(2)]
        v = [value_rng.randrange(2), value_rng.randrange(2)]
        for bit in range(2):
            if selector[bit] == 0:
                u[bit] = target_bits[bit]
            else:
                v[bit] = target_bits[bit]

        for cycle in range(NUM_CYCLES):
            for processor in range(NUM_PROCESSORS):
                _set_common_context(tokens[row, cycle, processor], cycle=cycle, processor=processor)

        # Frozen active geometry.
        c0 = (owner, 2, 3, 4)
        if scope == SHARED and availability == CURRENT:
            c12 = (0, 1, 5, 2)
        else:
            c12 = (1, 5, 2, 3)
        c3 = (5, 2, 3, 4)
        for processor in c0:
            active[row, 0, processor] = True
        for cycle in (1, 2):
            for processor in c12:
                active[row, cycle, processor] = True
        for processor in c3:
            active[row, 3, processor] = True

        selector_cycles = (0, 1, 2) if availability == CURRENT else (0,)
        for cycle in selector_cycles:
            _write_selector(tokens[row, cycle, owner], selector)

        _write_evidence(tokens[row, 1, EVIDENCE_SOURCE], (u[0], u[1]))
        _write_evidence(tokens[row, 2, EVIDENCE_SOURCE], (v[0], v[1]))

        targets[row] = int(target_class)
        selectors[row] = torch.tensor(selector, dtype=torch.long)
        u_tensor[row] = torch.tensor(u, dtype=torch.long)
        v_tensor[row] = torch.tensor(v, dtype=torch.long)

    if tokens.shape != (count, 4, 6, 46):
        raise AssertionError("unexpected token geometry")
    if not bool(torch.all(active.sum(-1) == ACTIVE_COUNT)):
        raise AssertionError("exactly four active processors required each cycle")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError("nonfinite task token")
    if float(tokens[..., ZERO_PADDING].abs().max()) != 0.0:
        raise AssertionError("zero padding must remain exact zero")

    return ScopeAvailabilityBatch(
        tokens=tokens,
        forced_active_mask=active,
        targets=targets,
        selectors=selectors,
        u_bits=u_tensor,
        v_bits=v_tensor,
        scope=scope,
        availability=availability,
        metadata={
            "cell": cid,
            "scope": scope,
            "availability": availability,
            "context_owner": int(owner),
            "experimental_seed": int(experimental_seed),
            "tag": str(tag),
            "schedule_seed": int(schedule_seed),
            "value_seed": int(value_seed),
            "count": int(count),
            "token_layout": "SGW_NATIVE_SCOPE_AVAILABILITY_46D_v0.1",
        },
    )


def symbolic_oracle(batch: ScopeAvailabilityBatch) -> torch.Tensor:
    selected = torch.where(batch.selectors == 0, batch.u_bits, batch.v_bits)
    return 2 * selected[:, 0] + selected[:, 1]


def selector_target_table(batch: ScopeAvailabilityBatch) -> torch.Tensor:
    selector_class = 2 * batch.selectors[:, 0] + batch.selectors[:, 1]
    table = torch.zeros((4, 4), dtype=torch.long)
    for s in range(4):
        for y in range(4):
            table[s, y] = int(((selector_class == s) & (batch.targets == y)).sum())
    return table


def p5_external_lookup_accuracy(batch: ScopeAvailabilityBatch) -> float:
    # Within PRIVATE cells the only sample-varying task information externally
    # visible at p5 is the selector; in SHARED cells there is no task content at p5.
    # The generator cross-balances selector×target exactly, so the empirical best
    # target lookup from p5 external task information is exactly chance.
    if batch.scope == PRIVATE:
        groups = 2 * batch.selectors[:, 0] + batch.selectors[:, 1]
    else:
        groups = torch.zeros_like(batch.targets)
    correct = 0
    for g in torch.unique(groups).tolist():
        idx = (groups == int(g)).nonzero(as_tuple=False).reshape(-1)
        counts = torch.bincount(batch.targets[idx], minlength=4)
        correct += int(counts.max())
    return correct / float(batch.targets.numel())


__all__ = [
    "ACTIVE_COUNT",
    "AVAILABILITIES",
    "CONTEXT_SOURCE",
    "CURRENT",
    "EVIDENCE_BIT0",
    "EVIDENCE_BIT1",
    "EVIDENCE_PRESENT",
    "EVIDENCE_SOURCE",
    "PAST_ONLY",
    "PRIVATE",
    "QUALIFICATION_SEEDS",
    "READOUT_PROCESSOR",
    "SCOPES",
    "SELECTOR_BIT0",
    "SELECTOR_BIT1",
    "SELECTOR_PRESENT",
    "SHARED",
    "ScopeAvailabilityBatch",
    "TOKEN_WIDTH",
    "ZERO_PADDING",
    "build_batch",
    "cell_id",
    "context_owner",
    "p5_external_lookup_accuracy",
    "selector_target_table",
    "symbolic_oracle",
]
