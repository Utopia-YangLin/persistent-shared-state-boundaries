"""Standalone canonical Task-v2.2 four-model family after frozen pruning.

Scientific identities are fixed by the completed pruning chain:

- Local primary: RCL340 with the latent candidate MLP removed (MLP0).
- Global primary: Innovation Persistent-SGW with one shared write-MLP application (MLP1).
- Global control: matched Innovation Reset-SGW with MLP1.
- Negative control: NoComm.

This module intentionally has no dependency on historical/candidate/pruning
architecture modules. It materializes only the common host and the selected
communication mechanisms so the final active dependency graph is clean.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

NUM_SPECIALISTS = 6
SPECIALIST_WIDTH = 85
LOCAL_TOKEN_WIDTH = 46
ACTIVE_TOPK = 4
INPUT_HEADS = 4
INPUT_KEY_WIDTH = 64
INPUT_VALUE_WIDTH = 85
INPUT_DROPOUT = 0.1
WORKSPACE_SLOTS = 4
WRITE_KEY_WIDTH = 32
BROADCAST_HEADS = 4
BROADCAST_KEY_WIDTH = 32
BROADCAST_VALUE_WIDTH = 32
BROADCAST_ATTENDED_WIDTH = BROADCAST_HEADS * BROADCAST_VALUE_WIDTH
NUM_CLASSES = 4
LATENT_WIDTH = ACTIVE_TOPK * SPECIALIST_WIDTH

PERSISTENT = "sgw_persistent_innovation_clean"
RESET = "sgw_reset_innovation_clean"
RCL340 = "rcl340_local_clean"
NOCOMM = "nocomm_clean"
ARCHS = (PERSISTENT, RESET, RCL340, NOCOMM)

GLOBAL_WRITE_MLP_REPEATS = 1
LOCAL_LATENT_MLP_REPEATS = 0


def _derive_common(seed: int, tag: str) -> int:
    payload = f"task-v2.1-phase0-native-rim-v0.1:{seed}:{tag}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _derive_rcl340(seed: int, tag: str) -> int:
    payload = f"task-v2.1-rcl340-v0.1:{seed}:{tag}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _linear_common(
    seed: int, tag: str, din: int, dout: int, *, bias: bool = True
) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_derive_common(seed, tag))
        return nn.Linear(din, dout, bias=bias)


def _linear_rcl340(
    seed: int, tag: str, din: int, dout: int, *, bias: bool = True
) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_derive_rcl340(seed, tag))
        return nn.Linear(din, dout, bias=bias)


def _lstm_common(seed: int, tag: str, din: int, dh: int) -> nn.LSTMCell:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_derive_common(seed, tag))
        return nn.LSTMCell(din, dh)


class SpecialistGroupedLinear(nn.Module):
    """Per-specialist projection used by the frozen common input-attention host."""

    def __init__(self, seed: int, tag: str, din: int, dout: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derive_common(seed, tag))
            self.weight = nn.Parameter(0.01 * torch.randn(NUM_SPECIALISTS, din, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != NUM_SPECIALISTS:
            raise ValueError("grouped input must be [B,6,D]")
        return torch.einsum("bsd,sdo->bso", x, self.weight)


@dataclass(frozen=True)
class InputAttentionOutput:
    fresh_values: torch.Tensor
    activation_scores: torch.Tensor
    active_mask: torch.Tensor
    active_indices: torch.Tensor
    local_probabilities: torch.Tensor


class CleanRIMLocalInputAttention(nn.Module):
    """Exact common-host four-head local-vs-null RIM input attention."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.query = SpecialistGroupedLinear(
            seed, "input:q", SPECIALIST_WIDTH, INPUT_HEADS * INPUT_KEY_WIDTH
        )
        self.key = _linear_common(
            seed,
            "input:k",
            LOCAL_TOKEN_WIDTH,
            INPUT_HEADS * INPUT_KEY_WIDTH,
            bias=False,
        )
        self.value = _linear_common(
            seed, "input:v", LOCAL_TOKEN_WIDTH, INPUT_VALUE_WIDTH, bias=False
        )
        self.dropout = nn.Dropout(INPUT_DROPOUT)

    @staticmethod
    def _validate(tokens: torch.Tensor, h_prev: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            NUM_SPECIALISTS,
            LOCAL_TOKEN_WIDTH,
        ):
            raise ValueError("tokens must be [B,6,46]")
        if h_prev.shape != (tokens.shape[0], NUM_SPECIALISTS, SPECIALIST_WIDTH):
            raise ValueError("h_prev must be [B,6,85]")

    @staticmethod
    def _forced_indices(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2 or mask.shape[1] != NUM_SPECIALISTS:
            raise ValueError("forced_active_mask must be [B,6]")
        mask = mask.to(dtype=torch.bool)
        if not bool(torch.all(mask.sum(-1) == ACTIVE_TOPK)):
            raise ValueError("forced_active_mask must select exactly four specialists")
        ids = torch.arange(NUM_SPECIALISTS, device=mask.device, dtype=torch.float32)
        score = mask.float() * 2.0 - ids.view(1, -1) * 1e-4
        return torch.topk(
            score, k=ACTIVE_TOPK, dim=-1, largest=True, sorted=True
        ).indices

    def forward(
        self,
        tokens: torch.Tensor,
        h_prev: torch.Tensor,
        *,
        forced_active_mask: torch.Tensor | None = None,
    ) -> InputAttentionOutput:
        self._validate(tokens, h_prev)
        b = tokens.shape[0]
        token_for_attn = self.dropout(tokens) if self.training else tokens
        q = self.query(h_prev).reshape(
            b, NUM_SPECIALISTS, INPUT_HEADS, INPUT_KEY_WIDTH
        )
        k_local = self.key(token_for_attn).reshape(
            b, NUM_SPECIALISTS, INPUT_HEADS, INPUT_KEY_WIDTH
        )
        local_logits = (q * k_local).sum(-1) / math.sqrt(INPUT_KEY_WIDTH)
        logits = torch.stack((torch.zeros_like(local_logits), local_logits), dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., 1]
        activation = probs.mean(dim=-1)
        fresh = activation.unsqueeze(-1) * self.value(token_for_attn)

        if forced_active_mask is None:
            indices = torch.topk(
                activation, k=ACTIVE_TOPK, dim=-1, largest=True, sorted=True
            ).indices
            active = torch.zeros_like(activation, dtype=torch.bool)
            active.scatter_(1, indices, True)
        else:
            forced = forced_active_mask.to(device=tokens.device, dtype=torch.bool)
            indices = self._forced_indices(forced)
            active = forced

        return InputAttentionOutput(
            fresh_values=fresh,
            activation_scores=activation,
            active_mask=active,
            active_indices=indices,
            local_probabilities=probs,
        )


class CleanRIMBlockLSTM(nn.Module):
    """Exact common-host six independent 85D LSTMCells."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [
                _lstm_common(
                    seed, f"block-lstm:{k}", SPECIALIST_WIDTH, SPECIALIST_WIDTH
                )
                for k in range(NUM_SPECIALISTS)
            ]
        )

    def forward(
        self,
        fresh: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fresh.shape != h_prev.shape or h_prev.shape != c_prev.shape:
            raise ValueError("fresh/h/c must have the same [B,6,85] shape")
        if active_mask.shape != h_prev.shape[:2]:
            raise ValueError("active_mask must be [B,6]")
        hs: list[torch.Tensor] = []
        cs: list[torch.Tensor] = []
        for k, cell in enumerate(self.cells):
            hk, ck = cell(fresh[:, k], (h_prev[:, k], c_prev[:, k]))
            hs.append(hk)
            cs.append(ck)
        h_candidate = torch.stack(hs, dim=1)
        c_candidate = torch.stack(cs, dim=1)
        m = active_mask.unsqueeze(-1).to(dtype=h_prev.dtype)
        hbar = m * h_candidate + (1.0 - m) * h_prev
        cbar = m * c_candidate + (1.0 - m) * c_prev
        return hbar, cbar


class CleanInnovationSharedWorkspace(nn.Module):
    """Selected Innovation-SGW with exactly one candidate MLP application."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.write_q = _linear_common(
            seed, "sgw:write:q", SPECIALIST_WIDTH, WRITE_KEY_WIDTH
        )
        self.write_k = _linear_common(
            seed, "sgw:write:k", SPECIALIST_WIDTH, WRITE_KEY_WIDTH
        )
        self.write_v = _linear_common(
            seed, "sgw:write:v", SPECIALIST_WIDTH, SPECIALIST_WIDTH
        )
        self.write_ln1 = nn.LayerNorm(SPECIALIST_WIDTH)
        self.write_ln2 = nn.LayerNorm(SPECIALIST_WIDTH)
        self.write_mlp = _linear_common(
            seed, "sgw:write:mlp", SPECIALIST_WIDTH, SPECIALIST_WIDTH
        )
        self.input_gate = _linear_common(
            seed, "sgw:gate:input", SPECIALIST_WIDTH, 2 * SPECIALIST_WIDTH
        )
        self.memory_gate = _linear_common(
            seed, "sgw:gate:memory", SPECIALIST_WIDTH, 2 * SPECIALIST_WIDTH
        )
        self.forget_bias = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.insert_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.broadcast_q = _linear_common(
            seed,
            "sgw:broadcast:q",
            SPECIALIST_WIDTH,
            BROADCAST_HEADS * BROADCAST_KEY_WIDTH,
        )
        self.broadcast_k = _linear_common(
            seed,
            "sgw:broadcast:k",
            SPECIALIST_WIDTH,
            BROADCAST_HEADS * BROADCAST_KEY_WIDTH,
        )
        self.broadcast_v = _linear_common(
            seed,
            "sgw:broadcast:v",
            SPECIALIST_WIDTH,
            BROADCAST_HEADS * BROADCAST_VALUE_WIDTH,
        )
        self.broadcast_out = _linear_common(
            seed,
            "sgw:broadcast:out",
            BROADCAST_ATTENDED_WIDTH,
            SPECIALIST_WIDTH,
        )

    @staticmethod
    def initial_memory(reference: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(WORKSPACE_SLOTS, device=reference.device, dtype=reference.dtype)
        padded = F.pad(eye, (0, SPECIALIST_WIDTH - WORKSPACE_SLOTS))
        return padded.unsqueeze(0).expand(reference.shape[0], -1, -1).clone()

    @staticmethod
    def _gather_active(fresh: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, fresh.shape[-1])
        return torch.gather(fresh, 1, gather_idx)

    def forward(
        self,
        fresh: torch.Tensor,
        hbar: torch.Tensor,
        active_indices: torch.Tensor,
        previous_memory: torch.Tensor | None,
        *,
        reset_each_cycle: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        initial = self.initial_memory(hbar)
        previous = (
            initial if reset_each_cycle or previous_memory is None else previous_memory
        )
        active_fresh = self._gather_active(fresh, active_indices)

        q = self.write_q(previous)
        k = self.write_k(active_fresh)
        v = self.write_v(active_fresh)
        write_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            WRITE_KEY_WIDTH
        )
        write_weights = torch.softmax(write_scores, dim=-1)
        innovation = torch.matmul(write_weights, v)

        residual = self.write_ln1(innovation)
        mlp = F.relu(self.write_mlp(residual))
        candidate = self.write_ln2(residual + mlp)

        pooled = active_fresh.mean(dim=1)
        gates = self.memory_gate(torch.tanh(previous)) + self.input_gate(
            pooled
        ).unsqueeze(1)
        insert, forget = torch.chunk(gates, 2, dim=-1)
        insert = torch.sigmoid(insert + self.insert_bias)
        forget = torch.sigmoid(forget + self.forget_bias)
        updated = insert * torch.tanh(candidate) + forget * previous

        b = hbar.shape[0]
        bq = self.broadcast_q(hbar).reshape(
            b, NUM_SPECIALISTS, BROADCAST_HEADS, BROADCAST_KEY_WIDTH
        ).permute(0, 2, 1, 3)
        bk = self.broadcast_k(updated).reshape(
            b, WORKSPACE_SLOTS, BROADCAST_HEADS, BROADCAST_KEY_WIDTH
        ).permute(0, 2, 1, 3)
        bv = self.broadcast_v(updated).reshape(
            b, WORKSPACE_SLOTS, BROADCAST_HEADS, BROADCAST_VALUE_WIDTH
        ).permute(0, 2, 1, 3)
        bscore = torch.matmul(bq, bk.transpose(-1, -2)) / math.sqrt(
            BROADCAST_KEY_WIDTH
        )
        bweights = torch.softmax(bscore, dim=-1)
        battended = (
            torch.matmul(bweights, bv)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(b, NUM_SPECIALISTS, BROADCAST_ATTENDED_WIDTH)
        )
        received = self.broadcast_out(battended)

        return received, updated, MappingProxyType(
            {
                "workspace_previous": previous,
                "workspace_active_fresh": active_fresh,
                "workspace_innovation": innovation,
                "workspace_candidate": candidate,
                "workspace_updated": updated,
                "workspace_insert_gate": insert,
                "workspace_forget_gate": forget,
                "workspace_write_weights": write_weights,
                "workspace_broadcast_weights": bweights,
                "workspace_received": received,
                "write_source_kind": "fresh_active_A_only__old_M_query_gate_carry",
                "selected_mlp_repeats": GLOBAL_WRITE_MLP_REPEATS,
            }
        )


class CleanRCL340Composer(nn.Module):
    """Selected RCL340 local composer with no latent candidate MLP."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.sender_proj = _linear_rcl340(
            seed, "rcl340:sender", SPECIALIST_WIDTH, LATENT_WIDTH
        )
        self.receiver_proj = _linear_rcl340(
            seed, "rcl340:receiver", SPECIALIST_WIDTH, LATENT_WIDTH
        )
        self.latent_ln1 = nn.LayerNorm(SPECIALIST_WIDTH)
        self.latent_ln2 = nn.LayerNorm(SPECIALIST_WIDTH)
        self.output = _linear_rcl340(
            seed, "rcl340:output", LATENT_WIDTH, SPECIALIST_WIDTH
        )

    def forward(
        self,
        hbar: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        if hbar.ndim != 3 or hbar.shape[1:] != (
            NUM_SPECIALISTS,
            SPECIALIST_WIDTH,
        ):
            raise ValueError("hbar must be [B,6,85]")
        if active_mask.shape != hbar.shape[:2]:
            raise ValueError("active_mask must be [B,6]")
        counts = active_mask.to(dtype=torch.long).sum(-1)
        if not bool(torch.all(counts == ACTIVE_TOPK)):
            raise ValueError("active_mask must select exactly four specialists")

        b = hbar.shape[0]
        sender = self.sender_proj(hbar)
        receiver = self.receiver_proj(hbar)
        pair = F.relu(receiver.unsqueeze(2) + sender.unsqueeze(1))
        sender_mask = active_mask.to(dtype=pair.dtype).unsqueeze(1).unsqueeze(-1)
        pooled = (pair * sender_mask).sum(dim=2) / float(ACTIVE_TOPK)
        blocks = pooled.reshape(
            b, NUM_SPECIALISTS, ACTIVE_TOPK, SPECIALIST_WIDTH
        )
        residual = self.latent_ln1(blocks)
        candidate = self.latent_ln2(residual)
        flat = candidate.reshape(b, NUM_SPECIALISTS, LATENT_WIDTH)
        message = torch.tanh(self.output(flat))
        message = message * active_mask.unsqueeze(-1).to(dtype=message.dtype)

        return message, MappingProxyType(
            {
                "rcl340_sender_projection": sender,
                "rcl340_receiver_projection": receiver,
                "rcl340_pair_features": pair,
                "rcl340_pooled_latent": pooled,
                "rcl340_latent_blocks": blocks,
                "rcl340_candidate_blocks": candidate,
                "rcl340_message": message,
                "rcl340_pooling": "ACTIVE_SENDER_ARITHMETIC_MEAN",
                "rcl340_sender_order_sensitive": False,
                "rcl340_receiver_private": True,
                "rcl340_shared_state": False,
                "rcl340_cross_cycle_state": False,
                "selected_mlp_repeats": LOCAL_LATENT_MLP_REPEATS,
            }
        )


@dataclass(frozen=True)
class TaskV22CleanFamilyOutput:
    logits: torch.Tensor
    final_h: torch.Tensor
    final_c: torch.Tensor
    input_fresh_history: torch.Tensor
    active_mask_history: torch.Tensor
    hbar_history: torch.Tensor
    received_history: torch.Tensor
    diagnostics: tuple[Mapping[str, Any], ...]


class TaskV22CleanFamilyModel(nn.Module):
    """Canonical standalone four-arm Task-v2.2 factory."""

    def __init__(
        self,
        architecture: str,
        *,
        experimental_seed: int,
        blind_receiver_head: bool,
    ) -> None:
        super().__init__()
        if architecture not in ARCHS:
            raise ValueError(f"unknown clean architecture: {architecture}")
        self.architecture = architecture
        self.experimental_seed = int(experimental_seed)
        self.blind_receiver_head = bool(blind_receiver_head)
        self.input_attention = CleanRIMLocalInputAttention(experimental_seed)
        self.recurrent = CleanRIMBlockLSTM(experimental_seed)
        self.workspace: CleanInnovationSharedWorkspace | None = None
        self.rcl340: CleanRCL340Composer | None = None

        if architecture in (PERSISTENT, RESET):
            self.workspace = CleanInnovationSharedWorkspace(experimental_seed)

        if self.blind_receiver_head:
            self.readout = nn.Sequential(
                _linear_common(
                    experimental_seed, "readout:blind:1", SPECIALIST_WIDTH, 64
                ),
                nn.ReLU(),
                _linear_common(experimental_seed, "readout:blind:2", 64, NUM_CLASSES),
            )
        else:
            self.readout = _linear_common(
                experimental_seed,
                "readout:full",
                NUM_SPECIALISTS * SPECIALIST_WIDTH,
                NUM_CLASSES,
            )

        if architecture == RCL340:
            self.rcl340 = CleanRCL340Composer(experimental_seed)

    @staticmethod
    def common_geometry() -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "specialists": NUM_SPECIALISTS,
                "specialist_width": SPECIALIST_WIDTH,
                "active_topk": ACTIVE_TOPK,
                "input_heads": INPUT_HEADS,
                "input_key_width": INPUT_KEY_WIDTH,
                "input_value_width": INPUT_VALUE_WIDTH,
                "input_dropout": INPUT_DROPOUT,
                "recurrent_cell": "BlockLSTM",
                "workspace_slots": WORKSPACE_SLOTS,
                "workspace_content_width": SPECIALIST_WIDTH,
                "global_write_mlp_repeats": GLOBAL_WRITE_MLP_REPEATS,
                "local_latent_mlp_repeats": LOCAL_LATENT_MLP_REPEATS,
                "broadcast_heads": BROADCAST_HEADS,
                "broadcast_key_width": BROADCAST_KEY_WIDTH,
                "broadcast_value_width": BROADCAST_VALUE_WIDTH,
            }
        )

    def common_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = ("input_attention.", "recurrent.", "readout.")
        return {
            n: v.detach().cpu().clone()
            for n, v in self.state_dict().items()
            if n.startswith(prefixes)
        }

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        forced_active_mask: torch.Tensor | None = None,
    ) -> TaskV22CleanFamilyOutput:
        if tokens.ndim != 4 or tokens.shape[1:] != (
            4,
            NUM_SPECIALISTS,
            LOCAL_TOKEN_WIDTH,
        ):
            raise ValueError("tokens must be [B,4,6,46]")
        b = tokens.shape[0]
        if forced_active_mask is not None and forced_active_mask.shape != (
            b,
            4,
            NUM_SPECIALISTS,
        ):
            raise ValueError("forced_active_mask must be [B,4,6]")

        h = torch.zeros(
            b,
            NUM_SPECIALISTS,
            SPECIALIST_WIDTH,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        c = torch.zeros_like(h)
        memory: torch.Tensor | None = None
        fresh_hist: list[torch.Tensor] = []
        mask_hist: list[torch.Tensor] = []
        hbar_hist: list[torch.Tensor] = []
        recv_hist: list[torch.Tensor] = []
        diags: list[Mapping[str, Any]] = []

        for cycle in range(4):
            forced = (
                None if forced_active_mask is None else forced_active_mask[:, cycle]
            )
            inp = self.input_attention(
                tokens[:, cycle], h, forced_active_mask=forced
            )
            hbar, c = self.recurrent(
                inp.fresh_values, h, c, inp.active_mask
            )
            received = torch.zeros_like(hbar)
            extra: dict[str, Any] = {}

            if self.architecture == RCL340:
                assert self.rcl340 is not None
                received, d = self.rcl340(hbar, inp.active_mask)
                h = hbar + received
                extra.update(dict(d))
            elif self.architecture in (PERSISTENT, RESET):
                assert self.workspace is not None
                received, memory, d = self.workspace(
                    inp.fresh_values,
                    hbar,
                    inp.active_indices,
                    memory,
                    reset_each_cycle=self.architecture == RESET,
                )
                h = hbar + received
                extra.update(dict(d))
            else:
                h = hbar

            fresh_hist.append(inp.fresh_values)
            mask_hist.append(inp.active_mask)
            hbar_hist.append(hbar)
            recv_hist.append(received)
            extra.update(
                {
                    "architecture": self.architecture,
                    "cycle": cycle,
                    "activation_scores": inp.activation_scores,
                    "active_indices": inp.active_indices,
                    "active_mask": inp.active_mask,
                    "fresh_values_A": inp.fresh_values,
                    "clean_scope": "TASK_V2_2_FOUR_MODEL_PRUNING_FAIL_FAST_SCOPE_v0.1",
                }
            )
            diags.append(MappingProxyType(extra))

        logits = (
            self.readout(h[:, 5])
            if self.blind_receiver_head
            else self.readout(h.reshape(b, NUM_SPECIALISTS * SPECIALIST_WIDTH))
        )
        return TaskV22CleanFamilyOutput(
            logits=logits,
            final_h=h,
            final_c=c,
            input_fresh_history=torch.stack(fresh_hist, dim=1),
            active_mask_history=torch.stack(mask_hist, dim=1),
            hbar_history=torch.stack(hbar_hist, dim=1),
            received_history=torch.stack(recv_hist, dim=1),
            diagnostics=tuple(diags),
        )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ACTIVE_TOPK",
    "ARCHS",
    "GLOBAL_WRITE_MLP_REPEATS",
    "LOCAL_LATENT_MLP_REPEATS",
    "LOCAL_TOKEN_WIDTH",
    "NOCOMM",
    "NUM_SPECIALISTS",
    "PERSISTENT",
    "RCL340",
    "RESET",
    "SPECIALIST_WIDTH",
    "CleanInnovationSharedWorkspace",
    "CleanRCL340Composer",
    "TaskV22CleanFamilyModel",
    "TaskV22CleanFamilyOutput",
    "trainable_parameter_count",
]
