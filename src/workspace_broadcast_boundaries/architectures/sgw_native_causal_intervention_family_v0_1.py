"""Forward-only causal intervention host for qualified SGW-native checkpoints.

No trainable parameter is added. Canonical clean modules are inherited unchanged.
The only intervention controls are:
1) remove cross-cycle Persistent workspace state after cycle 0 by invoking the
   canonical reset semantics on later cycles;
2) mask already-computed workspace broadcast messages at specified receivers.
The SGW-native cycle-0 ownership firewall is preserved exactly.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from workspace_broadcast_boundaries.architectures.task_v2_2_clean_family_v0_1 import (
    NUM_SPECIALISTS,
    PERSISTENT,
    RESET,
    SPECIALIST_WIDTH,
    TaskV22CleanFamilyModel,
    TaskV22CleanFamilyOutput,
)


class SGWNativeCausalInterventionModel(TaskV22CleanFamilyModel):
    """Canonical Persistent/Reset SGW under frozen forward-only interventions."""

    def __init__(
        self,
        architecture: str,
        *,
        experimental_seed: int,
        blind_receiver_head: bool,
    ) -> None:
        if architecture not in (PERSISTENT, RESET):
            raise ValueError("causal intervention model supports only Persistent/Reset SGW")
        super().__init__(
            architecture,
            experimental_seed=experimental_seed,
            blind_receiver_head=blind_receiver_head,
        )

    @staticmethod
    def _receiver_mask_tensor(
        receiver_masks: Sequence[Sequence[int] | None] | None,
        cycle: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if receiver_masks is None:
            return torch.ones(
                (1, NUM_SPECIALISTS, 1),
                device=reference.device,
                dtype=reference.dtype,
            )
        if len(receiver_masks) != 4:
            raise ValueError("receiver_masks must have exactly four cycle entries")
        spec = receiver_masks[cycle]
        if spec is None:
            ids = tuple(range(NUM_SPECIALISTS))
        else:
            ids = tuple(int(x) for x in spec)
            if len(set(ids)) != len(ids) or any(x not in range(NUM_SPECIALISTS) for x in ids):
                raise ValueError(f"invalid receiver ids for cycle {cycle}: {ids}")
        mask = torch.zeros(NUM_SPECIALISTS, device=reference.device, dtype=reference.dtype)
        if ids:
            mask[list(ids)] = 1.0
        return mask.view(1, NUM_SPECIALISTS, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        forced_active_mask: torch.Tensor | None = None,
        context_owner: int,
        remove_persistence_after_cycle0: bool = False,
        receiver_masks: Sequence[Sequence[int] | None] | None = None,
    ) -> TaskV22CleanFamilyOutput:
        if tokens.ndim != 4 or tokens.shape[1:] != (4, NUM_SPECIALISTS, 46):
            raise ValueError("tokens must be [B,4,6,46]")
        if int(context_owner) not in range(NUM_SPECIALISTS):
            raise ValueError("context_owner must be a valid processor id")
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
            forced = None if forced_active_mask is None else forced_active_mask[:, cycle]
            inp = self.input_attention(tokens[:, cycle], h, forced_active_mask=forced)
            hbar, c = self.recurrent(inp.fresh_values, h, c, inp.active_mask)

            assert self.workspace is not None
            reset_this_cycle = bool(
                self.architecture == RESET
                or (remove_persistence_after_cycle0 and cycle > 0)
            )
            raw_received, memory, d = self.workspace(
                inp.fresh_values,
                hbar,
                inp.active_indices,
                memory,
                reset_each_cycle=reset_this_cycle,
            )
            receiver_mask = self._receiver_mask_tensor(receiver_masks, cycle, raw_received)
            received = raw_received * receiver_mask
            h = hbar + received

            firewall_applied = cycle == 0
            owner_h_max = None
            owner_c_max = None
            nonowner_h_max = None
            nonowner_c_max = None
            if firewall_applied:
                keep = torch.zeros(NUM_SPECIALISTS, device=h.device, dtype=h.dtype)
                keep[int(context_owner)] = 1.0
                keep = keep.view(1, NUM_SPECIALISTS, 1)
                h = h * keep
                c = c * keep
                owner_h_max = float(h[:, int(context_owner)].detach().abs().max())
                owner_c_max = float(c[:, int(context_owner)].detach().abs().max())
                nonowner = [k for k in range(NUM_SPECIALISTS) if k != int(context_owner)]
                nonowner_h_max = float(h[:, nonowner].detach().abs().max())
                nonowner_c_max = float(c[:, nonowner].detach().abs().max())

            fresh_hist.append(inp.fresh_values)
            mask_hist.append(inp.active_mask)
            hbar_hist.append(hbar)
            recv_hist.append(received)
            extra = dict(d)
            extra.update(
                {
                    "architecture": self.architecture,
                    "cycle": cycle,
                    "activation_scores": inp.activation_scores,
                    "active_indices": inp.active_indices,
                    "active_mask": inp.active_mask,
                    "fresh_values_A": inp.fresh_values,
                    "scope_availability_ownership_firewall_applied": firewall_applied,
                    "scope_availability_context_owner": int(context_owner),
                    "scope_availability_post_firewall_owner_h_max_abs": owner_h_max,
                    "scope_availability_post_firewall_owner_c_max_abs": owner_c_max,
                    "scope_availability_post_firewall_nonowner_h_max_abs": nonowner_h_max,
                    "scope_availability_post_firewall_nonowner_c_max_abs": nonowner_c_max,
                    "causal_remove_persistence_after_cycle0": bool(remove_persistence_after_cycle0),
                    "causal_workspace_reset_this_cycle": reset_this_cycle,
                    "causal_receiver_ids": tuple(
                        int(i)
                        for i in range(NUM_SPECIALISTS)
                        if float(receiver_mask[0, i, 0]) == 1.0
                    ),
                    "causal_raw_workspace_received": raw_received,
                    "causal_effective_workspace_received": received,
                    "canonical_architecture_parameters_modified": False,
                    "causal_scope": "SGW_NATIVE_CAUSAL_DECOMPOSITION_v0.1",
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


__all__ = ["SGWNativeCausalInterventionModel"]
