"""Architecture-neutral ownership-firewall host for SGW-native qualification.

Canonical Task-v2.2 clean modules are imported unchanged. This wrapper adds one
task-level state operation after cycle 0: keep the frozen context owner's h/c
and zero every other processor recurrent state. No trainable parameter is added.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import torch

from workspace_broadcast_boundaries.architectures.task_v2_2_clean_family_v0_1 import (
    NOCOMM,
    NUM_SPECIALISTS,
    PERSISTENT,
    RCL340,
    RESET,
    SPECIALIST_WIDTH,
    TaskV22CleanFamilyModel,
    TaskV22CleanFamilyOutput,
)


class SGWNativeScopeAvailabilityModel(TaskV22CleanFamilyModel):
    """Canonical clean model under the frozen cycle-0 ownership firewall."""

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        forced_active_mask: torch.Tensor | None = None,
        context_owner: int,
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
            elif self.architecture == NOCOMM:
                h = hbar
            else:
                raise RuntimeError(f"unexpected architecture {self.architecture}")

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
                    "canonical_architecture_equations_modified": False,
                    "scope_availability_instrument": "SGW_NATIVE_SCOPE_X_AVAILABILITY_v0.1",
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


def canonical_parameter_identity(architecture: str, seed: int) -> dict[str, Any]:
    base = TaskV22CleanFamilyModel(
        architecture, experimental_seed=seed, blind_receiver_head=True
    )
    wrapped = SGWNativeScopeAvailabilityModel(
        architecture, experimental_seed=seed, blind_receiver_head=True
    )
    b = base.state_dict()
    w = wrapped.state_dict()
    names_equal = tuple(b) == tuple(w)
    shapes_equal = names_equal and all(b[n].shape == w[n].shape for n in b)
    max_diff = 0.0
    if names_equal and shapes_equal:
        for name in b:
            if b[name].numel():
                max_diff = max(
                    max_diff,
                    float((b[name].detach().cpu() - w[name].detach().cpu()).abs().max()),
                )
    return {
        "architecture": architecture,
        "seed": int(seed),
        "names_equal": names_equal,
        "shapes_equal": shapes_equal,
        "parameter_tensor_count": len(b),
        "max_initial_tensor_abs_diff": max_diff,
        "pass": bool(names_equal and shapes_equal and max_diff == 0.0),
    }


__all__ = ["SGWNativeScopeAvailabilityModel", "canonical_parameter_identity"]
