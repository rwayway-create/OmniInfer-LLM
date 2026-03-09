# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Experts with shared expert for Granite 4.0 hybrid models."""

import torch
import torch.nn.functional as F
from torch import nn


class GraniteMoEFeedForward(nn.Module):
    """
    Granite 4.0 MoE layer with shared expert.

    Architecture:
    - Router selects top-k experts from num_experts
    - Each expert is a SwiGLU FFN (gate_proj, up_proj, down_proj)
    - A shared expert always contributes to the output
    - Output = weighted sum of routed expert outputs + shared expert output
    """

    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int = 64,
        num_experts_per_tok: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Expert weights (batched)
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_size, dim))  # gate_proj
        self.w2 = nn.Parameter(torch.randn(num_experts, intermediate_size, dim))  # down_proj (transposed)
        self.w3 = nn.Parameter(torch.randn(num_experts, intermediate_size, dim))  # up_proj

        # Shared expert (always active)
        self.shared_gate_proj = nn.Linear(dim, shared_intermediate_size, bias=False)
        self.shared_up_proj = nn.Linear(dim, shared_intermediate_size, bias=False)
        self.shared_down_proj = nn.Linear(shared_intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)  # (T, D)

        # Router scores
        scores = self.gate(x_flat)  # (T, E)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1
        )  # (T, K), (T, K)
        expert_weights = expert_weights.softmax(dim=-1)  # (T, K)

        # Compute routed expert outputs
        # For each token, gather the selected expert weights
        w1_sel = self.w1[expert_indices]  # (T, K, intermediate, dim)
        w3_sel = self.w3[expert_indices]  # (T, K, intermediate, dim)
        w2_sel = self.w2[expert_indices]  # (T, K, intermediate, dim)

        # SwiGLU: silu(x @ w1^T) * (x @ w3^T) @ w2
        x1 = F.silu(torch.einsum("td,tkid->tki", x_flat, w1_sel))  # (T, K, intermediate)
        x3 = torch.einsum("td,tkid->tki", x_flat, w3_sel)  # (T, K, intermediate)
        expert_out = torch.einsum("tki,tkid->tkd", x1 * x3, w2_sel)  # (T, K, dim)

        # Weighted sum of expert outputs
        routed_out = torch.einsum("tkd,tk->td", expert_out, expert_weights)  # (T, D)

        # Shared expert (always active)
        shared_out = self.shared_down_proj(
            F.silu(self.shared_gate_proj(x_flat)) * self.shared_up_proj(x_flat)
        )

        # Combine routed + shared
        output = routed_out + shared_out
        return output.view(orig_shape)
