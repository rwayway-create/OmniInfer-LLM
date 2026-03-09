# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba-2 SSM block for Granite 4.0 hybrid models in ExecuTorch."""

from typing import Optional

import torch
import torch.nn.functional as F
from executorch.examples.models.llama.attention import ForwardOptions
from executorch.examples.models.llama.norm import RMSNorm
from torch import nn


class Mamba2Block(nn.Module):
    """
    Mamba-2 Selective State Space Model block.

    Implements the Mamba-2 SSM layer compatible with ExecuTorch's export pipeline.
    Decode mode (seq_len=1) uses single-step recurrence with standard ops.
    Prefill mode (seq_len>1) uses sequential scan (no custom ops needed).

    Based on: "Transformers are SSMs: Generalized Models and Efficient Algorithms
    Through Structured State Space Duality" (Dao & Gu, 2024)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 48,
        n_groups: int = 1,
        d_head: int = 64,
        chunk_size: int = 256,
        conv_bias: bool = True,
        proj_bias: bool = False,
        norm_eps: float = 1e-5,
        residual_multiplier: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.chunk_size = chunk_size
        self.residual_multiplier = residual_multiplier

        # Input projection: x -> (z, x_proj, B, C, dt)
        # z: gate, x_proj: input to conv, B: SSM input matrix, C: SSM output matrix, dt: time step
        d_in_proj = 2 * self.d_inner + 2 * n_groups * d_state + n_heads
        self.in_proj = nn.Linear(dim, d_in_proj, bias=proj_bias)

        # Depthwise causal convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=0,  # manual padding with state
            groups=self.d_inner,
            bias=conv_bias,
        )

        # Conv state buffer for causal convolution cache
        conv_state = torch.zeros(1, self.d_inner, d_conv - 1)
        self.register_buffer("conv_state", conv_state)

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.D = nn.Parameter(torch.zeros(n_heads))

        # SSM state buffer: (batch, n_heads, d_state, d_head)
        ssm_state = torch.zeros(1, n_heads, d_state, d_head)
        self.register_buffer("ssm_state", ssm_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=proj_bias)

        # Norms
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.inner_norm = RMSNorm(self.d_inner, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        _unused_attn_options: Optional[ForwardOptions] = None,
    ):
        """
        Forward pass compatible with ExecuTorch's layer interface.
        Returns (output, None) to match TransformerBlock signature.
        """
        residual = x
        x = self.norm(x)

        batch_size, seq_len, _ = x.shape

        # Input projection
        proj = self.in_proj(x)  # (B, L, d_in_proj)

        # Split projections
        z, x_conv, B, C, dt = proj.split(
            [self.d_inner, self.d_inner, self.n_groups * self.d_state,
             self.n_groups * self.d_state, self.n_heads],
            dim=-1,
        )

        # Causal Conv1d with state
        x_conv = x_conv.transpose(1, 2)  # (B, d_inner, L)
        x_conv = torch.cat([self.conv_state, x_conv], dim=-1)  # (B, d_inner, L + d_conv - 1)

        # Update conv state
        new_conv_state = x_conv[:, :, -(self.d_conv - 1):]
        with torch.no_grad():
            self.conv_state.copy_(new_conv_state)

        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # (B, d_inner, L)
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)

        # Reshape for SSM computation
        # x_conv: (B, L, n_heads, d_head)
        x_ssm = x_conv.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        # B, C: (B, L, n_groups, d_state) -> expand to (B, L, n_heads, d_state)
        B = B.reshape(batch_size, seq_len, self.n_groups, self.d_state)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.d_state)

        # Expand groups to heads
        heads_per_group = self.n_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=2)  # (B, L, n_heads, d_state)
        C = C.repeat_interleave(heads_per_group, dim=2)  # (B, L, n_heads, d_state)

        # Compute dt (time step)
        dt = dt + self.dt_bias  # (B, L, n_heads)
        dt = F.softplus(dt)  # (B, L, n_heads)

        # Compute A (decay)
        A = -torch.exp(self.A_log)  # (n_heads,)

        # Sequential SSM scan (works for both prefill and decode)
        # For ExecuTorch export: unrolled sequential scan using standard ops
        y_list = []
        h = self.ssm_state.clone()  # (B, n_heads, d_state, d_head)

        for t in range(seq_len):
            # Extract current step
            x_t = x_ssm[:, t, :, :]  # (B, n_heads, d_head)
            B_t = B[:, t, :, :]  # (B, n_heads, d_state)
            C_t = C[:, t, :, :]  # (B, n_heads, d_state)
            dt_t = dt[:, t, :]  # (B, n_heads)

            # Discrete A: exp(A * dt)
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, n_heads)
            dA = dA.unsqueeze(-1).unsqueeze(-1)  # (B, n_heads, 1, 1)

            # dB * x: outer product of B_t and x_t
            dBx = B_t.unsqueeze(-1) * x_t.unsqueeze(-2)  # (B, n_heads, d_state, d_head)

            # State update: h = A_disc * h + B_disc * x
            h = dA * h + dBx  # (B, n_heads, d_state, d_head)

            # Output: y = C^T @ h + D * x
            y_t = torch.einsum("bhsd,bhs->bhd", h, C_t)  # (B, n_heads, d_head)

            # Add D skip connection
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_t  # (B, n_heads, d_head)

            y_list.append(y_t)

        # Update SSM state
        with torch.no_grad():
            self.ssm_state.copy_(h)

        # Stack outputs
        y = torch.stack(y_list, dim=1)  # (B, L, n_heads, d_head)
        y = y.reshape(batch_size, seq_len, self.d_inner)  # (B, L, d_inner)

        # Apply inner norm
        y = self.inner_norm(y)

        # Gate
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Residual connection with multiplier
        output = residual + output * self.residual_multiplier

        return output, None

    def reset_cache(self):
        """Reset conv and SSM state caches."""
        self.conv_state.zero_()
        self.ssm_state.zero_()
