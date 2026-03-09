# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet linear attention block for Qwen3.5 models in ExecuTorch."""

from typing import Optional

import torch
import torch.nn.functional as F
from executorch.examples.models.llama.attention import ForwardOptions
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.norm import RMSNorm
from torch import nn


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention mechanism.

    Implements the gated delta rule for linear-complexity attention:
        S_t = g_t * S_{t-1} + k_t * [beta_t * (v_t - g_t * S_{t-1}^T @ k_t)]
        o_t = S_t^T @ q_t

    Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel = conv_kernel

        k_dim = n_heads * key_head_dim
        v_dim = n_heads * value_head_dim

        # Input projections
        self.q_proj = nn.Linear(dim, k_dim, bias=False)
        self.k_proj = nn.Linear(dim, k_dim, bias=False)
        self.v_proj = nn.Linear(dim, v_dim, bias=False)

        # Gate and beta projections
        # a_proj -> gating logits, b_proj -> beta logits
        self.a_proj = nn.Linear(dim, n_heads, bias=False)
        self.b_proj = nn.Linear(dim, n_heads, bias=False)

        # Output gate and projection
        self.g_proj = nn.Linear(dim, v_dim, bias=False)
        self.o_proj = nn.Linear(v_dim, dim, bias=False)

        # Causal conv1d for q, k, v (depthwise)
        self.conv_q = nn.Conv1d(k_dim, k_dim, kernel_size=conv_kernel, padding=0, groups=k_dim, bias=True)
        self.conv_k = nn.Conv1d(k_dim, k_dim, kernel_size=conv_kernel, padding=0, groups=k_dim, bias=True)
        self.conv_v = nn.Conv1d(v_dim, v_dim, kernel_size=conv_kernel, padding=0, groups=v_dim, bias=True)

        # Conv state buffers
        conv_state_q = torch.zeros(1, k_dim, conv_kernel - 1)
        conv_state_k = torch.zeros(1, k_dim, conv_kernel - 1)
        conv_state_v = torch.zeros(1, v_dim, conv_kernel - 1)
        self.register_buffer("conv_state_q", conv_state_q)
        self.register_buffer("conv_state_k", conv_state_k)
        self.register_buffer("conv_state_v", conv_state_v)

        # DeltaNet state: (batch, n_heads, key_head_dim, value_head_dim)
        deltanet_state = torch.zeros(1, n_heads, key_head_dim, value_head_dim)
        self.register_buffer("deltanet_state", deltanet_state)

        # A_log parameter for gating (learnable log decay)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

        # Layer norm for output
        self.norm = RMSNorm(v_dim, eps=norm_eps)

    def _causal_conv(self, x, conv, conv_state):
        """Apply causal conv1d with state management."""
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = torch.cat([conv_state, x], dim=-1)

        # Update state
        new_state = x[:, :, -(self.conv_kernel - 1):]
        with torch.no_grad():
            conv_state.copy_(new_state)

        # Conv and activation
        seq_len = x.size(-1) - (self.conv_kernel - 1)
        out = conv(x)[:, :, :seq_len]
        out = F.silu(out)
        return out.transpose(1, 2)  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project q, k, v
        q = self.q_proj(x)  # (B, L, k_dim)
        k = self.k_proj(x)  # (B, L, k_dim)
        v = self.v_proj(x)  # (B, L, v_dim)

        # Causal conv1d
        q = self._causal_conv(q, self.conv_q, self.conv_state_q)
        k = self._causal_conv(k, self.conv_k, self.conv_state_k)
        v = self._causal_conv(v, self.conv_v, self.conv_state_v)

        # Compute gate and beta from original input
        a = self.a_proj(x)  # (B, L, n_heads)
        b = self.b_proj(x)  # (B, L, n_heads)

        # Gating: g = exp(-A_log * softplus(a + dt_bias))
        g = torch.exp(-torch.exp(self.A_log).unsqueeze(0).unsqueeze(0) *
                       F.softplus(a + self.dt_bias.unsqueeze(0).unsqueeze(0)))  # (B, L, n_heads)

        # Beta: update rate
        beta = torch.sigmoid(b)  # (B, L, n_heads)

        # Reshape for head computation
        q = q.reshape(batch_size, seq_len, self.n_heads, self.key_head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.key_head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.value_head_dim)

        # L2 normalize q and k
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Sequential DeltaNet recurrence (no custom op needed)
        S = self.deltanet_state.clone()  # (B, n_heads, key_head_dim, value_head_dim)
        o_list = []

        for t in range(seq_len):
            q_t = q[:, t, :, :]  # (B, n_heads, key_head_dim)
            k_t = k[:, t, :, :]  # (B, n_heads, key_head_dim)
            v_t = v[:, t, :, :]  # (B, n_heads, value_head_dim)
            g_t = g[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B, n_heads, 1, 1)
            beta_t = beta[:, t, :].unsqueeze(-1)  # (B, n_heads, 1)

            # Retrieve: S_{t-1}^T @ k_t -> (B, n_heads, value_head_dim)
            retrieved = torch.einsum("bhkv,bhk->bhv", S, k_t)

            # Prediction error: v_t - g_t * retrieved
            # g_t is (B, n_heads, 1, 1), squeeze for broadcast
            error = v_t - g_t.squeeze(-1) * retrieved  # (B, n_heads, value_head_dim)

            # Update: S_t = g_t * S_{t-1} + k_t (x) [beta_t * error]
            # k_t: (B, n_heads, key_head_dim), beta_t * error: (B, n_heads, value_head_dim)
            update = k_t.unsqueeze(-1) * (beta_t * error).unsqueeze(-2)  # (B, n_heads, key_head_dim, value_head_dim)
            S = g_t * S + update

            # Output: o_t = S_t^T @ q_t
            o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)  # (B, n_heads, value_head_dim)
            o_list.append(o_t)

        # Update state
        with torch.no_grad():
            self.deltanet_state.copy_(S)

        # Stack and reshape
        o = torch.stack(o_list, dim=1)  # (B, L, n_heads, value_head_dim)
        o = o.reshape(batch_size, seq_len, self.n_heads * self.value_head_dim)

        # Norm
        o = self.norm(o)

        # Output gate
        o = o * F.silu(self.g_proj(x))

        # Output projection
        o = self.o_proj(o)

        return o

    def reset_cache(self):
        """Reset all state caches."""
        self.conv_state_q.zero_()
        self.conv_state_k.zero_()
        self.conv_state_v.zero_()
        self.deltanet_state.zero_()


class GatedDeltaNetBlock(nn.Module):
    """
    Full GatedDeltaNet block with FFN, following ExecuTorch's layer interface.
    Compatible with construct_transformer()'s layer interface:
        forward(x, freqs_cos, freqs_sin, attn_options) -> (output, None)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int = 16,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel: int = 4,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.deltanet = GatedDeltaNet(
            dim=dim,
            n_heads=n_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel=conv_kernel,
            norm_eps=norm_eps,
        )
        self.feed_forward = FeedForward(dim, hidden_dim)
        # Use attention_norm name to unify with TransformerBlock
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        _unused_attn_options: Optional[ForwardOptions] = None,
    ):
        # DeltaNet attention with residual
        h = self.deltanet(self.attention_norm(x))
        h = x + h
        # FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, None

    def reset_cache(self):
        self.deltanet.reset_cache()
