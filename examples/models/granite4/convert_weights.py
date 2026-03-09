# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight conversion from HuggingFace Granite 4.0 Hybrid format to Meta/ExecuTorch format."""

import re
from typing import Dict


# Mapping from Meta format keys to HuggingFace format keys
# {meta_key_pattern: hf_key_pattern}
_GRANITE4_FROM_META = {
    # Embeddings
    "tok_embeddings.weight": "model.embed_tokens.weight",
    # Final norm
    "norm.weight": "model.norm.weight",
    # Output (tied with embeddings in Granite 4.0)
    "output.weight": "model.embed_tokens.weight",
    # Attention layers
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    # FFN / MoE layers (for attention blocks)
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.block_sparse_moe.experts.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.block_sparse_moe.experts.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.block_sparse_moe.experts.up_proj.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    # MoE router
    "layers.{}.block_sparse_moe.gate.weight": "model.layers.{}.block_sparse_moe.gate.weight",
    # Shared expert
    "layers.{}.block_sparse_moe.shared_gate_proj.weight": "model.layers.{}.block_sparse_moe.shared_expert.gate_proj.weight",
    "layers.{}.block_sparse_moe.shared_up_proj.weight": "model.layers.{}.block_sparse_moe.shared_expert.up_proj.weight",
    "layers.{}.block_sparse_moe.shared_down_proj.weight": "model.layers.{}.block_sparse_moe.shared_expert.down_proj.weight",
    # Mamba2 layers
    "layers.{}.mamba2.in_proj.weight": "model.layers.{}.mamba.in_proj.weight",
    "layers.{}.mamba2.conv1d.weight": "model.layers.{}.mamba.conv1d.weight",
    "layers.{}.mamba2.conv1d.bias": "model.layers.{}.mamba.conv1d.bias",
    "layers.{}.mamba2.dt_bias": "model.layers.{}.mamba.dt_bias",
    "layers.{}.mamba2.A_log": "model.layers.{}.mamba.A_log",
    "layers.{}.mamba2.D": "model.layers.{}.mamba.D",
    "layers.{}.mamba2.out_proj.weight": "model.layers.{}.mamba.out_proj.weight",
    "layers.{}.mamba2.norm.weight": "model.layers.{}.mamba.norm.weight",
    "layers.{}.mamba2.inner_norm.weight": "model.layers.{}.mamba.inner_layernorm.weight",
    # Mamba layer norms
    "layers.{}.norm.weight": "model.layers.{}.input_layernorm.weight",
}


def _build_key_map(num_layers: int) -> Dict[str, str]:
    """Build a complete key mapping for the given number of layers."""
    key_map = {}
    for meta_pattern, hf_pattern in _GRANITE4_FROM_META.items():
        if "{}" in meta_pattern:
            for i in range(num_layers):
                meta_key = meta_pattern.format(i)
                hf_key = hf_pattern.format(i)
                key_map[hf_key] = meta_key
        else:
            key_map[hf_pattern] = meta_pattern
    return key_map


def convert_weights(
    state_dict: Dict[str, "torch.Tensor"],
    num_layers: int = 40,
    **kwargs,
) -> Dict[str, "torch.Tensor"]:
    """
    Convert HuggingFace Granite 4.0 state dict to Meta/ExecuTorch format.

    Args:
        state_dict: HuggingFace format state dict
        num_layers: Number of layers in the model

    Returns:
        Converted state dict in Meta format
    """
    key_map = _build_key_map(num_layers)
    new_state_dict = {}

    for hf_key, tensor in state_dict.items():
        if hf_key in key_map:
            meta_key = key_map[hf_key]
            new_state_dict[meta_key] = tensor
        else:
            # Keep unmapped keys as-is (will be filtered later)
            new_state_dict[hf_key] = tensor

    return new_state_dict
