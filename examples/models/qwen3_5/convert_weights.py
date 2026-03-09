# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight conversion from HuggingFace Qwen3.5 format to Meta/ExecuTorch format."""

from typing import Dict


# Mapping from Meta format keys to HuggingFace format keys
_QWEN3_5_FROM_META = {
    # Embeddings
    "tok_embeddings.weight": "model.embed_tokens.weight",
    # Final norm
    "norm.weight": "model.norm.weight",
    # Output (tied with embeddings)
    "output.weight": "model.embed_tokens.weight",
    # Full attention layers (every 4th layer)
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.q_norm.weight": "model.layers.{}.self_attn.q_norm.weight",
    "layers.{}.attention.k_norm.weight": "model.layers.{}.self_attn.k_norm.weight",
    # Norms
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    # FFN
    "layers.{}.feed_forward.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "model.layers.{}.mlp.up_proj.weight",
    # DeltaNet layers (linear_attention layers)
    "layers.{}.deltanet.q_proj.weight": "model.layers.{}.deltanet.q_proj.weight",
    "layers.{}.deltanet.k_proj.weight": "model.layers.{}.deltanet.k_proj.weight",
    "layers.{}.deltanet.v_proj.weight": "model.layers.{}.deltanet.v_proj.weight",
    "layers.{}.deltanet.a_proj.weight": "model.layers.{}.deltanet.a_proj.weight",
    "layers.{}.deltanet.b_proj.weight": "model.layers.{}.deltanet.b_proj.weight",
    "layers.{}.deltanet.g_proj.weight": "model.layers.{}.deltanet.g_proj.weight",
    "layers.{}.deltanet.o_proj.weight": "model.layers.{}.deltanet.o_proj.weight",
    "layers.{}.deltanet.A_log": "model.layers.{}.deltanet.a_log",
    "layers.{}.deltanet.dt_bias": "model.layers.{}.deltanet.dt_bias",
    # DeltaNet conv weights
    "layers.{}.deltanet.conv_q.weight": "model.layers.{}.deltanet.q_conv1d.weight",
    "layers.{}.deltanet.conv_q.bias": "model.layers.{}.deltanet.q_conv1d.bias",
    "layers.{}.deltanet.conv_k.weight": "model.layers.{}.deltanet.k_conv1d.weight",
    "layers.{}.deltanet.conv_k.bias": "model.layers.{}.deltanet.k_conv1d.bias",
    "layers.{}.deltanet.conv_v.weight": "model.layers.{}.deltanet.v_conv1d.weight",
    "layers.{}.deltanet.conv_v.bias": "model.layers.{}.deltanet.v_conv1d.bias",
    # DeltaNet norms
    "layers.{}.deltanet.norm.weight": "model.layers.{}.deltanet.o_norm.weight",
}


def _build_key_map(num_layers: int) -> Dict[str, str]:
    """Build a complete key mapping for the given number of layers."""
    key_map = {}
    for meta_pattern, hf_pattern in _QWEN3_5_FROM_META.items():
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
    num_layers: int = 24,
    **kwargs,
) -> Dict[str, "torch.Tensor"]:
    """
    Convert HuggingFace Qwen3.5 state dict to Meta/ExecuTorch format.

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
            new_state_dict[hf_key] = tensor

    return new_state_dict
