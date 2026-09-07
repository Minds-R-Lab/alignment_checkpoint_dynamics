"""Uniform access to the per-unit factorization of transformer blocks.

Every supported architecture exposes, per layer:
  read  : (m, d) matrix whose ROWS are the read vectors a_k of the m hidden units
          (the linear input projection; for gated MLPs this is `up_proj`)
  gate  : (m, d) or None (gated MLPs only)
  write : (d, m) matrix whose COLUMNS are the write vectors d_k
  v     : (n_heads*dh, d) value projection, rows = per-channel read vectors of attention
  o     : (d, n_heads*dh) output projection, columns = per-channel write vectors

plus the module handles needed for hooks. Conventions match the sandbox analysis
(`overnight_results.md`): cos_k = cos(read[k], write[:, k]).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LayerHandles:
    idx: int
    block: nn.Module            # the whole transformer block
    mlp: nn.Module              # MLP submodule (output hook target: dL/d(mlp_out))
    read_mod: nn.Linear         # up / dense_h_to_4h  (pre-activation hook target)
    gate_mod: nn.Module | None  # gate_proj or None
    write_mod: nn.Linear        # down / dense_4h_to_h
    mlp_norm: nn.Module         # the norm feeding the MLP (its INPUT is the residual h the MLP writes to)
    v_mod: nn.Module | None
    o_mod: nn.Module | None
    packed_qkv: bool            # gpt_neox packs [heads, 3, dh]
    n_heads: int
    head_dim: int
    parallel_residual: bool

    # --- weight accessors (always return float32 tensors with the conventions above) ---
    def read(self) -> torch.Tensor:
        return self.read_mod.weight.detach().float()                      # (m, d)

    def gate(self) -> torch.Tensor | None:
        return None if self.gate_mod is None else self.gate_mod.weight.detach().float()

    def write(self) -> torch.Tensor:
        return self.write_mod.weight.detach().float()                     # (d, m)

    def v(self) -> torch.Tensor | None:
        if self.v_mod is None:
            return None
        W = self.v_mod.weight.detach().float()
        if self.packed_qkv:   # (n_heads*3*dh, d) laid out as [head][q|k|v][dh]
            W = W.view(self.n_heads, 3, self.head_dim, -1)[:, 2].reshape(self.n_heads * self.head_dim, -1)
        return W                                                          # (n_heads*dh, d)

    def o(self) -> torch.Tensor | None:
        return None if self.o_mod is None else self.o_mod.weight.detach().float()   # (d, n_heads*dh)


def _first(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def get_layers(model) -> list[LayerHandles]:
    """Detect the architecture and return LayerHandles for every block."""
    cfg = model.config
    mt = cfg.model_type
    out = []
    if mt == "gpt_neox":
        layers = model.gpt_neox.layers
        for i, blk in enumerate(layers):
            out.append(LayerHandles(
                idx=i, block=blk, mlp=blk.mlp, read_mod=blk.mlp.dense_h_to_4h, gate_mod=None,
                write_mod=blk.mlp.dense_4h_to_h, mlp_norm=blk.post_attention_layernorm,
                v_mod=blk.attention.query_key_value, o_mod=blk.attention.dense, packed_qkv=True,
                n_heads=cfg.num_attention_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
                parallel_residual=bool(getattr(cfg, "use_parallel_residual", True))))
        return out
    if mt in {"llama", "mistral", "qwen2", "qwen3", "olmo", "olmo2", "gemma", "gemma2", "phi3", "smollm3"}:
        layers = model.model.layers
        for i, blk in enumerate(layers):
            mlp = blk.mlp
            out.append(LayerHandles(
                idx=i, block=blk, mlp=mlp, read_mod=mlp.up_proj, gate_mod=_first(mlp, "gate_proj"),
                write_mod=mlp.down_proj,
                mlp_norm=_first(blk, "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"),
                v_mod=blk.self_attn.v_proj, o_mod=blk.self_attn.o_proj, packed_qkv=False,
                n_heads=cfg.num_attention_heads, head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                parallel_residual=False))
        return out
    if mt == "gpt2":
        # Conv1D stores weight as (in, out): transpose on access via wrapper below
        raise NotImplementedError("gpt2 uses Conv1D (transposed weights); add a wrapper if needed")
    raise NotImplementedError(f"model_type {mt} not supported; add it to rwloop/adapters.py")


def firing_module(h: LayerHandles) -> nn.Module:
    """Module whose OUTPUT defines 'firing': pre-activation of the nonlinearity.
    GeLU MLPs: read_mod output. Gated MLPs: the gate pre-activation (the up branch is linear)."""
    return h.gate_mod if h.gate_mod is not None else h.read_mod
