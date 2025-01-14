# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import torch
import transformers


class LongRoPEScaledRotaryEmbedding(torch.nn.Module):
    """
    LongRoPE Scaled Rotary Positional Encoding class for Llama-like model.

    Args:
        dim (int): Head dimension.
        rescale_factors (list): List of rescale factors for each dimension.
        scale (float, optional): Length scale for code compatibility.
        max_position_embeddings (int, optional): Maximum number of position embeddings (after scaled).
        original_max_position_embeddings (int, optional): Original maximum number of position embeddings (before scaled).
        base (int, optional): Base value for the positional encoding. Defaults to 10000.
        magnitude_scaling_policy (str, optional): Attention temperature scaling function. Defaults to "su".
        device (torch.device, optional): Device on which to create the embedding. Defaults to None.
    """

    def __init__(
        self,
        dim, 
        rescale_factors,
        layer_idx_list,
        scale=1.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        model_type="llama",
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base

        if magnitude_scaling_policy == "su":
            calc_mscale = self._calc_mscale_su
        elif magnitude_scaling_policy == "yarn":
            calc_mscale = self._calc_mscale_yarn
        else:
            calc_mscale = lambda scale: float(magnitude_scaling_policy)
        self.mscale = calc_mscale(self.max_position_embeddings / self.original_max_position_embeddings)

        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32, device=device)
        assert rescale_factors.shape[-1] == self.dim // 2, \
            f"misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}"
        self.rescale_factors = {}
        for i in range(len(layer_idx_list)):
            self.rescale_factors[layer_idx_list[i]] = rescale_factors[i]

        if model_type == "llama":
            self.forward = self._forward_llama
        elif model_type == "mistral":
            self.forward = self._forward_mistral
            self.register_buffer("inv_freq", self._calc_inv_freq(max_position_embeddings, device))
        else:
            raise ValueError(f"Unsupported model type for LongRoPE: {model_type}")

    def _calc_mscale_su(self, scale):
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def _calc_mscale_yarn(self, scale):
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _calc_inv_freq(self, seq_len, device, layer_idx):
        rescale_factors = self.rescale_factors[layer_idx].to(device)
        exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        return 1.0 / (rescale_factors * (self.base ** exponent))

    @torch.no_grad()
    def _forward_mistral(self, x, seq_len=None):
        seq_len = x.shape[-2] if seq_len is None else seq_len
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(x.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.mscale).to(x.dtype), (emb.sin() * self.mscale).to(x.dtype)

    @torch.no_grad()
    def _forward_llama(self, x, position_ids, layer_idx, seq_len=None):
        if layer_idx not in self.rescale_factors.keys():
            # Placeholder for non-attention layers
            return None, None
        seq_len = x.shape[-2] if seq_len is None else seq_len
        inv_freq = self._calc_inv_freq(seq_len, x.device, layer_idx)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.mscale
            sin = emb.sin() * self.mscale
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DynamicLongRoPEScaledRotaryEmbedding(LongRoPEScaledRotaryEmbedding):

    def _calc_inv_freq(self, seq_len, device, layer_idx):
        rescale_factors = self.rescale_factors[layer_idx].to(device)
        current_scale = seq_len / self.original_max_position_embeddings
        original_scale = self.max_position_embeddings / self.original_max_position_embeddings
        dynamic_scale = (current_scale - 1.0) / (original_scale - 1.0)
        rescale_factors = 1.0 + (self.rescale_factors[layer_idx] - 1.0) * dynamic_scale
        exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        return 1.0 / (rescale_factors * (self.base ** exponent))

class MixedLongRoPEScaledRotaryEmbedding(LongRoPEScaledRotaryEmbedding):

    def __init__(
        self,
        dim, 
        rescale_factors,
        layer_idx_list,
        start_token_idx,
        original_rope,
        scale=1.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        model_type="llama",
        device=None,
    ):
        super().__init__(
            dim, rescale_factors, layer_idx_list, scale, max_position_embeddings, 
            original_max_position_embeddings, base, magnitude_scaling_policy, model_type, device,
        )
        assert model_type == "llama", "Only llama support added for Mixed LongRoPE"
        self.start_token_idx = start_token_idx
        self.original_rope = original_rope
        self._longrope_forward = super().forward

    def forward(self, x, position_ids, layer_idx, seq_len=None):
        emb_cos_origin, emb_sin_origin = self.original_rope(x, position_ids)
        emb_cos, emb_sin = self._longrope_forward(x, position_ids, layer_idx, seq_len)
        assert emb_cos_origin.shape == emb_cos.shape and emb_sin_origin.shape == emb_cos.shape, \
                        'original embeddings shape should be the same with current embeddings'
        assert self.start_token_idx <= emb_cos.shape[-2], "start_token_idx exceeds sequence length"

        emb_cos[..., :self.start_token_idx, :] = emb_cos_origin[..., :self.start_token_idx, :]
        emb_sin[..., :self.start_token_idx, :] = emb_sin_origin[..., :self.start_token_idx, :]
        return emb_cos, emb_sin
