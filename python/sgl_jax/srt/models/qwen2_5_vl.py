import math
import logging
from functools import partial
from typing import Callable, List, Literal, NamedTuple, Optional, TypedDict, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import modeling_flax_utils
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.logits_processor import LogitsProcessor, LogitsMetadata
from sgl_jax.srt.models.qwen2 import (
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2DecoderLayer,
    Qwen2Attention,
    Qwen2MLP,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

init_fn = nnx.initializers.uniform()
DEFAULT_BLOCK_K_MAJOR = 128
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SegmentIds(NamedTuple):
    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: tuple[tuple[int, int, int], ...]


class Qwen2_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: jax.Array
    image_grid_thw: jax.Array


Qwen2_5_VLImageInputs = Union[Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs]


class Qwen2_5_VisionMLP(nnx.Module):
    def __init__(
        self, config: Qwen2_5_VLVisionConfig, dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh
    ):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        self.gate_proj = LinearBase(
            input_size=in_features,
            output_size=hidden_features,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=in_features,
            output_size=hidden_features,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=hidden_features,
            output_size=in_features,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.act_fn = act_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x)[0])
        up = self.up_proj(x)[0]
        fuse = gate * up
        result = self.down_proj(fuse)[0]
        return result


def apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
    _, _, _, H = x.shape
    half_dim = H // 2

    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    return jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)


def generate_window_segment_ids(
    cu_seqlens: jax.Array, seq_len: int, padded_seq_len: int
) -> SegmentIds:
    indices = jnp.arange(seq_len, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side="right") + 1
    padding_segment_ids = jnp.zeros(padded_seq_len - seq_len, dtype=jnp.int32)
    segment_ids = jnp.concatenate([segment_ids, padding_segment_ids])
    segment_ids = segment_ids.reshape(1, -1)

    return SegmentIds(q=segment_ids, kv=segment_ids)


class Qwen2_5_VisionAttention(nnx.Module):
    def __init__(self, config: Qwen2_5_VLConfig, dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_kv_heads = self.num_heads
        self.head_dim_original = self.hidden_size // self.num_heads
        self.head_dim = self.head_dim_original
        self.mesh = mesh

        self.qkv_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=3 * self.hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_kv_heads,
            layer_id=0,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: Optional[jax.Array] = None,
        use_fullattn: bool = True,
    ) -> jax.Array:
        T, B, D = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        block_k_major = DEFAULT_BLOCK_K_MAJOR
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major - 1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, "constant")
        k = jnp.pad(k, pad_width, "constant")
        v = jnp.pad(v, pad_width, "constant")

        segment_ids = generate_window_segment_ids(cu_window_seqlens, T_attn, padded_T)

        forward_batch = ForwardBatch(
            input_ids=jnp.zeros((1, padded_T), dtype=jnp.int32),
            positions=jnp.arange(padded_T),
            cu_seq_lens=jnp.array([0, padded_T], dtype=jnp.int32),
            slot_mapping=jnp.arange(padded_T),
        )
        kv_cache = KVCache()

        attn_output, _ = self.attn(q, k, v, forward_batch, kv_cache)
        attn_output = attn_output[:, :, :T_attn, :]
        attn_output = jnp.transpose(attn_output, (2, 0, 1, 3))
        attn_output = attn_output.reshape(T, B, D)

        output, _ = self.proj(attn_output)
        return output


class Qwen2_5_VisionBlock(nnx.Module):
    def __init__(self, config: Qwen2_5_VLConfig, dtype: jnp.dtype, rngs: nnx.Rngs, mesh: Mesh):
        vision_config = config.vision_config
        dim = vision_config.hidden_size
        norm_layer = partial(RMSNorm, epsilon=config.rms_norm_eps)

        self.norm1 = norm_layer(dim, param_dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, param_dtype=dtype, rngs=rngs)
        self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(config=vision_config, dtype=dtype, rngs=rngs, mesh=mesh)

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: Optional[jax.Array] = None,
        use_fullattn: bool = True,
    ) -> jax.Array:

        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens, use_fullattn)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh = None,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        self.proj = LinearBase(
            input_size=in_channels * temporal_patch_size * patch_size * patch_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        output, _ = self.proj(x)
        return output


class Qwen2_5_VisionPatchMerger(nnx.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim, param_dtype=dtype, rngs=rngs)
        self.mlp_fc1 = LinearBase(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = LinearBase(
            input_size=self.hidden_size,
            output_size=d_model,
            use_bias=True,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x, _ = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x, _ = self.mlp_fc2(x)
        return x


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2_5_VisionTransformer(nnx.Module):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        norm_eps: float = 1e-6,
    ):
        vision_config = config.vision_config
        self.config = vision_config
        self.dtype = dtype

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=vision_config.hidden_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.data(
            [
                Qwen2_5_VisionBlock(
                    config=config,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for _ in range(vision_config.depth)
            ]
        )

        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            norm_layer=partial(RMSNorm, epsilon=norm_eps),
            spatial_merge_size=vision_config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = (
            hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size

        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
        )
        index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        num_valid_indices = grid_t * llm_grid_h * llm_grid_w
        valid_indices = jnp.nonzero(index_padded != -100, size=num_valid_indices)[0]
        index_new = index_padded[valid_indices]
        cu_seqlens_tmp = jnp.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.astype(jnp.int32)

        return index_new, cu_seqlens_tmp

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(t, h, w)

        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)

        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(-1, rotary_pos_emb_thw.shape[-1])
        cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)

        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw, cu_seqlens_thw)

    def compute_aux_arrays(self, grid_thw: tuple[tuple[int, int, int]]):
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [jnp.array([0], dtype=jnp.int32)]
        cu_seqlens: list = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for i in range(num_grids):
            t, h, w = grid_thw[i]

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
        window_index = jnp.concatenate(window_index, axis=0)
        cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0),), mode="constant", constant_values=0)
        return window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens

    def compute_hidden_states(
        self,
        x: jax.Array,
        window_index: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_seqlens: jax.Array,
        cu_window_seqlens: jax.Array,
    ) -> jax.Array:
        hidden_states = self.patch_embed(x)

        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    cu_window_seqlens=cu_seqlens,
                    use_fullattn=True,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    cu_window_seqlens=cu_window_seqlens,
                    use_fullattn=False,
                )

        hidden_states = self.merger(hidden_states)
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int, int]]) -> jax.Array:
        window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = self.compute_aux_arrays(
            grid_thw
        )
        return self.compute_hidden_states(
            x, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens
        )


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ) -> None:
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
        self.config = config
        self.dtype = dtype
        self.mesh = mesh

        self.visual = Qwen2_5_VisionTransformer(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.language_model = Qwen2ForCausalLM(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        logger.info(f"Qwen2.5VLForCausalLM initialized with dtype {dtype}")

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array) -> None:
        """加载模型权重，支持JAX分布式加载"""
        self.rng = nnx.Rngs(rng_key)

        # 根据mesh配置决定加载策略
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen2_5_vl_weight_mappings()

        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(weight_mappings)
        else:
            loader.load_weights_from_safetensors(weight_mappings)

        logger.info("Qwen2.5 VL weights loaded successfully!")

    def _create_qwen2_5_vl_weight_mappings(self) -> dict:
        mappings = {
            # Text embedding - 修正路径，添加language_model前缀
            "model.embed_tokens.weight": WeightMapping(
                target_path="language_model.model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            # Final norm - 修正路径，添加language_model前缀
            "model.norm.weight": WeightMapping(
                target_path="language_model.model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        # Vision mappings (检查视觉模型是否存在)
        if hasattr(self, "visual"):
            mappings.update(
                {
                    # Vision embedding - 修正视觉模型路径
                    "vision_model.embed_tokens.weight": WeightMapping(
                        target_path="visual.patch_embed.proj.weight",
                        sharding=("tensor", None),
                        transpose=False,
                    ),
                    # Vision norm - 修正路径
                    "vision_model.norm.weight": WeightMapping(
                        target_path="visual.merger.ln_q.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    # Vision projection
                    "vision_proj.weight": WeightMapping(
                        target_path="visual.merger.mlp_fc2.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                }
            )

        # LM head mapping if not tying word embeddings
        if not getattr(self.config, "tie_word_embeddings", False) and hasattr(self.language_model, "lm_head"):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="language_model.lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # Text layers mappings - 注意添加language_model前缀
        num_text_layers = self.config.num_hidden_layers
        for layer_idx in range(num_text_layers):
            text_layer_mappings = self._create_text_layer_mappings(layer_idx)
            mappings.update(text_layer_mappings)

        # Vision layers mappings
        if hasattr(self.config, "vision_config"):
            num_vision_layers = getattr(self.config.vision_config, "depth", 0)
            for layer_idx in range(num_vision_layers):
                vision_layer_mappings = self._create_vision_layer_mappings(layer_idx)
                mappings.update(vision_layer_mappings)

        # 添加MOE层映射（如果模型包含MOE结构）
        if hasattr(self.config, "num_experts") and self.config.num_experts > 0:
            num_layers = self.config.num_hidden_layers
            for layer_idx in range(num_layers):
                moe_mappings = self._create_moe_layer_mappings(layer_idx)
                mappings.update(moe_mappings)

        return mappings

    def _create_text_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"language_model.model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        # Attention bias mappings if enabled
        if getattr(self.config, "attention_bias", True):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
            }
            mappings.update(bias_mappings)

        return mappings

    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        # 修正视觉层前缀路径，与实际权重文件保持一致
        prefix = f"model.vision_model.layers.{layer_idx}"
        target_prefix = f"model.vision_layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        # Vision attention bias mappings if enabled
        vision_config = getattr(self.config, "vision_config", None)
        if vision_config and getattr(vision_config, "attention_bias", True):
            vision_bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
            }
            mappings.update(vision_bias_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int) -> dict:
        """添加MOE层权重映射（如果模型包含混合专家结构）"""
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        num_experts = getattr(self.config, "num_experts", 8)

        mappings = {
            f"{prefix}.mlp.gate.weight": WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
        }

        # 专家层权重映射
        for expert_type in ["gate_proj", "up_proj", "down_proj"]:
            target_name = {
                "gate_proj": "wi_0",
                "up_proj": "wi_1",
                "down_proj": "wo",
            }[expert_type]
            expert_keys = [
                f"{prefix}.mlp.experts.{i}.{expert_type}.weight" for i in range(num_experts)
            ]

            if expert_type == "down_proj":
                sharding = ("expert", "tensor", None)
            else:
                sharding = ("expert", None, "tensor")

            mappings[f"__MOE_EXPERTS__{prefix}.mlp.{target_name}"] = WeightMapping(
                target_path=[f"{target_prefix}.mlp.{target_name}"] + expert_keys,
                sharding=sharding,
                transpose=True,
            )

        return mappings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: int | None = None,
    ) -> tuple[jax.Array, int]:
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 1.0)

        input_tokens_tensor = np.array(input_tokens)
        vision_start_indices = np.argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = np.sum(vision_tokens == image_token_id)
        video_nums = np.sum(vision_tokens == video_token_id)
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1), (3, text_len)
                )
                + st_idx
            )

            t_index = (
                (
                    jnp.broadcast_to(
                        jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
                        (llm_grid_t, llm_grid_h * llm_grid_w),
                    )
                    * video_second_per_grid_t
                    * tokens_per_second
                )
                .astype(jnp.int32)
                .flatten()
            )

            h_index = jnp.broadcast_to(
                jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).flatten()
            w_index = jnp.broadcast_to(
                jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).flatten()

            llm_pos_ids_list.append(jnp.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st

            llm_pos_ids_list.append(
                jnp.broadcast_to(
                    jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1), (3, text_len)
                )
                + st_idx
            )

        llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> jax.Array:
        if isinstance(mm_input, list):
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        if hasattr(mm_input, "ndim"):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. " f"Got type: {type(mm_input)}")

    def _parse_and_validate_image_input(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values", pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )

        return None

    def _parse_and_validate_multimodal_inputs(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> dict:
        mm_input_by_modality = {}

        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    image_grid_thw, **kwargs
                )
        return mm_input_by_modality

    def get_single_image_embedding(self, image_pixel_values, image_grid_thw):
        return self.visual(image_pixel_values, (image_grid_thw,))

    def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[jax.Array, ...]:

        grid_thw = image_input["image_grid_thw"]

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = []
            current_idx = 0
            for image_thw in grid_thw:
                t, h, w = image_thw
                image_size = t * h * w
                end_idx = current_idx + image_size
                image_pixel_values = pixel_values[current_idx:end_idx, :]
                image_embeds.append(self.get_single_image_embedding(image_pixel_values, image_thw))
                current_idx = end_idx
            image_embeds = jnp.concatenate(image_embeds, axis=0)

        merge_size = self.visual.config.spatial_merge_size
        sizes = np.prod(np.array(grid_thw, dtype=np.int64), axis=-1) // merge_size // merge_size

        if sizes.size == 0:
            return ()
        if sizes.size == 1:
            return (image_embeds,)

        split_indices = np.cumsum(sizes)[:-1]
        return tuple(jnp.split(image_embeds, split_indices))

    def get_multimodal_embeddings(
        self, image_grid_thw: tuple[tuple[int, int, int], ...], **kwargs: object
    ) -> List[jax.Array]:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[jax.Array, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings

        return list(multimodal_embeddings)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
        input_embeds: Optional[List[jax.Array]] = None,
    ):
        hidden_states = self.language_model.model.embed_tokens(forward_batch.input_ids)

        if False: # input_embeds:
            # 新增：校验类型，确保是可迭代的嵌入数据
            if not isinstance(input_embeds, (list, np.ndarray, jnp.ndarray)):
                raise TypeError(f"Expected list/array for multimodal_embeddings, got {type(input_embeds)}")

            input_ids = forward_batch.input_ids
            image_token_id = self.config.image_token_id
            video_token_id = self.config.video_token_id

            img_mask = (input_ids == image_token_id) | (input_ids == video_token_id)

            # 处理输入维度（2D为批量，1D为单条）
            if input_ids.ndim == 2:
                max_seq_len = input_ids.shape[1]
                dim_index = 1
            else:
                max_seq_len = input_ids.size
                dim_index = 0

            # 获取图像/视频令牌的位置，用-1填充无效位置
            img_indices = jnp.where(img_mask, size=max_seq_len, fill_value=-1)[dim_index]

            # 生成有效位置索引（固定长度，-1表示无效）
            all_valid_mask = img_indices != -1
            valid_positions = jnp.where(all_valid_mask, jnp.arange(max_seq_len), -1)

            # 限制有效索引数量不超过嵌入数量（避免索引越界）
            num_embeddings = len(input_embeds)
            valid_indices = valid_positions[:num_embeddings]

            # 向量化更新隐藏状态（替代循环，更符合JAX特性）
            # 1. 过滤无效索引（-1）
            valid_mask = valid_indices != -1
            valid_idx = valid_indices[valid_mask]
            valid_embeds = jnp.array(input_embeds)[valid_mask]

            # 2. 批量更新隐藏状态
            if valid_idx.size > 0:
                hidden_states = hidden_states.at[:, valid_idx, :].set(valid_embeds)

        residual = None
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.language_model.model.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.language_model.model.norm(hidden_states)

        logits = self.logits_processor(hidden_states, self.language_model.model.embed_tokens, logits_metadata)

        return logits, layers_kv_fused, layers_callback_flag


# 注册模型入口类
EntryClass = Qwen2_5_VLForConditionalGeneration