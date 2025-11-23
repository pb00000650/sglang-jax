import logging
from typing import Any, Optional, Tuple, List, Dict, Union

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.logits_processor import LogitsProcessor, LogitsMetadata
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.srt.configs.model_config import ModelConfig

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Qwen2_5VLMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: Optional[nnx.Rngs] = None,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Optional[jax.sharding.Mesh] = None,
    ) -> None:
        self.layer_id = layer_id
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        intermediate = up * self.act_fn(gate)
        output, _ = self.down_proj(intermediate)
        return output


class Qwen2_5VLAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 1000000,
        rope_scaling: Optional[dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0, "Num heads must be divisible by num kv heads"
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=True,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            positions: 位置编码张量，形状为 [total_tokens]
            hidden_states: 输入隐藏状态，形状为 [total_tokens, hidden_size]
            forward_batch: 包含批量处理信息的ForwardBatch对象
            token_to_kv_pool: KV缓存池对象，用于存储和更新键值缓存

        Returns:
            注意力输出张量和融合的KV缓存张量
        """
        # 投影得到Q、K、V
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # 重塑为多头格式并指定分片策略
        from jax.sharding import PartitionSpec as P

        q = q.reshape(
            (-1, self.q_head_num, self.head_dim),
            out_sharding=P(None, "tensor", None),  # 保持在注意力头维度的分片
        )
        k = k.reshape((-1, self.kv_head_num, self.head_dim), out_sharding=P(None, "tensor", None))
        v = v.reshape((-1, self.kv_head_num, self.head_dim), out_sharding=P(None, "tensor", None))

        # 应用旋转位置编码（适配视觉-语言模型的位置处理）
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        # 调用注意力计算核心逻辑
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        # 投影回隐藏状态维度
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Qwen2_5VLDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        # 修复注意力头参数获取逻辑
        num_attention_heads = getattr(
            config, "num_attention_heads", getattr(config, "num_heads", 0)
        )
        num_key_value_heads = getattr(
            config, "num_key_value_heads", getattr(config, "num_kv_heads", num_attention_heads)
        )

        self.self_attn = Qwen2_5VLAttention(
            hidden_size=config.hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.mlp = Qwen2_5VLMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, List[Any]]:
        layer_callback_flag: List[Any] = []

        # 初始输入维度检查与修复
        if hidden_states.ndim != 3:
            # logger.warning(
            #     f"hidden_states 期望3D，实际{hidden_states.ndim}D，形状: {hidden_states.shape}"
            # )
            # 尝试从批次信息推断维度
            batch_size = forward_batch.batch_size
            if hidden_states.ndim == 2:
                seq_len = hidden_states.shape[0] // batch_size
                hidden_size = hidden_states.shape[1]
                hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)
                #logger.info(f"修复 hidden_states 为3D: {hidden_states.shape}")
            else:
                raise ValueError(f"无法修复 hidden_states 维度: {hidden_states.shape}")

        # 处理残差连接和输入层归一化
        if residual is None:
            # 初始化残差时确保3D
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 检查残差维度并修复
            if residual.ndim != 3:
                #logger.warning(f"residual 期望3D，实际{residual.ndim}D，形状: {residual.shape}")
                batch_size = forward_batch.batch_size
                if residual.ndim == 2:
                    seq_len = residual.shape[0] // batch_size
                    hidden_size = residual.shape[1]
                    residual = residual.reshape(batch_size, seq_len, hidden_size)
                    #logger.info(f"修复 residual 为3D: {residual.shape}")
                else:
                    raise ValueError(f"无法修复 residual 维度: {residual.shape}")

            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # 注意力计算
        attn_output, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # 验证并修复注意力输出维度
        if attn_output.ndim != 3:
            # logger.warning(
            #     f"attn_output 期望3D，实际{attn_output.ndim}D，形状: {attn_output.shape}"
            # )

            # 从残差和批次信息推断正确维度
            batch_size = forward_batch.batch_size
            if residual.ndim == 3:
                target_shape = residual.shape
            else:
                seq_len = attn_output.shape[0] // batch_size
                hidden_size = attn_output.shape[-1]
                target_shape = (batch_size, seq_len, hidden_size)

            # 尝试修复
            if attn_output.ndim == 2:
                try:
                    attn_output = attn_output.reshape(target_shape)
                    #logger.info(f"修复 attn_output 为3D: {attn_output.shape}")
                except ValueError as e:
                    logger.error(f"修复失败: {e}")
                    raise
            else:
                raise ValueError(f"无法修复注意力输出维度: {attn_output.shape}")

        hidden_states = attn_output

        # 残差连接和后注意力层归一化
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP计算
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, kv_fused, layer_callback_flag


class Qwen2_5VLModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        # 视觉相关组件初始化
        self.has_vision = hasattr(config, "vision_config")
        if self.has_vision:
            vision_config = config.vision_config
            # 为视觉配置设置默认值
            vision_config.num_channels = getattr(vision_config, "num_channels", 3)
            vision_config.rms_norm_eps = getattr(
                vision_config, "rms_norm_eps", getattr(config, "rms_norm_eps", 1e-6)
            )
            vision_config.hidden_size = getattr(vision_config, "hidden_size", config.hidden_size)
            vision_config.intermediate_size = getattr(
                vision_config, "intermediate_size", config.intermediate_size
            )
            vision_config.num_attention_heads = getattr(
                vision_config, "num_attention_heads", config.num_attention_heads
            )
            vision_config.num_key_value_heads = getattr(
                vision_config, "num_key_value_heads", config.num_key_value_heads
            )
            vision_config.max_position_embeddings = getattr(
                vision_config, "max_position_embeddings", 1024
            )

            # 计算视觉嵌入的输入维度（基于patch size和通道数）
            patch_size = getattr(vision_config, "patch_size", 14)
            self.vision_embed_tokens = Embed(
                num_embeddings=patch_size**2 * vision_config.num_channels,
                features=vision_config.hidden_size,
                rngs=rngs,
                dtype=dtype,
                kernel_axes=("tensor", None),
                param_dtype=dtype,
                mesh=mesh,
            )

            # 视觉位置嵌入
            num_patches = getattr(vision_config, "num_patches", 14**2)  # 默认14x14 patches
            self.vision_pos_embed = nnx.Param(
                jnp.zeros((1, num_patches + 1, vision_config.hidden_size), dtype=dtype),
                name="vision_pos_embed",
            )
            self.vision_norm = RMSNorm(
                vision_config.hidden_size,
                epsilon=vision_config.rms_norm_eps,
                param_dtype=dtype,
                rngs=rngs,
            )
            self.vision_proj = LinearBase(
                input_size=vision_config.hidden_size,
                output_size=config.hidden_size,
                kernel_axes=(None, "tensor"),
                use_bias=False,
                params_dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            )
            # 视觉解码器层
            self.vision_layers = nnx.data(
                [
                    Qwen2_5VLDecoderLayer(
                        config=vision_config,
                        layer_id=i,
                        dtype=dtype,
                        rngs=rngs,
                        mesh=mesh,
                    )
                    for i in range(getattr(vision_config, "depth", 0))
                ]
            )

        # 文本嵌入层
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        # 解码器层
        self.layers = nnx.data(
            [
                Qwen2_5VLDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # 最终归一化层
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> Tuple[jax.Array, List[jax.Array], List[Any]]:
        residual: Optional[jax.Array] = None
        # 文本嵌入
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        # 处理视觉输入
        if (
            self.has_vision
            and hasattr(forward_batch, "vision_inputs")
            and forward_batch.vision_inputs is not None
        ):
            vision_inputs = forward_batch.vision_inputs
            # 视觉输入处理（确保输入维度正确）
            if vision_inputs.ndim != 3:
                raise ValueError(
                    f"视觉输入期望3D，实际{vision_inputs.ndim}D: {vision_inputs.shape}"
                )

            vision_emb = self.vision_embed_tokens(vision_inputs)
            # 添加位置嵌入（确保广播维度正确）
            if vision_emb.shape[1] != self.vision_pos_embed.shape[1]:
                logger.warning(
                    f"视觉嵌入与位置嵌入长度不匹配: {vision_emb.shape[1]} vs {self.vision_pos_embed.shape[1]}"
                )

            vision_emb = vision_emb + self.vision_pos_embed
            # 视觉层处理
            vision_residual = None
            for vision_layer in self.vision_layers:
                vision_emb, vision_residual, _, _ = vision_layer(
                    positions=getattr(
                        forward_batch, "vision_positions", jnp.arange(vision_emb.shape[1])
                    ),
                    hidden_states=vision_emb,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    residual=vision_residual,
                )
            # 视觉归一化和投影
            vision_emb = self.vision_norm(vision_emb)
            vision_emb, _ = self.vision_proj(vision_emb)

            # 将视觉嵌入与文本嵌入拼接（确保批次维度匹配）
            if hidden_states.shape[0] != vision_emb.shape[0]:
                raise ValueError(
                    f"文本与视觉批次不匹配: {hidden_states.shape[0]} vs {vision_emb.shape[0]}"
                )
            hidden_states = jnp.concatenate([hidden_states, vision_emb], axis=1)

        layers_kv_fused: List[jax.Array] = []
        layers_callback_flag: List[Any] = []

        # 逐层处理
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        # 最终处理
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: Optional[nnx.Rngs] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.model = Qwen2_5VLModel(config, dtype=dtype, rngs=rngs, mesh=mesh)
        # 词嵌入共享配置
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
                rngs=rngs,
            )
        else:
            self.lm_head = None  # 共享时使用嵌入层权重

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)
        self.config = config
        self.dtype = dtype
        self.mesh = mesh
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
            # Text embedding
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            # Final norm
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        # Vision mappings if available
        if self.model.has_vision:
            # 修正视觉嵌入层权重路径映射
            mappings.update(
                {
                    # Vision embedding (修正权重路径)
                    "model.vision_model.embed_tokens.weight": WeightMapping(
                        target_path="model.vision_embed_tokens.embedding",
                        sharding=("tensor", None),
                        transpose=False,
                    ),
                    # Vision position embedding
                    "model.vision_model.pos_embed": WeightMapping(
                        target_path="model.vision_pos_embed",
                        sharding=(None, None),
                        transpose=False,
                    ),
                    # Vision norm (修正权重路径)
                    "model.vision_model.norm.weight": WeightMapping(
                        target_path="model.vision_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    # Vision projection
                    "model.vision_proj.weight": WeightMapping(
                        target_path="model.vision_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                }
            )

        # LM head mapping if not tying word embeddings
        if not getattr(self.config, "tie_word_embeddings", False) and self.lm_head is not None:
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # Text layers mappings
        num_text_layers = self.config.num_hidden_layers
        for layer_idx in range(num_text_layers):
            text_layer_mappings = self._create_text_layer_mappings(layer_idx)
            mappings.update(text_layer_mappings)

        # Vision layers mappings
        if self.model.has_vision and hasattr(self.config, "vision_config"):
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
        target_prefix = f"model.layers.{layer_idx}"

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

    @nnx.jit
    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[jax.Array, List[jax.Array], List[Any]]:
        # 自动设备放置
        if self.mesh is not None:
            with self.mesh:
                hidden_states, layers_kv_fused, layers_callback_flag = self.model(
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                )
        else:
            hidden_states, layers_kv_fused, layers_callback_flag = self.model(
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        # 计算logits
        if not self.tie_word_embeddings and self.lm_head is not None:
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, layers_callback_flag


# 注册模型入口类
EntryClass = Qwen2_5_VLForConditionalGeneration
