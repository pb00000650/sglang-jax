import math
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

# 全局缓存字典，用于存储已创建的旋转嵌入实例
_ROPE_DICT: Dict[Tuple, Any] = {}


def _rotate_neox(x: jnp.ndarray) -> jnp.ndarray:
    """Neox风格的旋转操作"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def _rotate_gptj(x: jnp.ndarray) -> jnp.ndarray:
    """GPT-J风格的旋转操作"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def _apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    is_neox_style: bool,
) -> jnp.ndarray:
    """
    应用旋转位置嵌入

    参数:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: 是否使用Neox风格的旋转位置嵌入
    """
    cos = jnp.expand_dims(cos, axis=-2)
    sin = jnp.expand_dims(sin, axis=-2)

    if is_neox_style:
        x1, x2 = jnp.split(x, 2, axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return jnp.concatenate((o1, o2), axis=-1)
    else:
        return jnp.stack((o1, o2), axis=-1).reshape(*x.shape[:-1], -1)


class RotaryEmbedding:
    """基础旋转位置嵌入类"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # 计算cos和sin缓存
        self.cos_sin_cache = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: Union[int, float]) -> jnp.ndarray:
        """计算逆频率"""
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, self.rotary_dim, 2, dtype=self.dtype) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """计算cos和sin缓存"""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        return jnp.concatenate((cos, sin), axis=-1)

    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """前向传播"""
        if offsets is not None:
            positions = positions + offsets

        positions = positions.flatten()
        num_tokens = positions.shape[0]

        # 获取对应的cos和sin值
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        # 处理query
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        # 处理key
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """支持线性缩放的旋转位置嵌入"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

        # 计算缩放因子到偏移量的映射
        self._scaling_factor_to_offset: Dict[float, int] = {}
        offsets: List[int] = []
        for i, scaling_factor in enumerate(self.scaling_factors):
            max_len = self.max_position_embeddings * scaling_factor
            if not offsets:
                offset = 0
            else:
                last_offset = offsets[-1]
                prev_max_len = self.max_position_embeddings * self.scaling_factors[i - 1]
                offset = last_offset + int(prev_max_len)
            offsets.append(offset)
            self._scaling_factor_to_offset[scaling_factor] = offset

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """计算cos和sin缓存，支持多个缩放因子"""
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[jnp.ndarray] = []

        for scaling_factor in self.scaling_factors:
            max_len = int(self.max_position_embeddings * scaling_factor)
            t = jnp.arange(max_len, dtype=self.dtype) / scaling_factor

            freqs = jnp.einsum("i,j -> ij", t, inv_freq)
            cos = jnp.cos(freqs)
            sin = jnp.sin(freqs)
            cache = jnp.concatenate((cos, sin), axis=-1)
            cache_list.append(cache)

        return jnp.concatenate(cache_list, axis=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """支持动态NTK缩放的旋转位置嵌入"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """计算应用动态NTK缩放后的cos和sin缓存"""
        max_len = int(self.max_position_embeddings * self.scaling_factor)
        # 调整base以实现动态缩放
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings)
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))

        inv_freq = self._compute_inv_freq(base)
        t = jnp.arange(max_len, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        return jnp.concatenate((cos, sin), axis=-1)


# YaRN相关辅助函数
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: jnp.dtype) -> jnp.ndarray:
    if low == high:
        high += 0.001  # 防止除以零

    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    return jnp.clip(linear_func, 0, 1)


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """支持YaRN缩放方法的旋转位置嵌入"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype = jnp.float32,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # 获取n-d magnitude缩放
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> jnp.ndarray:
        pos_freqs = self.base ** (
            jnp.arange(0, self.rotary_dim, 2, dtype=self.dtype) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )

        # 获取n-d旋转缩放（针对外推进行校正）
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, self.dtype)
        ) * self.extrapolation_factor

        return inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        max_len = int(self.max_position_embeddings * self.scaling_factor)
        t = jnp.arange(max_len, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs) * self.mscale
        sin = jnp.sin(freqs) * self.mscale
        return jnp.concatenate((cos, sin), axis=-1)


class MRotaryEmbedding(RotaryEmbedding):
    """支持多模态部分的旋转嵌入"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype = jnp.float32,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        # 验证并调整mrope_section
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE部分和不匹配: 预期 {expected_sum}, 得到 {actual_sum}."
                    f" 调整mrope_section以匹配rotary_dim // 2 = {expected_sum}"
                )
                # 按比例调整mrope_section
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor)) for section in self.mrope_section
                    ]
                    # 调整最后一个元素以确保总和完全匹配
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum

        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """前向传播"""
        assert positions.ndim == 1 or positions.ndim == 2

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        # 处理多模态位置
        if positions.ndim == 2 and self.mrope_section:
            if self.mrope_interleaved:
                cos = self._apply_interleaved_rope(cos)
                sin = self._apply_interleaved_rope(sin)
            else:
                # 按部分拼接
                cos_sections = jnp.split(cos, self.mrope_section, axis=-1)
                sin_sections = jnp.split(sin, self.mrope_section, axis=-1)
                cos = jnp.concatenate([cos_sections[i] for i in range(len(cos_sections))], axis=-1)
                sin = jnp.concatenate([sin_sections[i] for i in range(len(sin_sections))], axis=-1)

        # 处理query
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        # 处理key
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key

    def _apply_interleaved_rope(self, x: jnp.ndarray) -> jnp.ndarray:
        """应用交错的MRoPE"""
        x_t = x[0].copy()
        x_t = x_t.at[..., 1 : self.mrope_section[1] * 3 : 3].set(
            x[1, ..., 1 : self.mrope_section[1] * 3 : 3]
        )
        x_t = x_t.at[..., 2 : self.mrope_section[2] * 3 : 3].set(
            x[2, ..., 2 : self.mrope_section[2] * 3 : 3]
        )
        return x_t

    # 补全get_rope_index静态方法（适配JAX）
    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: Optional[int] = None,
        input_ids: Optional[jnp.ndarray] = None,
        image_grid_thw: Optional[jnp.ndarray] = None,
        video_grid_thw: Optional[jnp.ndarray] = None,
        second_per_grid_ts: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if model_type == "qwen3_omni_moe":
            # 适配qwen3-omni模型
            return MRotaryEmbedding.get_rope_index_qwen3_omni(
                spatial_merge_size,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                tokens_per_second,
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                **kwargs,
            )

        # 处理视频grid_thw重复
        if (
            model_type.startswith("qwen3_vl") or model_type.startswith("qwen3_vl_moe")
        ) and video_grid_thw is not None:
            # JAX版本的repeat_interleave
            video_grid_thw = jnp.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
            video_grid_thw = video_grid_thw.at[:, 0].set(1)

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            batch_size, seq_len = total_input_ids.shape

            # 初始化position_ids（JAX版本）
            position_ids = jnp.ones(
                (3, batch_size, seq_len), dtype=total_input_ids.dtype, device=total_input_ids.device
            )

            image_index, video_index = 0, 0

            for i in range(batch_size):
                input_ids_i = total_input_ids[i]
                # 查找vision_start_token的位置（JAX版本）
                vision_start_indices = jnp.where(input_ids_i == vision_start_token_id)[0]

                # 修复：多变量分别赋值，避免解包错误
                image_nums = 0
                video_nums = 0

                if vision_start_indices.size > 0:
                    vision_tokens = input_ids_i[vision_start_indices + 1]
                    # 新增：判断 vision_tokens 是否为空（避免索引越界）
                    if vision_tokens.size > 0:
                        image_nums = jnp.sum(vision_tokens == image_token_id)
                        video_nums = jnp.sum(vision_tokens == video_token_id)
                    else:
                        image_nums = 0
                        video_nums = 0

                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums.item(), video_nums.item()

                for _ in range(remain_images + remain_videos):
                    # 查找下一个图像或视频token的位置
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if (remain_images > 0 and image_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if (remain_videos > 0 and video_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )

                    if ed_image < ed_video:
                        # 处理图像
                        t, h, w = (
                            image_grid_thw[image_index][0].item(),
                            image_grid_thw[image_index][1].item(),
                            image_grid_thw[image_index][2].item(),
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        # 处理视频
                        t, h, w = (
                            video_grid_thw[video_index][0].item(),
                            video_grid_thw[video_index][1].item(),
                            video_grid_thw[video_index][2].item(),
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index].item()
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    # 计算网格大小
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    text_len = ed - st

                    # 添加文本位置ID
                    if text_len > 0:
                        st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                        text_pos = jnp.arange(text_len, dtype=position_ids.dtype) + st_idx
                        llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))  # (3, text_len)

                    # 添加视觉位置ID
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                    if model_type == "qwen2_5_vl":
                        # 时间维度计算
                        range_tensor = jnp.arange(llm_grid_t)[:, None]
                        expanded_range = jnp.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        t_index = time_tensor.astype(position_ids.dtype).flatten()
                    elif model_type in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe"):
                        t_index = jnp.repeat(
                            jnp.arange(llm_grid_t, dtype=position_ids.dtype),
                            llm_grid_h * llm_grid_w,
                        )
                    else:
                        raise RuntimeError(f"未实现的模型类型: {model_type}")

                    # 高度和宽度维度
                    h_index = jnp.tile(
                        jnp.repeat(jnp.arange(llm_grid_h, dtype=position_ids.dtype), llm_grid_w),
                        llm_grid_t,
                    )
                    w_index = jnp.tile(
                        jnp.arange(llm_grid_w, dtype=position_ids.dtype), llm_grid_t * llm_grid_h
                    )

                    # 堆叠T/H/W位置ID
                    vision_pos = jnp.stack([t_index, h_index, w_index]) + st_idx
                    llm_pos_ids_list.append(vision_pos)

                    # 更新起始位置
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # 处理剩余文本
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    text_pos = jnp.arange(text_len, dtype=position_ids.dtype) + st_idx
                    llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))

                # 拼接位置ID
                llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)  # (3, seq_len)
                position_ids = position_ids.at[:, i, :].set(llm_positions)

                # 计算delta
                mrope_position_deltas.append(llm_positions.max().item() + 1 - len(input_tokens))

            # 转换delta为JAX数组
            mrope_position_deltas = jnp.array(
                mrope_position_deltas, dtype=position_ids.dtype, device=position_ids.device
            ).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # 纯文本情况
            s = input_ids.shape[1]
            position_ids = jnp.tile(
                jnp.arange(s, dtype=input_ids.dtype)[None, None, :], (3, input_ids.shape[0], 1)
            ).astype(input_ids.dtype)

            # 计算delta
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True).squeeze(2)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    # 补全qwen3_omni专用的get_rope_index（适配JAX）
    @staticmethod
    def get_rope_index_qwen3_omni(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        tokens_per_second: Optional[int] = None,
        input_ids: Optional[jnp.ndarray] = None,
        image_grid_thw: Optional[jnp.ndarray] = None,
        video_grid_thw: Optional[jnp.ndarray] = None,
        second_per_grid_ts: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        audio_token_id = kwargs["audio_token_id"]
        audio_start_token_id = kwargs["audio_start_token_id"]
        position_id_per_seconds = kwargs["position_id_per_seconds"]
        use_audio_in_video = kwargs.get("use_audio_in_video", False)
        audio_seqlens = kwargs.get("audio_seqlens", None)
        second_per_grids = second_per_grid_ts

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            batch_size, seq_len = total_input_ids.shape

            # 初始化position_ids（JAX版本，使用float类型）
            position_ids = jnp.zeros(
                (3, batch_size, seq_len), dtype=jnp.float32, device=total_input_ids.device
            )

            image_idx, video_idx, audio_idx = 0, 0, 0

            for i in range(batch_size):
                current_input_ids = total_input_ids[i]
                # 计算多模态token数量
                vision_start_indices = jnp.where(current_input_ids == vision_start_token_id)[0]
                image_nums, video_nums, audio_nums = 0, 0, 0

                if vision_start_indices.size > 0:
                    vision_tokens = current_input_ids[vision_start_indices + 1]
                    image_nums = jnp.sum(vision_tokens == image_token_id)
                    video_nums = (
                        jnp.sum(vision_tokens == audio_start_token_id)
                        if use_audio_in_video
                        else jnp.sum(vision_tokens == video_token_id)
                    )

                audio_nums = jnp.sum(current_input_ids == audio_start_token_id)
                input_tokens = current_input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = (
                    image_nums.item(),
                    video_nums.item(),
                    audio_nums.item(),
                )

                multimodal_nums = (
                    image_nums + audio_nums
                    if use_audio_in_video
                    else image_nums + video_nums + audio_nums
                )

                for _ in range(multimodal_nums.item()):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0.0

                    # 查找下一个模态起始位置
                    ed_vision_start = (
                        input_tokens.index(vision_start_token_id, st)
                        if (
                            (remain_videos > 0 or remain_images > 0)
                            and vision_start_token_id in input_tokens[st:]
                        )
                        else len(input_tokens) + 1
                    )

                    ed_audio_start = (
                        input_tokens.index(audio_start_token_id, st)
                        if (remain_audios > 0 and audio_start_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )

                    min_ed = min(ed_vision_start, ed_audio_start)
                    text_len = min_ed - st

                    # 添加文本位置
                    if text_len != 0:
                        text_pos = jnp.arange(text_len, dtype=jnp.float32) + st_idx
                        llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))
                        st_idx = llm_pos_ids_list[-1].max().item() + 1

                    # 处理BOS/EOS长度
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1

                    # 添加BOS位置
                    bos_pos = jnp.arange(bos_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(bos_pos, (3, 1)))
                    st_idx = llm_pos_ids_list[-1].max().item() + 1

                    # 处理音频单独输入
                    if min_ed == ed_audio_start:
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx].item()
                        )
                        audio_pos = jnp.arange(audio_len, dtype=jnp.float32) + st_idx
                        llm_pos_ids_list.append(jnp.tile(audio_pos, (3, 1)))

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    # 处理图像输入
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == image_token_id
                    ):
                        grid_t = image_grid_thw[image_idx][0].item()
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32) * 1 * position_id_per_seconds
                        )

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            image_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        image_len = (
                            image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    # 处理视频输入
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == video_token_id
                    ):
                        grid_t = video_grid_thw[video_idx][0].item()
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        )

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        video_len = (
                            video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    # 处理视频中的音频
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        # 音频部分
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx].item()
                        )
                        audio_pos = jnp.arange(audio_len, dtype=jnp.float32) + st_idx
                        audio_llm_pos_ids = jnp.tile(audio_pos, (3, 1))

                        # 视频部分
                        grid_t = video_grid_thw[video_idx][0].item()
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        )

                        video_llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        # 交错合并音频和视频位置
                        merged_pos = []
                        audio_idx_ptr, video_idx_ptr = 0, 0
                        while (
                            audio_idx_ptr < audio_llm_pos_ids.shape[1]
                            and video_idx_ptr < video_llm_pos_ids.shape[1]
                        ):
                            if (
                                audio_llm_pos_ids[0, audio_idx_ptr]
                                <= video_llm_pos_ids[0, video_idx_ptr]
                            ):
                                merged_pos.append(
                                    audio_llm_pos_ids[:, audio_idx_ptr : audio_idx_ptr + 1]
                                )
                                audio_idx_ptr += 1
                            else:
                                merged_pos.append(
                                    video_llm_pos_ids[:, video_idx_ptr : video_idx_ptr + 1]
                                )
                                video_idx_ptr += 1

                        # 添加剩余部分
                        if audio_idx_ptr < audio_llm_pos_ids.shape[1]:
                            merged_pos.append(audio_llm_pos_ids[:, audio_idx_ptr:])
                        if video_idx_ptr < video_llm_pos_ids.shape[1]:
                            merged_pos.append(video_llm_pos_ids[:, video_idx_ptr:])

                        if merged_pos:
                            llm_pos_ids_list.append(jnp.concatenate(merged_pos, axis=1))

                        video_len = (
                            video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        st += text_len + bos_len + audio_len + video_len + eos_len

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                    # 添加EOS位置
                    eos_pos = jnp.arange(eos_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(eos_pos, (3, 1)))

                # 处理剩余文本
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0.0
                    text_len = len(input_tokens) - st
                    text_pos = jnp.arange(text_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))

                # 拼接位置ID
                llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)
                position_ids = position_ids.at[:, i, :].set(llm_positions)

                # 计算delta
                mrope_position_deltas.append(llm_positions.max().item() + 1 - len(input_tokens))

            # 转换delta为JAX数组
            mrope_position_deltas = jnp.array(
                mrope_position_deltas, dtype=jnp.float32, device=position_ids.device
            ).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # 纯文本情况
            s = input_ids.shape[1]
            position_ids = jnp.tile(
                jnp.arange(s, dtype=jnp.float32)[None, None, :], (3, input_ids.shape[0], 1)
            )

            # 计算delta
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True).squeeze(2)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    # 辅助方法：计算特征提取输出长度
    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths: int) -> int:
        """计算卷积层的输出长度和音频编码器的输出长度"""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    # 辅助方法：获取视觉模态的位置ID
    @staticmethod
    def _get_llm_pos_ids_for_vision(
        st_idx: float,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: jnp.ndarray,
        grid_hs: jnp.ndarray,
        grid_ws: jnp.ndarray,
        device: Any,
    ) -> jnp.ndarray:
        grid_h = (grid_hs[vision_idx] // spatial_merge_size).item()
        grid_w = (grid_ws[vision_idx] // spatial_merge_size).item()

        # 生成H和W维度的位置ID
        h_index = jnp.tile(
            jnp.repeat(jnp.arange(grid_h, dtype=jnp.float32, device=device), grid_w), len(t_index)
        )

        w_index = jnp.tile(
            jnp.arange(grid_w, dtype=jnp.float32, device=device), len(t_index) * grid_h
        )

        # 生成T维度的位置ID
        t_index = jnp.repeat(t_index, grid_h * grid_w)

        # 堆叠并添加起始偏移
        llm_pos_ids = jnp.stack([t_index, h_index, w_index], axis=0) + st_idx
        return llm_pos_ids


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[jnp.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    """获取旋转嵌入实例（带缓存）"""
    if dtype is None:
        dtype = jnp.float32

    # 处理rope_scaling参数用于缓存键
    if rope_scaling is not None:
        rope_scaling_tuple = tuple(sorted(rope_scaling.items()))
    else:
        rope_scaling_tuple = None

    # 处理部分旋转因子
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    # 生成缓存键
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_tuple,
        dtype,
    )

    # 检查缓存
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    # 创建新的旋转嵌入实例
    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "dynamic":
            if "alpha" in rope_scaling:
                raise NotImplementedError("DynamicNTKAlphaRotaryEmbedding尚未实现")
            else:
                scaling_factor = rope_scaling["factor"]
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "default" and "mrope_section" in rope_scaling:
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_scaling["mrope_section"],
                mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
            )
        else:
            raise ValueError(f"未知的RoPE缩放类型: {scaling_type}")

    # 存入缓存
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


# 旋转辅助函数
def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """旋转输入的一半隐藏维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    unsqueeze_dim: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """应用旋转位置嵌入"""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
