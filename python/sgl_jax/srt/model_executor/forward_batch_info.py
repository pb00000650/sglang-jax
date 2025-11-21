"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on TPU.
  It will be transformed from CPU scheduler to TPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of TPU tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, Optional

import jax
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput


class MediaType(IntEnum):
    """Types of media in multi-modal inputs"""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self.is_extend()

    def is_extend(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.TARGET_VERIFY
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self):
        return self == ForwardMode.DRAFT_EXTEND

    def is_extend_or_draft_extend_or_mixed(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.MIXED
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE


@total_ordering
class CaptureHiddenMode(IntEnum):
    # Do not capture anything.
    NULL = 0
    # Capture a hidden state of the last token.
    LAST = 1
    # Capture hidden states of all tokens.
    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class MediaInfo:
    """Information about multi-modal media in the batch"""
    # Type of media (image, audio, etc.)
    media_type: MediaType
    # Indices of requests that contain this media
    req_indices: jax.Array
    # Shapes of media data for each request [num_media, ...]
    shapes: jax.Array
    # Positions where media is inserted in the input sequence [num_media]
    positions: jax.Array
    # Total number of media elements in the batch
    count: int


@register_pytree_node_class
@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass, including multi-modal data"""

    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids [total_tokens]
    input_ids: jax.Array
    # The indices of requests in the req_to_token_pool
    req_pool_indices: jax.Array
    # The sequence length for each request [batch_size]
    seq_lens: jax.Array
    # decode token position in kv cache
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None

    attn_backend: AttentionBackend = None

    cache_loc: jax.Array = None

    # For extend
    extend_prefix_lens: jax.Array | None = None
    extend_seq_lens: jax.Array | None = None

    trace_request_ids: list[str] | None = None
    trace_request_objects: list | None = None

    spec_info: EagleVerifyInput | EagleDraftInput | None = None
    spec_algorithm: SpeculativeAlgorithm = None
    capture_hidden_mode: CaptureHiddenMode = None

    # Multi-modal data
    # Image data [total_image_patches] or [num_images, channels, height, width]
    image_data: Optional[jax.Array] = None
    # Audio data [total_audio_frames] or [num_audios, channels, frames]
    audio_data: Optional[jax.Array] = None
    # Video data [total_video_frames * patches] or [num_videos, frames, channels, height, width]
    video_data: Optional[jax.Array] = None
    # Metadata about media in the batch
    media_info: Optional[MediaInfo] = None
    # Mapping from media positions to token positions [num_media]
    media_to_token_mapping: Optional[jax.Array] = None

    def tree_flatten(self):
        children = (
            self.input_ids,
            self.req_pool_indices,
            self.seq_lens,
            self.out_cache_loc,
            self.positions,
            self.extend_start_loc,
            self.attn_backend,
            self.cache_loc,
            self.extend_prefix_lens,
            self.extend_seq_lens,
            self.spec_info,
            self.image_data,
            self.audio_data,
            self.video_data,
            self.media_to_token_mapping,
            (self.media_info.req_indices, self.media_info.shapes, self.media_info.positions) 
            if self.media_info else None,
        )

        aux_data = {
            "forward_mode": self.forward_mode,
            "batch_size": self.batch_size,
            "spec_algorithm": self.spec_algorithm,
            "capture_hidden_mode": self.capture_hidden_mode,
            "trace_request_ids": self.trace_request_ids,
            "trace_request_objects": self.trace_request_objects,
            "media_type_count": self.media_info.media_type.value if self.media_info else None,
            "media_count": self.media_info.count if self.media_info else 0,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.forward_mode = aux_data["forward_mode"]
        obj.batch_size = aux_data["batch_size"]
        obj.spec_algorithm = aux_data["spec_algorithm"]
        obj.capture_hidden_mode = aux_data["capture_hidden_mode"]
        obj.trace_request_ids = aux_data["trace_request_ids"]
        obj.trace_request_objects = aux_data["trace_request_objects"]

        obj.input_ids = children[0]
        obj.req_pool_indices = children[1]
        obj.seq_lens = children[2]
        obj.out_cache_loc = children[3]
        obj.positions = children[4]
        obj.extend_start_loc = children[5]
        obj.attn_backend = children[6]
        obj.cache_loc = children[7]
        obj.extend_prefix_lens = children[8]
        obj.extend_seq_lens = children[9]
        obj.spec_info = children[10]
        obj.image_data = children[11]
        obj.audio_data = children[12]
        obj.video_data = children[13]
        obj.media_to_token_mapping = children[14]

        # Reconstruct media info if present
        media_data = children[15]
        if media_data is not None and aux_data["media_type_count"] is not None:
            obj.media_info = MediaInfo(
                media_type=MediaType(aux_data["media_type_count"]),
                req_indices=media_data[0],
                shapes=media_data[1],
                positions=media_data[2],
                count=aux_data["media_count"]
            )
        else:
            obj.media_info = None

        return obj

    def __repr__(self) -> str:
        jax_array_fields = []

        for field_name in [
            "input_ids",
            "req_pool_indices",
            "seq_lens",
            "out_cache_loc",
            "positions",
            "extend_start_loc",
            "cache_loc",
            "extend_prefix_lens",
            "extend_seq_lens",
            "image_data",
            "audio_data",
            "video_data",
            "media_to_token_mapping",
        ]:
            value = getattr(self, field_name, None)
            if value is not None and isinstance(value, jax.Array):
                jax_array_fields.append(f"{field_name}={value.shape}")

        # Add media info if present
        media_str = ""
        if self.media_info:
            media_str = f", media_type={self.media_info.media_type}, media_count={self.media_info.count}"

        jax_arrays_str = ", ".join(jax_array_fields)
        return f"ForwardBatch(forward_mode={self.forward_mode}, batch_size={self.batch_size}, {jax_arrays_str}{media_str})"

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        # Process text and common fields
        (
            input_ids,
            seq_lens,
            out_cache_loc,
            positions,
            extend_start_loc,
            req_pool_indices,
            cache_loc,
            extend_prefix_lens,
            extend_seq_lens,
            image_data,
            audio_data,
            video_data,
            media_to_token_mapping,
            media_req_indices,
            media_shapes,
            media_positions,
        ) = device_array(
            (
                batch.input_ids,
                batch.seq_lens,
                batch.out_cache_loc,
                batch.positions,
                batch.extend_start_loc,
                batch.req_pool_indices,
                batch.cache_loc,
                batch.extend_prefix_lens,
                batch.extend_seq_lens,
                batch.image_data if hasattr(batch, 'image_data') else None,
                batch.audio_data if hasattr(batch, 'audio_data') else None,
                batch.video_data if hasattr(batch, 'video_data') else None,
                batch.media_to_token_mapping if hasattr(batch, 'media_to_token_mapping') else None,
                batch.media_req_indices if hasattr(batch, 'media_req_indices') else None,
                batch.media_shapes if hasattr(batch, 'media_shapes') else None,
                batch.media_positions if hasattr(batch, 'media_positions') else None,
            ),
            sharding=(
                NamedSharding(model_runner.mesh, PartitionSpec())
                if jax.process_count() == 1
                else None
            ),
        )

        # Create media info if available
        media_info = None
        if hasattr(batch, 'media_type') and batch.media_type is not None:
            media_info = MediaInfo(
                media_type=batch.media_type,
                req_indices=media_req_indices,
                shapes=media_shapes,
                positions=media_positions,
                count=batch.media_count if hasattr(batch, 'media_count') else 0
            )

        obj = cls(
            bid=batch.bid,
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=input_ids,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            extend_start_loc=extend_start_loc,
            req_pool_indices=req_pool_indices,
            cache_loc=cache_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            attn_backend=model_runner.attn_backend,
            spec_info=batch.spec_info,
            spec_algorithm=batch.spec_algorithm,
            capture_hidden_mode=batch.capture_hidden_mode,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            media_info=media_info,
            media_to_token_mapping=media_to_token_mapping,
        )

        return obj