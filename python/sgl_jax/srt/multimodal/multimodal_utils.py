# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PreTrainedTokenizer

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.image_processor import JAXImageProcessor

logger = logging.getLogger(__name__)


def process_multimodal_inputs(
    inputs: Union[str, List[Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
    image_processor: JAXImageProcessor,
    model_config: ModelConfig,
) -> Dict[str, jnp.ndarray]:
    """
    Process multimodal inputs (text + images) into model-compatible format
    
    Args:
        inputs: Inputs in format:
            - List of dicts with "role" and "content" (OpenAI format)
            - Each content can be text or list of {"type": "text"/"image", "text"/"image": ...}
        tokenizer: Text tokenizer
        image_processor: Image processor
        model_config: Model configuration
        
    Returns:
        Dictionary with processed inputs:
            - input_ids: Tokenized text
            - image_embeddings: Processed images
            - image_token_positions: Positions of image tokens in input_ids
    """
    if isinstance(inputs, str):
        return _process_text_only(inputs, tokenizer)

    # Process chat format inputs
    text_segments = []
    images = []
    image_token_positions = []
    image_token = model_config.image_token or "<image>"

    for message in inputs:
        content = message.get("content", "")
        if isinstance(content, str):
            text_segments.append(content)
            continue

        # Handle multimodal content
        for item in content:
            if item.get("type") == "text":
                text_segments.append(item.get("text", ""))
            elif item.get("type") == "image":
                images.append(item.get("image"))
                text_segments.append(image_token)
                # Track position (will be adjusted after tokenization)
                image_token_positions.append(len(text_segments) - 1)

    # Combine text segments and tokenize
    full_text = " ".join(text_segments)
    tokenized = tokenizer(
        full_text,
        return_tensors="jax",
        padding="longest",
        truncation=True,
        max_length=model_config.max_seq_len,
    )
    input_ids = tokenized["input_ids"]

    # Process images
    image_embeddings = None
    if images:
        image_embeddings = image_processor.preprocess(images)

    # Adjust image token positions to actual token indices
    adjusted_positions = []
    if image_token_positions and image_token in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        for pos in image_token_positions:
            # Find the actual position of image token in input_ids
            token_pos = jnp.where(input_ids == image_token_id)[1]
            if len(token_pos) > pos:
                adjusted_positions.append(token_pos[pos].item())

    return {
        "input_ids": input_ids,
        "image_embeddings": image_embeddings,
        "image_token_positions": jnp.array(adjusted_positions, dtype=jnp.int32)
        if adjusted_positions
        else None,
    }


def _process_text_only(text: str, tokenizer: PreTrainedTokenizer) -> Dict[str, jnp.ndarray]:
    """Helper for text-only inputs"""
    tokenized = tokenizer(
        text,
        return_tensors="jax",
        padding="longest",
        truncation=True,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "image_embeddings": None,
        "image_token_positions": None,
    }


def embed_images(
    images: jnp.ndarray,
    image_encoder: nnx.Module,
    mesh: jax.sharding.Mesh,
) -> jnp.ndarray:
    """
    Embed images using a pre-trained image encoder
    
    Args:
        images: Preprocessed image tensors (batch x channels x height x width)
        image_encoder: Image encoder model
        mesh: JAX mesh for parallel processing
        
    Returns:
        Image embeddings (batch x embedding_dim)
    """
    with jax.sharding.use_mesh(mesh):
        # Shard images across data parallel dimension
        images = jax.sharding.reshard(images, jax.sharding.PartitionSpec("data", None, None, None))
        return image_encoder(images)