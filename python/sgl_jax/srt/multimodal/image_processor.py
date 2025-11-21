# SPDX-License-Identifier: Apache-2.0
import io
import logging
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx
from PIL import Image
from transformers import AutoImageProcessor, PreTrainedImageProcessor

logger = logging.getLogger(__name__)


class JAXImageProcessor(nnx.Module):
    """JAX-compatible image processor for multimodal models"""
    
    def __init__(
        self,
        image_processor: PreTrainedImageProcessor,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.image_processor = image_processor
        self.dtype = dtype
        self.normalize = image_processor.do_normalize
        self.mean = jnp.array(image_processor.image_mean, dtype=dtype)
        self.std = jnp.array(image_processor.image_std, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ) -> "JAXImageProcessor":
        """Load image processor from Hugging Face Hub"""
        image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,** kwargs
        )
        return cls(image_processor, dtype=dtype)

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        return_tensors: str = "jax",
    ) -> jnp.ndarray:
        """
        Preprocess images for multimodal model input
        
        Args:
            images: Input images (PIL Image, list of Images, or base64 strings)
            return_tensors: Return type (only "jax" supported)
            
        Returns:
            Preprocessed image tensors
        """
        if return_tensors != "jax":
            raise ValueError("Only 'jax' tensor type is supported")

        # Handle base64 encoded images
        if isinstance(images, (str, list)):
            images = self._decode_base64_images(images)

        # Convert to list if single image
        if not isinstance(images, list):
            images = [images]

        # Use HF image processor to get numpy arrays
        processed = self.image_processor(
            images=images,
            return_tensors="np",
        )
        pixel_values = processed["pixel_values"]

        # Convert to JAX array and normalize
        pixel_values = jnp.array(pixel_values, dtype=self.dtype)
        if self.normalize:
            pixel_values = (pixel_values - self.mean) / self.std

        return pixel_values

    def _decode_base64_images(
        self, images: Union[str, List[str]]
    ) -> Union[Image.Image, List[Image.Image]]:
        """Decode base64 encoded images to PIL Images"""
        import base64

        if isinstance(images, str):
            images = [images]

        decoded_images = []
        for img_str in images:
            try:
                img_data = base64.b64decode(img_str)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                decoded_images.append(img)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {e}")
                raise

        return decoded_images if len(decoded_images) > 1 else decoded_images[0]