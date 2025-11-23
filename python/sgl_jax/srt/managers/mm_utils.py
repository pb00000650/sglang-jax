import hashlib
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.multimodal import gpu_tensor_hash
from sgl_jax.srt.utils import flatten_nested_list


def hash_feature(f: Any) -> int:
    """Hash multimodal features"""
    if isinstance(f, list):
        if isinstance(f[0], jnp.ndarray):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        return data_hash(arr.tobytes())
    elif isinstance(f, jnp.ndarray):
        return tensor_hash([f])
    return data_hash(pickle.dumps(f))


def data_hash(data: Any) -> int:
    """Hash raw data bytes"""
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list: Any) -> int:
    """Hash JAX tensors or tensor lists"""
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [x.reshape(-1) for x in tensor_list if isinstance(x, jnp.ndarray)]
        tensor = jnp.concatenate(tensor_list)
    else:
        tensor = tensor_list

    if isinstance(tensor, jnp.ndarray):
        if tensor.device().platform == "gpu":
            return gpu_tensor_hash(tensor)

        tensor_cpu = jax.device_get(tensor).astype(jnp.float32)
        return data_hash(tensor_cpu.tobytes())

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")
