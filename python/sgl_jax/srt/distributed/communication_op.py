from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding

from sgl_jax.srt.distributed.parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: jax.Array) -> jax.Array:
    """All-reduce the input tensor across model parallel group using JAX's psum."""
    tp_group = get_tp_group()
    if tp_group is None or len(tp_group) <= 1:
        return input_
    # 在tensor并行组内进行all-reduce（等价于跨设备求和）
    return jax.lax.psum(input_, axis_name="tensor")


def tensor_model_parallel_all_gather(input_: jax.Array, dim: int = -1) -> jax.Array:
    """All-gather the input tensor across model parallel group using JAX's all_gather."""
    tp_group = get_tp_group()
    if tp_group is None or len(tp_group) <= 1:
        return input_

    tp_size = len(tp_group)
    # 在tensor维度上收集所有分片
    gathered = jax.lax.all_gather(input_, axis_name="tensor", axis=dim, tiled=True)

    # 调整形状以合并收集的维度
    new_size = input_.shape[dim] * tp_size
    return gathered.reshape(input_.shape[:dim] + (new_size,) + input_.shape[dim + 1 :])


def tensor_model_parallel_gather(
    input_: jax.Array, dst: int = 0, dim: int = -1
) -> Optional[jax.Array]:
    """Gather the input tensor across model parallel group to destination rank."""
    tp_group = get_tp_group()
    if tp_group is None or len(tp_group) <= 1:
        return input_

    tp_size = len(tp_group)
    current_rank = jax.process_index() % tp_size  # 获取当前进程在TP组中的排名

    # 收集所有设备的张量
    gathered = jax.lax.all_gather(input_, axis_name="tensor", axis=dim, tiled=True)

    # 只在目标rank保留完整结果
    if current_rank == dst:
        new_size = input_.shape[dim] * tp_size
        return gathered.reshape(input_.shape[:dim] + (new_size,) + input_.shape[dim + 1 :])
    return None


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[jax.Array, Any]]] = None, src: int = 0
) -> Optional[Dict[Any, Union[jax.Array, Any]]]:
    """Broadcast a dictionary of tensors across model parallel group."""
    if tensor_dict is None:
        return None

    tp_group = get_tp_group()
    if tp_group is None or len(tp_group) <= 1:
        return tensor_dict

    tp_size = len(tp_group)
    current_rank = jax.process_index() % tp_size
    is_source = current_rank == src

    broadcasted_dict = {}
    for key, value in tensor_dict.items():
        if isinstance(value, jax.Array):
            # 从源rank广播到所有TP设备
            broadcasted = jax.lax.broadcast_one_to_all(
                value, axis_name="tensor", is_source=is_source
            )
            broadcasted_dict[key] = broadcasted
        else:
            # 非张量值直接传递
            broadcasted_dict[key] = value
    return broadcasted_dict
