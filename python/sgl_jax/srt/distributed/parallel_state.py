from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import broadcast_one_to_all

# 从代码库中获取已有的mesh信息（参考test_model_loader和tp_worker中的实现）
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


# 全局变量存储并行组信息
_TP_GROUP: Optional["JAXGroupCoordinator"] = None


class JAXGroupCoordinator:
    """JAX版本的并行组协调器，适配sglang-jax的mesh架构"""

    def __init__(self, mesh: Mesh, axis_name: str = "tensor"):
        self.mesh = mesh
        self.axis_name = axis_name
        self.world_size = mesh.shape.get(axis_name, 1)
        self.rank_in_group = jax.process_index() % self.world_size
        self.is_single_device = self.world_size == 1

        # 初始化分片规范
        self.sharding = NamedSharding(mesh, P(None))
        if self.axis_name in mesh.axis_names:
            self.axis_index = mesh.axis_names.index(axis_name)

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    def all_reduce(self, input_: jax.Array) -> jax.Array:
        """跨张量并行组执行all-reduce"""
        if self.is_single_device:
            return input_
        return jax.lax.psum(input_, axis_name=self.axis_name)

    def all_gather(self, input_: jax.Array, dim: int = -1) -> jax.Array:
        """跨张量并行组执行all-gather"""
        if self.is_single_device:
            return input_

        # 收集所有分片并合并维度
        gathered = jax.lax.all_gather(input_, axis_name=self.axis_name, axis=dim, tiled=True)

        # 调整形状以合并收集的维度
        new_size = input_.shape[dim] * self.world_size
        return gathered.reshape(input_.shape[:dim] + (new_size,) + input_.shape[dim + 1 :])

    def gather(self, input_: jax.Array, dst: int = 0, dim: int = -1) -> Optional[jax.Array]:
        """将张量收集到目标rank"""
        if self.is_single_device:
            return input_

        # 收集所有设备的张量
        gathered = jax.lax.all_gather(input_, axis_name=self.axis_name, axis=dim, tiled=True)

        # 只在目标rank返回完整结果
        if self.rank_in_group == dst:
            new_size = input_.shape[dim] * self.world_size
            return gathered.reshape(input_.shape[:dim] + (new_size,) + input_.shape[dim + 1 :])
        return None

    def broadcast_tensor_dict(
        self, tensor_dict: Optional[Dict[Any, Union[jax.Array, Any]]] = None, src: int = 0
    ) -> Optional[Dict[Any, Union[jax.Array, Any]]]:
        """广播张量字典到所有并行组成员"""
        if tensor_dict is None or self.is_single_device:
            return tensor_dict

        is_source = self.rank_in_group == src
        broadcasted_dict = {}

        for key, value in tensor_dict.items():
            if isinstance(value, jax.Array):
                # 从源rank广播到所有设备
                broadcasted = jax.lax.broadcast_one_to_all(
                    value, axis_name=self.axis_name, is_source=is_source
                )
                broadcasted_dict[key] = broadcasted
            else:
                # 非张量值直接传递
                broadcasted_dict[key] = value

        return broadcasted_dict

    def barrier(self) -> None:
        """并行组内同步屏障"""
        if self.is_single_device:
            return
        # 使用空数组的all-reduce实现屏障
        jax.lax.psum(jnp.array([0]), axis_name=self.axis_name).block_until_ready()


def initialize_tensor_model_parallel(
    tensor_model_parallel_size: int = 1, backend: Optional[str] = None
) -> None:
    """初始化张量并行组，适配sglang-jax的设备网格创建逻辑"""
    global _TP_GROUP

    # 参考test_model_loader和scheduler中的mesh创建方式
    mesh = create_device_mesh(
        ici_parallelism=[-1, tensor_model_parallel_size], dcn_parallelism=[1, 1]
    )

    _TP_GROUP = JAXGroupCoordinator(mesh, axis_name="tensor")


def get_tp_group() -> JAXGroupCoordinator:
    """获取张量并行组，与原PyTorch版本接口保持一致"""
    assert _TP_GROUP is not None, "Tensor parallel group not initialized"
    return _TP_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


# 保持与原代码相同的接口函数
def tensor_model_parallel_all_reduce(input_: jax.Array) -> jax.Array:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: jax.Array, dim: int = -1) -> jax.Array:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: jax.Array, dst: int = 0, dim: int = -1
) -> Optional[jax.Array]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[jax.Array, Any]]] = None, src: int = 0
) -> Optional[Dict[Any, Union[jax.Array, Any]]]:
    if tensor_dict is None:
        return None
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
