"""
Adapted from FastVideo and vLLM's distributed utilities, converted to JAX.
Original SPDX-License-Identifier: Apache-2.0
"""

import dataclasses
import pickle
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.distributed import initialize, shutdown
from jax._src.distributed import global_state
import logging

logger = logging.getLogger(__name__)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: jnp.ndarray,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[jnp.ndarray]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: Input JAX array.
        num_partitions: Number of partitions to split the tensor.
        contiguous_split_chunks: If True, make each chunk contiguous in memory.
                                 JAX arrays are already contiguous, so this is a no-op.

    Returns:
        A list of JAX arrays.
    """
    last_dim = tensor.ndim - 1
    last_dim_size = divide(tensor.shape[last_dim], num_partitions)

    # Split along the last dimension
    splits = jnp.split(tensor, num_partitions, axis=last_dim)

    # JAX arrays are already contiguous, so no need for explicit contiguous() call
    return splits


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store and communication counters for JAX processes.
    Used for metadata communication between processes. For data-plane communication,
    use JAX's built-in collective operations.
    """

    rank: int
    world_size: int
    data_expiration_seconds: int = 3600  # 1 hour

    # Communication counters
    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)

    # For data expiration tracking
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank using JAX's distributed storage."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        data = pickle.dumps(obj)

        # Use JAX's global distributed store
        global_state().store.set(key.encode(), data)
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.perf_counter()))

    def expire_data(self) -> None:
        """Expire data that is older than data_expiration_seconds."""
        current_time = time.perf_counter()
        while self.entries:
            key, timestamp = self.entries[0]
            if current_time - timestamp > self.data_expiration_seconds:
                global_state().store.delete(key.encode())
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        key = f"send_to/{self.rank}/{self.recv_src_counter[src]}"
        data = global_state().store.get(key.encode())
        obj = pickle.loads(data)
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Any | None, src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks."""
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
            data = pickle.dumps(obj)
            global_state().store.set(key.encode(), data)
            self.broadcast_send_counter += 1
            self.entries.append((key, time.perf_counter()))
            return obj
        else:
            key = f"broadcast_from/{src}/{self.broadcast_recv_src_counter[src]}"
            data = global_state().store.get(key.encode())
            recv_obj = pickle.loads(data)
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = [None] * self.world_size

        # First, broadcast our own object to all others
        self.broadcast_obj(obj, self.rank)

        # Then collect objects from all ranks
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs[i] = obj
            else:
                gathered_objs[i] = self.broadcast_obj(None, i)

        return gathered_objs

    def barrier(self):
        """Synchronize all ranks using broadcast."""
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast_obj(None, self.rank)
            else:
                self.broadcast_obj(None, i)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
    ) -> "StatelessProcessGroup":
        """Create a new stateless process group using JAX's distributed backend.

        Initializes JAX's distributed system if it hasn't been initialized yet.
        """
        # Initialize JAX distributed if not already initialized
        if not global_state().is_initialized:
            initialize(f"tcp://{host}:{port}", rank=rank, world_size=world_size)

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            data_expiration_seconds=data_expiration_seconds,
        )
