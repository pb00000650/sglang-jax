"""Common utilities."""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import ipaddress
import itertools
import json
import logging
import os
import random
import re
import resource
import requests
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, Union, Literal, Tuple
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
import jax.numpy as jnp

import numpy as np
import psutil
import zmq
from fastapi.responses import ORJSONResponse
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

PRECOMPILE_DEFAULT_TOKEN_PADDINGS = [1 << i for i in range(6, 14)]
PRECOMPILE_DEFAULT_BS_PADDINGS = [1 << i for i in range(0, 9)]

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                "get_bool_env_var(%s) see non-understandable value=%s and treat as false",
                name,
                value,
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()

    if include_parent:
        with contextlib.suppress(psutil.NoSuchProcess):
            try:
                if parent_pid == os.getpid():
                    itself.kill()
                    sys.exit(0)

                itself.kill()

                # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
                # so we send an additional signal to kill them.
                itself.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                pass


def set_ulimit(target_soft_limit=65535):
    # number of open files
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning("Fail to set RLIMIT_NOFILE: %s", e)

    # stack size
    resource_type = resource.RLIMIT_STACK
    current_soft, current_hard = resource.getrlimit(resource_type)
    target_soft_limit_stack_size = 1024 * target_soft_limit
    if current_soft < target_soft_limit_stack_size:
        try:
            resource.setrlimit(resource_type, (target_soft_limit_stack_size, current_hard))
        except ValueError as e:
            logger.warning("Fail to set RLIMIT_STACK: %s", e)


def add_api_key_middleware(app, api_key: str):
    @app.middleware("http")
    async def authentication(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + api_key:
            return ORJSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    if get_bool_env_var("SGLANG_USE_MODELSCOPE") and not os.path.exists(model_path):
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
        tokenizer_path = snapshot_download(
            tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
        )
    return model_path, tokenizer_path


def configure_logger(server_args, prefix: str = ""):
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                f"Setting SGLANG_LOGGING_CONFIG_PATH from env with {SGLANG_LOGGING_CONFIG_PATH} does not exists"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in [zmq.PUSH, zmq.PUB]:
        set_send_opt()
    elif socket_type in [zmq.PULL, zmq.SUB]:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REP, zmq.REQ]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def delete_directory(dirpath):
    try:
        # This will remove the directory and all its contents
        shutil.rmtree(dirpath)
    except OSError as e:
        print(f"Warning: {dirpath} : {e.strerror}")


def dataclass_to_string_truncated(data, max_length=2048, skip_names: set[str] | None = None):
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return str(data[:half_length]) + " ... " + str(data[-half_length:])
        else:
            return str(data)
    elif isinstance(data, dict):
        return (
            "{"
            + ", ".join(
                f"'{k}': {dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={dataclass_to_string_truncated(getattr(data, f.name), max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def pyspy_dump_schedulers():
    """py-spy dump on all scheduler in a local node."""
    try:
        pid = psutil.Process().pid
        # Command to run py-spy with the PID
        cmd = f"py-spy dump --pid {pid}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        logger.error("Pyspy dump for PID %s:\n%s", pid, result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("Pyspy failed to dump PID %s. Error: %s", pid, e.stderr)


def kill_itself_when_parent_died():
    if sys.platform == "linux":
        # sigkill this process when parent worker manager dies
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def launch_dummy_health_check_server(host, port):
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Response

    app = FastAPI()

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    try:
        loop = asyncio.get_running_loop()
        logger.info("Dummy health check server scheduled on existing loop at %s:%s", host, port)
        loop.create_task(server.serve())

    except RuntimeError:
        logger.info("Starting dummy health check server at %s:%s", host, port)
        server.run()


def is_remote_url(url: str | Path) -> bool:
    """
    Check if the URL is a remote URL of the format:
    <connector_type>://<host>:<port>/<model_name>
    """
    if isinstance(url, Path):
        return False

    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    return m is not None


def retry(
    fn,
    max_retry: int,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    should_retry: Callable[[Any], bool] = lambda e: True,
):
    for try_index in itertools.count():
        try:
            return fn()
        except Exception as e:
            if try_index >= max_retry:
                raise Exception("retry() exceed maximum number of retries.") from e

            if not should_retry(e):
                raise Exception("retry() observe errors that should not be retried.") from e

            delay = min(initial_delay * (2**try_index), max_delay) * (0.75 + 0.25 * random.random())

            logger.warning(
                "retry() failed once (%sth try, maximum %s retries). Will delay %.2fs and retry. Error: %s",
                try_index,
                max_retry,
                delay,
                e,
            )
            traceback.print_exc()

            time.sleep(delay)


def lru_cache_frozenset(maxsize=128):
    def _to_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            # Not hashable; convert based on type
            if isinstance(o, (dict)):
                return frozenset((_to_hashable(k), _to_hashable(v)) for k, v in o.items())
            elif isinstance(o, set):
                return frozenset(_to_hashable(v) for v in o)
            elif isinstance(o, (list, tuple)) or (
                isinstance(o, Sequence) and not isinstance(o, (str, bytes))
            ):
                return tuple(_to_hashable(v) for v in o)
            else:
                raise TypeError(f"Cannot make hashable: {type(o)}") from None

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            h_args = tuple(_to_hashable(a) for a in args)
            h_kwargs = frozenset((_to_hashable(k), _to_hashable(v)) for k, v in kwargs.items())
            key = (h_args, h_kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # For manual cache clearing
        return wrapper

    return decorator


def cdiv(a, b):
    assert b != 0, f"b is equal to 0, b={b}"
    return (a + b - 1) // b


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


# 多模态相关工具函数
def image_to_base64(image: Union[Image.Image, str, Path], format: str = "PNG") -> str:
    """
    将图像转换为base64编码字符串
    
    Args:
        image: 可以是PIL Image对象、图像文件路径或Path对象
        format: 图像编码格式，默认为PNG
    
    Returns:
        base64编码的图像字符串
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.copy()
    
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """
    将base64编码字符串转换为PIL Image对象
    
    Args:
        base64_str: base64编码的图像字符串
    
    Returns:
        PIL Image对象
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def is_image_path(path: str) -> bool:
    """
    判断路径是否指向图像文件
    
    Args:
        path: 文件路径
    
    Returns:
        如果是图像文件则返回True，否则返回False
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    return os.path.splitext(path.lower())[1] in image_extensions


def preprocess_image(
    image: Union[Image.Image, str, Path], 
    target_size: Optional[tuple[int, int]] = None,
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    预处理图像用于模型输入
    
    Args:
        image: 图像来源（PIL Image、路径或Path对象）
        target_size: 目标尺寸 (height, width)，为None则保持原尺寸
        normalize: 是否进行归一化
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        预处理后的图像数组，形状为 (height, width, 3)
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")
    else:
        image = image.convert("RGB")
    
    # 调整尺寸
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # 归一化
    if normalize:
        img_array = (img_array - mean) / std
    
    return img_array


def parse_multimodal_prompt(prompt: str) -> list[dict[str, str]]:
    """
    解析包含图像引用的多模态提示词
    
    支持的格式:
    - 文本部分: 普通文本
    - 图像部分: <image>image_path_or_base64</image>
    
    Args:
        prompt: 包含文本和图像引用的提示词
    
    Returns:
        解析后的元素列表，每个元素为{"type": "text"|"image", "content": ...}
    """
    pattern = r'<image>(.*?)</image>'
    elements = []
    last_end = 0
    
    for match in re.finditer(pattern, prompt, re.DOTALL):
        # 添加匹配前的文本
        if match.start() > last_end:
            text = prompt[last_end:match.start()].strip()
            if text:
                elements.append({"type": "text", "content": text})
        
        # 处理图像内容
        image_content = match.group(1).strip()
        elements.append({"type": "image", "content": image_content})
        
        last_end = match.end()
    
    # 添加剩余的文本
    if last_end < len(prompt):
        text = prompt[last_end:].strip()
        if text:
            elements.append({"type": "text", "content": text})
    
    return elements


def build_multimodal_prompt(elements: list[dict[str, str]]) -> str:
    """
    将多模态元素列表构建为提示词字符串
    
    Args:
        elements: 多模态元素列表，每个元素为{"type": "text"|"image", "content": ...}
    
    Returns:
        构建的提示词字符串
    """
    prompt_parts = []
    for elem in elements:
        if elem["type"] == "text":
            prompt_parts.append(elem["content"])
        elif elem["type"] == "image":
            prompt_parts.append(f"<image>{elem['content']}</image>")
    return "".join(prompt_parts)


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


def base64_decode(encoded_str: str) -> jnp.ndarray:
    """
    JAX/Flax兼容的base64解码实现
    将base64编码字符串解码为JAX数组
    
    Args:
        encoded_str: base64编码的字符串
        
    Returns:
        解码后的字节数据，以jnp.uint8数组形式返回
    """
    # 处理填充字符
    padding = len(encoded_str) % 4
    if padding:
        encoded_str += '=' * (4 - padding)
    
    # 基础64字符映射表
    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    char_to_idx = {c: i for i, c in enumerate(base64_chars)}
    
    # 过滤非base64字符
    filtered = [c for c in encoded_str if c in char_to_idx or c == '=']
    encoded = jnp.array([char_to_idx[c] for c in filtered if c != '='], dtype=jnp.int32)
    
    # 分组处理（每4个字符一组）
    num_groups = len(encoded) // 4
    groups = encoded[:num_groups * 4].reshape(-1, 4)
    
    # 解码计算
    decoded = jnp.zeros(num_groups * 3, dtype=jnp.uint8)
    decoded = decoded.at[::3].set((groups[:, 0] << 2) | (groups[:, 1] >> 4))
    decoded = decoded.at[1::3].set(((groups[:, 1] & 0x0F) << 4) | (groups[:, 2] >> 2))
    decoded = decoded.at[2::3].set(((groups[:, 2] & 0x03) << 6) | groups[:, 3])
    
    # 处理填充导致的多余字节
    pad_count = encoded_str.count('=')
    if pad_count > 0:
        decoded = decoded[:-pad_count]
    
    return decoded


def load_image(
    image_file: Union[Image.Image, str, ImageData, bytes, jnp.ndarray],
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    加载图像并返回PIL Image对象和尺寸，兼容JAX数组输入
    
    Args:
        image_file: 多种类型的图像输入（PIL对象、路径、URL、ImageData、字节、JAX数组）
        
    Returns:
        图像对象和尺寸元组
    """
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image: Optional[Image.Image] = None
    image_size: Optional[Tuple[int, int]] = None

    try:
        if isinstance(image_file, Image.Image):
            image = image_file
            image_size = (image.width, image.height)
            
        elif isinstance(image_file, (bytes, jnp.ndarray)):
            # 处理字节数据或JAX数组
            if isinstance(image_file, jnp.ndarray):
                # 确保是uint8类型并转换为字节
                image_bytes = image_file.astype(jnp.uint8).tobytes()
            else:
                image_bytes = image_file
            image = Image.open(BytesIO(image_bytes))
            image_size = (image.width, image.height)
            
        elif isinstance(image_file, str):
            if image_file.startswith(("http://", "https://")):
                # 处理URL
                timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
                response = requests.get(image_file, stream=True, timeout=timeout)
                try:
                    response.raise_for_status()
                    image = Image.open(response.raw)
                    image.load()  # 强制加载以避免流关闭后出现问题
                    image_size = (image.width, image.height)
                finally:
                    response.close()
                    
            elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
                # 处理本地文件
                image = Image.open(image_file)
                image_size = (image.width, image.height)
                
            elif image_file.startswith("data:"):
                # 处理data URI格式
                _, image_data = image_file.split(",", 1)
                decoded = base64_decode(image_data)
                image = Image.open(BytesIO(decoded.tobytes()))
                image_size = (image.width, image.height)
                
            else:
                # 处理base64编码字符串
                decoded = base64_decode(image_file)
                image = Image.open(BytesIO(decoded.tobytes()))
                image_size = (image.width, image.height)
                
        else:
            raise ValueError(f"不支持的图像类型: {type(image_file)}")

        if image is None:
            raise ValueError(f"无法从 {image_file} 加载图像")

        return image, image_size
        
    except Exception as e:
        raise ValueError(f"加载图像时出错: {str(e)}") from e


def get_image_bytes(image_file: Union[str, bytes, ImageData, jnp.ndarray]) -> bytes:
    """
    获取图像字节数据，支持多种输入类型，使用JAX实现base64解码
    
    Args:
        image_file: 多种类型的图像输入
        
    Returns:
        图像的字节数据
    """
    try:
        # 处理ImageData类型
        if isinstance(image_file, ImageData):
            image_file = image_file.url

        if isinstance(image_file, (bytes, jnp.ndarray)):
            return image_file.tobytes() if isinstance(image_file, jnp.ndarray) else image_file
            
        elif isinstance(image_file, str):
            if image_file.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
                response = requests.get(image_file, timeout=timeout)
                response.raise_for_status()
                return response.content
                
            elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
                with open(image_file, "rb") as f:
                    return f.read()
                    
            elif image_file.startswith("data:"):
                # 处理data URI格式
                _, image_data = image_file.split(",", 1)
                decoded = base64_decode(image_data)
                return decoded.tobytes()
                
            else:
                # 处理base64编码字符串
                decoded = base64_decode(image_file)
                return decoded.tobytes()
                
        else:
            raise TypeError(f"不支持的图像类型: {type(image_file)}")
            
    except Exception as e:
        raise ValueError(f"获取图像字节时出错: {str(e)}") from e