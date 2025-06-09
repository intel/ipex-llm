#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from vllm.logger import init_logger
from typing import Dict, Optional, Any, Union, Type
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.utils import Counter
from vllm.config import VllmConfig
from ipex_llm.vllm.xpu.model_convert import _ipex_llm_convert
from vllm.usage.usage_lib import UsageContext
from vllm.engine.metrics import StatLoggerBase
from vllm.engine.multiprocessing.engine import MQLLMEngine
import signal
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.config import CompilationConfig
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
from vllm import envs
from vllm.v1.engine.async_llm import AsyncLLM
import os

logger = init_logger(__name__)


class IPEXLLMAsyncLLMEngine(AsyncLLMEngine):
    _is_converted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        load_in_low_bit: str = "sym_int4",
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_engine_args(engine_args=engine_args, engine_config=engine_config,
                                        start_engine_loop=start_engine_loop,
                                        usage_context=usage_context, stat_loggers=stat_loggers)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]]=None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "AsyncLLMEngine":
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_requests=disable_log_requests,
            disable_log_stats=disable_log_stats,
        )


class IPEXLLMAsyncV1Engine(AsyncLLM):
    _is_converted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        load_in_low_bit: str = "sym_int4",
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,  # noqa
    ) -> "AsyncLLM":
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_engine_args(engine_args=engine_args, engine_config=engine_config,
                                        start_engine_loop=start_engine_loop,
                                        usage_context=usage_context, stat_loggers=stat_loggers)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]]=None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "AsyncLLM":
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_requests=disable_log_requests,
            disable_log_stats=disable_log_stats,
        )


class IPEXLLMClass(LLM):

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[HfOverrides]=None,
        mm_processor_kwargs: Optional[dict[str, Any]]=None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]]=None,
        load_in_low_bit: str = "sym_int4",
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        # Logic to switch between engines is done at runtime instead of import
        # to avoid import order issues
        self.engine_class = self.get_engine_class()
        # print("!!! ", load_in_low_bit)
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS,
            load_in_low_bit=load_in_low_bit)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            return IPEXLLMLLMV1Engine
        return IPEXLLMLLMEngine


class IPEXLLMLLMV1Engine(V1LLMEngine):
    _is_converted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        enable_multiprocessing: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.

        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_engine_args(engine_args,
                                        usage_context,
                                        stat_loggers,
                                        enable_multiprocessing)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        disable_log_stats: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=disable_log_stats
        )


class IPEXLLMLLMEngine(LLMEngine):
    _is_converted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_engine_args(engine_args, usage_context, stat_loggers)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]]=None,
        disable_log_stats: bool = False,
        load_in_low_bit: str = "sym_int4",
    ) -> "LLMEngine":
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=disable_log_stats
        )


class IPEXLLMMQLLMEngine(MQLLMEngine):
    _is_converted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs,
                         usage_context: UsageContext, ipc_path: str, load_in_low_bit: str):
        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_engine_args(engine_args, usage_context, ipc_path)

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig,
                         usage_context: UsageContext,
                         disable_log_requests: bool, disable_log_stats: bool,
                         ipc_path: str, load_in_low_bit: str) -> "MQLLMEngine":

        if not cls._is_converted:
            _ipex_llm_convert(load_in_low_bit)
            cls._is_converted = True
        return super().from_vllm_config(
            vllm_config=vllm_config,
            ipc_path=ipc_path,
            usage_context=usage_context,
            disable_log_requests=disable_log_requests,
            disable_log_stats=disable_log_stats,
        )

from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)


def signal_handler(*_) -> None:
    raise KeyboardInterrupt("MQLLMEngine terminated")  # noqa


def run_mp_engine(vllm_config: VllmConfig, usage_context: UsageContext,
                  ipc_path: str, disable_log_stats: bool,
                  disable_log_requests: bool, load_in_low_bit, engine_alive):
    try:
        # Ensure we can serialize transformer config before spawning
        maybe_register_config_serialize_by_value()

        engine = IPEXLLMMQLLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=disable_log_stats,
            disable_log_requests=disable_log_requests,
            load_in_low_bit=load_in_low_bit,
            ipc_path=ipc_path)

        signal.signal(signal.SIGTERM, signal_handler)

        engine.start()

    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e  # noqa

if os.getenv("VLLM_USE_V1") == "1":
    IPEXLLMAsyncLLMEngine = IPEXLLMAsyncV1Engine
