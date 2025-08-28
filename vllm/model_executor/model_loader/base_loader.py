# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)

from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import torch_xla.core.xla_model as xm
from vllm.logger import init_logger
logger = init_logger(__name__)

class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        logger.info("base_loader::load_model started")
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            # Compiling the model for loading weights.
            model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            cc = CompilerConfig()
            cc.push_outputs_to_cpu = False
            cc.enable_consteval = True
            cc.consteval_parameters = True
            # cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
            options = BackendOptions()
            options.compiler_config = cc

            model = torch.compile(
                model,
                # model.to(xm.xla_device()),
                backend="tt-experimental",
                dynamic=False,
                options=options
            )

            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        temp = model.eval()
        logger.info("base_loader::load_model completed")
        return temp
