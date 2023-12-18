# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import datetime
import os
import threading
from functools import partial
from typing import List

import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    fake_initialize_model_parallel,
)
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    SamplingParam,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

__all__ = ["init_model", "pred_by_generation"]

"""
This is the script to run GPT text generation.
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(
        self,
    ):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def remove_padded_prompts(response, nb_paddings):
    result = {}
    for k, v in response.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result


def nemo_init_model(cfg: OmegaConf):
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        # callbacks=[CustomProgressBar()],
    )

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get("pipeline_model_parallel_split_rank", -1) < 0
    ):
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file
        model_config = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get(
                "tensor_model_parallel_size", 1
            )
            cfg.pipeline_model_parallel_size = model_config.get(
                "pipeline_model_parallel_size", 1
            )
            cfg.pipeline_model_parallel_split_rank = model_config.get(
                "pipeline_model_parallel_split_rank", 0
            )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.gpt_model_file):
        save_restore_connector.model_extracted_dir = cfg.gpt_model_file

    pretrained_cfg = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    OmegaConf.set_struct(pretrained_cfg, True)
    with open_dict(pretrained_cfg):
        pretrained_cfg.sequence_parallel = False
        pretrained_cfg.activations_checkpoint_granularity = None
        pretrained_cfg.activations_checkpoint_method = None
        pretrained_cfg.precision = trainer.precision
        if pretrained_cfg.get("mcore_gpt", False):
            # with dist checkpointing we can use the model parallel config specified by the user
            pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            pretrained_cfg.pipeline_model_parallel_size = (
                cfg.pipeline_model_parallel_size
            )
        if trainer.precision == "16":
            pretrained_cfg.megatron_amp_O2 = False
        elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get(
            "megatron_amp_O2", False
        ):
            pretrained_cfg.megatron_amp_O2 = True
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f"cuda:{trainer.local_rank}",  # map_location is needed for converted models
        )

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model, trainer


def nemo_generate(
    model, prompts: List[str], batch_size: int, trainer, cfg: OmegaConf
) -> List[str]:
    cfg_infer = OmegaConf.to_container(cfg.inference)

    cfg_infer["batch_size"] = batch_size
    ds = RequestDataSet(prompts)
    request_dl = DataLoader(dataset=ds, batch_size=batch_size)
    model.set_inference_config(cfg_infer)
    response = trainer.predict(model, request_dl)
    return response
