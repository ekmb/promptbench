# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import os
import time
from typing import Dict, List, Optional, Union

import bigbench.api.model as model
import numpy as np
import scipy
import torch
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import MegatronGPTPromptLearningModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_utils import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy,NLPSaveRestoreConnector
from nemo.utils.app_state import AppState
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence

from .huggingface_models import (gpt2_embedding_params,
                                 gpt2_non_embedding_params)

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

# squelch some excessive logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# ----------------------------------------------------------------------------
class _NMGPTModel:
    def __init__(
        self,
        nemo_model: str,
        num_gpus,
        num_nodes,
        log_likelihood_norm=True,
        use_megatron_amp_o2=None,
        precision=None,
        p_tuning_taskname=None,
        p_tuning_model_path=None,
        model_max_len=None,
        use_flash_attention=False,
    ):
        trainer = Trainer(
            strategy=NLPDDPStrategy(),
            accelerator="gpu",
            precision=precision,
            gpus=num_gpus,
            num_nodes=num_nodes,
        )

        self.p_tuning_taskname = p_tuning_taskname
        save_restore_connector = NLPSaveRestoreConnector()

        if os.path.isdir(nemo_model):
            save_restore_connector._model_extracted_dir = nemo_model

        assert os.path.exists(
            nemo_model
        ), f"Invalid NeMo GPT-3 checkpoint path: {nemo_model}"

        language_model_cfg = MegatronGPTModel.restore_from(
            nemo_model,
            return_config=True,
            save_restore_connector=save_restore_connector,
            trainer=trainer,
        )

        with open_dict(language_model_cfg):
            language_model_cfg.activations_checkpoint_num_layers = None
            language_model_cfg.sequence_parallel = False
            language_model_cfg.activations_checkpoint_granularity = None
            language_model_cfg.activations_checkpoint_method = None
            language_model_cfg.precision = trainer.precision
            language_model_cfg.megatron_amp_o2 = use_megatron_amp_o2
            if trainer.precision == "16":
                language_model_cfg.megatron_amp_o2 = False

            language_model_cfg.use_flash_attention = use_flash_attention

            language_model_cfg.apply_query_key_layer_scaling = False

        app_state = AppState()
        if num_gpus > 1:
            app_state.model_parallel_size = num_gpus
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=num_gpus,
                pipeline_model_parallel_size_=1,
                pipeline_model_parallel_split_rank_=0,
            )

        if self.p_tuning_taskname:
            # Update frozen GPT model path if it is given in case it has changed
            prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
                p_tuning_model_path, trainer=trainer, return_config=True
            )
            #

            for task_template in prompt_learning_cfg.task_templates:
                if self.p_tuning_taskname == task_template.taskname:
                    task_template.prompt_template = "<|VIRTUAL_PROMPT_0|> {input}"  ### currently only support this format
                    task_template.truncate_field = None
                    task_template.answer_field = None
            prompt_learning_cfg.language_model_path = nemo_model

            self._model = MegatronGPTPromptLearningModel.restore_from(
                restore_path=p_tuning_model_path,
                trainer=trainer,
                override_config_path=prompt_learning_cfg,
            )
            self._model.freeze()
        else:
            assert (
                app_state.model_parallel_size == app_state.tensor_model_parallel_size
            ), "Num GPUs should be equal to TP size"
            self._model = MegatronGPTModel.restore_from(
                restore_path=nemo_model,
                trainer=trainer,
                override_config_path=language_model_cfg,
                save_restore_connector=save_restore_connector,
                map_location=f"cuda:{trainer.local_rank}",  # map_location is needed for converted models
            )
        try:
            self._model.model.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass
        self._model.precision = precision
        
        self._model.megatron_amp_o2 = use_megatron_amp_o2
        self._model.eval()

        self._cfg = language_model_cfg
        print(f'\n{OmegaConf.to_yaml(self._model._cfg)}')
        
        self._model_name = nemo_model
        self._model_type = "nemo-megatron-gpt3"
        self._tokenizer = self._model.tokenizer

        if isinstance(self._tokenizer, SentencePieceTokenizer):
            pass
        else:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if isinstance(self._tokenizer, SentencePieceTokenizer):
            self._pad_id = (
                self._tokenizer.pad_id
                if self._tokenizer.pad_id is not None
                else self._tokenizer.unk_id
            )
        else:
            self._pad_id = self._tokenizer.pad_id

        self.model_max_len = (
            int(model_max_len)
            if model_max_len
            else language_model_cfg.max_position_embeddings
        )
        self.log_likelihood_norm = log_likelihood_norm

    @property
    def name(self) -> str:
        return self._model_name

    def _adjust_length_to_model(self, max_length: int) -> int:
        """clamp maximum length for compatibility with model

        Args:
        max_length: requested maximum length

        Returns:
        adjusted maximum length
        """

        if max_length < 0 < self.model_max_len:
            length = self.model_max_len
        elif 0 < self.model_max_len < max_length:
            length = self.model_max_len
        else:
            length = max_length

        return length

    def _maybe_truncate_input(self, input_ids: List, verbose: bool = True) -> List:
        for ind in range(len(input_ids)):
            if self.model_max_len < len(input_ids[ind]) + 1:
                context_len = len(input_ids[ind])
                if verbose:
                    print(
                        f"WARNING: input context too long for model {self._model_name}"
                    )
                    print(
                        f"context len: {context_len}, max_model_len: {self.model_max_len}"
                    )
                    print(
                        f"input context will be truncated to length {self.model_max_len}"
                    )
                input_ids[ind] = input_ids[ind][-self.model_max_len + 1 :]

        return input_ids

    def _preprocess_context(self, context: str, truncate: bool = True):
        """preprocess input context for model

        Args:
        context: input context string

        Returns:
        encoeded context for input to model
        """

        ids = [self._tokenizer.text_to_ids(context)]
        if truncate:
            return torch.tensor(self._maybe_truncate_input(ids))

        return torch.tensor(ids)

    def generate(self, context: str, sampling_params, length_params) -> List[str]:
        """Generates outputs from language model.

        Args:
        context: input context
        max_length: maximum output length
        temperature: 1.0 has no effect, lower tend toward greedy sampling
        top_k: tokens to consider at each step as context
        top_p: nucleus sampling threshold
        num_outputs: number of outputs to generate

        Returns:
        list of generated responses

        Raises:
        ValueError if max_length is invalid
        """
        if length_params["max_length"] < 1:
            raise ValueError("max_length must be > 0")
        context = context.rstrip(
            " "
        )  # small tweaks to remove the ending space in input

        input_ids = self._preprocess_context(context)
        context_len = len(input_ids[0])

        total_length = self._adjust_length_to_model(
            max_length=length_params["max_length"] + context_len
        )
        max_length = total_length - context_len

        input_ids = torch.cat(
            (input_ids, torch.tensor([[self._pad_id] * max_length])), -1
        )

        inputs = (input_ids.cuda(), torch.tensor([context_len]).cuda())

        if self.p_tuning_taskname:
            inputs = [{"taskname": self.p_tuning_taskname, "input": context}]
        inputs = [self._tokenizer.ids_to_text(inputs[0][0].tolist())]
        output = self._model.generate(
            inputs=inputs, length_params=length_params, sampling_params=sampling_params
        )

        output_sequences = output["sentences"]

        generated_sequences = []

        # Multiple sampling is not supported in NeMo
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # remove prompt
            if self.p_tuning_taskname:
                start_index = generated_sequence.find(context)
                if start_index == -1:
                    raise ValueError(
                        f"Context not found in output sequence: context: {context} generated sequence: {generated_sequence}"
                    )
                else:
                    text = generated_sequence[start_index + len(context) :]
                    generated_sequences.append(text)
            else:
                # text = generated_sequence[len(context) :]
                text = self._tokenizer.ids_to_text(output['token_ids'][0][len(self._tokenizer.text_to_ids(context)) + 1:]) # replaced the above to deal with unknown tokens, if there's an unknown token slice above is incorrect
                generated_sequences.append(text)
        return generated_sequences

    def score(
        self,
        inputs: Union[List[str], str],
        targets: Union[List[str], str],
        mask_token_id=-100,
    ) -> List[float]:
        """Scores one or a batch of example targets given their inputs.

        Args:
        inputs: input context
        targets:  targets to be scored

        Returns:
        list of log probabilities for each target given the input.
        """

        if self.p_tuning_taskname:
            raise ValueError(
                "Classification tasks for P-tuned models are not yet supported"
            )

        if isinstance(inputs, str):
            input_list = [inputs.rstrip(" ")]
            target_list = [
                targets
                if targets.startswith(" ") or targets.startswith("\n")
                else " " + targets
            ]
        else:
            input_list = [inp.rstrip(" ") for inp in inputs]
            target_list = [
                tgt if tgt.startswith(" ") or tgt.startswith("\n") else " " + tgt
                for tgt in targets
            ]

        tokenizer = self._tokenizer
        ragged_inputs_ids = [tokenizer.text_to_ids(x) for x in input_list]
        inputs_and_targets = [inp + tar for inp, tar in zip(input_list, target_list)]
        ragged_inputs_and_targets_ids = [
            tokenizer.text_to_ids(x) for x in inputs_and_targets
        ]
        ragged_targets_ids = []
        for idx, inp in enumerate(ragged_inputs_ids):
            ragged_targets_ids.append(ragged_inputs_and_targets_ids[idx][len(inp) :])
            assert (
                len(ragged_targets_ids[-1]) < self.model_max_len
            ), "Target sequence length is longger than max sequence length!"

        ragged_inputs_and_targets_ids = self._maybe_truncate_input(
            ragged_inputs_and_targets_ids,
            verbose=False,
        )

        inputs_and_targets = [
            tokenizer.ids_to_text(x) for x in ragged_inputs_and_targets_ids
        ]

        length_params = {
            "max_length": 1,
            "min_length": 0,
        }
        sampling_params = {
            "use_greedy": True,
            "temperature": 1,
            "top_k": 0,
            "top_p": 0.9,
            "repetition_penalty": False,
            "add_BOS": False,
            "all_probs": True,
            "compute_logprob": True,
        }
        output = self._model.generate(
            inputs=inputs_and_targets,
            length_params=length_params,
            sampling_params=sampling_params,
        )
        output_log_probs = output["logprob"]

        scores = []
        for log_prob, target_ids in zip(output_log_probs, ragged_targets_ids):
            log_prob = log_prob.cpu().numpy()
            log_prob = log_prob[:-1]
            target_len = len(target_ids)
            if self.log_likelihood_norm:
                scores.append(sum(log_prob[-target_len:]) / len(log_prob[-target_len:]))
            else:
                scores.append(sum(log_prob[-target_len:]))

        return scores


# ----------------------------------------------------------------------------
class BIGBenchNMModel(model.Model):
    def __init__(
        self,
        nemo_model: str,
        tokens_to_generate=256,
        show_progress=True,
        num_gpus=1,
        num_nodes=1,
        top_p=0.9,
        top_k=0,
        log_likelihood_norm=True,
        use_megatron_amp_o2=None,
        precision=None,
        p_tuning_taskname=None,
        p_tuning_model_path=None,
        model_max_len=None,
        greedy=True,
        temperature=1.0,
        repetition_penalty=1.2,
        add_bos=False,
        all_probs=False,
        compute_logprob=False,
        use_flash_attention=False,
    ):
        self._model = _NMGPTModel(
            nemo_model=nemo_model,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            log_likelihood_norm=log_likelihood_norm,
            use_megatron_amp_o2=use_megatron_amp_o2,
            precision=precision,
            p_tuning_taskname=p_tuning_taskname,
            p_tuning_model_path=p_tuning_model_path,
            model_max_len=model_max_len,
            use_flash_attention=use_flash_attention,
        )

        self._tokens_to_generate = tokens_to_generate
        self._show_progress = show_progress
        self._model_name = os.path.basename(nemo_model).split(".")[0]
        self.greedy = greedy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.add_bos = add_bos
        self.all_probs = all_probs
        self.compute_logprob = compute_logprob

    def model_data(self) -> model.ModelData:
        cfg = self._model._cfg
        non_embedding_params = gpt2_non_embedding_params(
            n_layers=cfg.num_layers,
            d_hidden=cfg.hidden_size,
            d_mlp=cfg.ffn_hidden_size,
        )
        embedding_params = gpt2_embedding_params(
            n_layers=cfg.num_layers,
            d_hidden=cfg.hidden_size,
            d_mlp=cfg.ffn_hidden_size,
        )
        model_info = model.ModelData(
            model_family="NeMo",
            model_name=self._model_name,
            total_params=non_embedding_params + embedding_params,
            non_embedding_params=non_embedding_params,
            flop_matched_non_embedding_params=non_embedding_params,
            training_batch_size=1,
            training_steps=1,
            description="NeMo Megatron Models",
            decoding_params={},
        )
        return model_info

    def generate_text(
        self,
        inputs: Union[str, List[str]],
        max_length: int = 0,  # TODO(guyga) should probably be the last argument ### THIS IS tokens_to_generate, renaming it would break many things across the repo
        stop_string: str = None,
        output_regex: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Generates text for given inputs.

        Args:
          inputs: String or list of strings as inputs for model.
          max_length: Maximum string length of output, if 0 uses tokens_to_generate passed
            to constructor
          stop_string: If specified, model output will be truncated to the shortest
            string which includes stop_string.
          output_regex: If specified, the first match to the python regular
            expression output_regex in the model output will be returned. If there is
            no match, an empty string will be returned.

        Returns:
          String or list of strings generated by model.

        Raises:
          ValueError if tokens_to_generate is invalid
        """
        tokens_to_generate = max_length or self._tokens_to_generate

        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        generated = []
        last = start = time.time()

        for idx, inpt in enumerate(input_list):
            length_params: LengthParam = {
                "max_length": tokens_to_generate,
                "min_length": 0,
            }

            sampling_params: SamplingParam = {
                "use_greedy": self.greedy,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "add_BOS": self.add_bos,
                "all_probs": self.all_probs,
                "compute_logprob": self.compute_logprob,
                "end_strings": ["<|endoftext|>", "<extra_id_1>"],
            }

            text = self._model.generate(
                context=inpt if inpt else "<|endoftext|>",
                length_params=length_params,
                sampling_params=sampling_params,
            )[0]
            generated.append(text)

            if self._show_progress and time.time() - last > 60:
                print(
                    f"generated {idx} of {len(input_list)} outputs in {int(time.time() - start)} secs..."
                )
                last = time.time()

        if isinstance(inputs, str):
            generated = generated[0]

        generated = model_utils.postprocess_output(
            generated, tokens_to_generate, stop_string, output_regex
        )

        return generated

    def flatten_multiple_choice_examples(self, inputs, targets):
        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(inputs, targets)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        return flat_idx, flat_inputs, flat_choices

    def cond_log_prob(
        self,
        inputs: Union[str, List[str]],
        targets: Union[List[str], List[List[str]]],
        batch_size: int = 4,
        absolute_normalization: Optional[bool] = False,
    ) -> Union[List[float], List[List[float]]]:
        """Computes conditional log probabilities of targets given inputs.

        Args:
          `inputs`: A single string input or a list of string inputs.

          `targets`: Possible string outputs for each input. If input is a
             string, this is a list `[t_1, t_2, ..., t_n]` of possible string
             outputs. If input is a list of strings, then this is a nested
             list `[[t_1, t_2, ..., t_n], ...]` with length equal to `len(inputs)`.

           `absolute_normalization`: When True, the function returns the log
             probability of unconstrained generation or the target sequence. When
             False (default), log probabilities are normalized so that the probabilities
             of generating `targets` sum to 1. Note that setting `absolute_normalization`
             to True restricts the class of models that can be evaluated to those that
             can assign absolute probabilities to sequences.

           Returns:
             If a single string input is provided, returns a list of
             log-probabilities `[lp_1, lp_2, ..., lp_n]` predicted by the model,
             where  `lp_i = log(prob(t_i | input)`  is the conditional log-prob
             to generate target `t_i` given input. If a list of string inputs
             was provided, returns a list of such elements of the form
             `[[lp_1, lp_2, ..., lp_n], ...]`, where each element contains the
             log-probabilities for the corresponding input and targets.
             In this case, the length of the returned list is `len(input)`.
        """

        if isinstance(inputs, str):
            input_list = [inputs]
            target_list = [targets]
        else:
            input_list = inputs
            target_list = targets

        flat_idx, flat_inputs, flat_choices = self.flatten_multiple_choice_examples(
            inputs=input_list, targets=target_list
        )
        num_examples = len(flat_idx)
        flat_scores = []
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx : min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx : min(idx + batch_size, num_examples)]

            batch_scores = self._model.score(batch_inputs, batch_choices)
            flat_scores += batch_scores

        scores = [[] for _ in range(len(input_list))]

        for idx, score in zip(flat_idx, flat_scores):
            scores[idx[0]].append(score)

        if not absolute_normalization:
            scores = [
                list(score_row - scipy.special.logsumexp(score_row))
                for score_row in scores
            ]

        if isinstance(inputs, str):
            scores = scores[0]

        return scores


# ----------------------------------------------------------------------------
def set_seed(seed: int):
    """sets random number generator seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """Use this to run a simple test of the HF model types."""

    local_rank = int(os.environ["LOCAL_RANK"], 0)

    def _print(*args):
        if local_rank == 0:
            print(*args)

    # test a few gpt models
    for nemo_model in [
        "/workspace/BIG-bench/notebooks/models/megatron_125M_gpt.nemo",
        "/workspace/BIG-bench/notebooks/models/megatron_gpt_1.3B_tp2.nemo",
    ]:
        _print("-" * 80)
        _print(f"model: {nemo_model}")
        set_seed(42)

        model = BIGBenchNMModel(
            nemo_model=nemo_model, use_megatron_amp_o2=True, precision="bf16"
        )
        prompt = "It was the best of times, it was"
        response = model.generate_text(
            inputs=prompt,
            max_length=32,
            stop_string="</s>",
        )

        _print(f"prompt: {prompt}")
        _print(f"response: {response}")

        prompts = ["These are the times that", "Stately, plump Buck Mulligan"]
        responses = model.generate_text(inputs=prompts, max_length=32, stop_string=".")

        for p, r in zip(prompts, responses):
            _print(f"prompt: {p}")
            _print(f"response: {r}")

        # for testing, the prompt here has no trailing space, while the
        # next scoring example has a trailing space
        prompt = (
            f"What color is the sky? Answer: blue\n" f"What color is grass? Answer:"
        )
        choices = ("red", "blue", "green")

        scores = model.cond_log_prob(inputs=prompt, targets=choices)

        _print("\n")
        _print(f"prompt:\n{prompt}")
        _print(f"scores:")
        for c, s in zip(choices, scores):
            _print(f"  {c:>8}: {s:0.2f}")

        prompts = [
            f"What color is the sky? Answer: blue\n" f"What color is grass? Answer: ",
            f"1+1=2\n" f"2+2=",
        ]
        choices = [("red", "blue", "green", "green blue"), ("1", "2", "3", "4")]

        scores = model.cond_log_prob(inputs=prompts, targets=choices)

        for p, c, s in zip(prompts, choices, scores):
            _print("\n")
            _print(f"prompt:\n{p}")
            _print(f"scores:")
            for ci, si in zip(c, s):
                _print(f"  {ci:>8}: {si:0.2f}")