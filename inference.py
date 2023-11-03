# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

try:
    import openai
except ImportError:
    print("OpenAI API is not installed, please install it by running: pip install openai")

import sys
import os
import math
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "nemo_utils"))

try:
    # to use Nemo generation directly
    import os
    from omegaconf import OmegaConf
    from megatron_gpt_eval import nemo_init_model, nemo_generate
    NEMO_AVAILABLE = True
except:
    NEMO_AVAILABLE = False

try:
    # to use TRT-LLM
    TRT_AVAILABLE = True
    from pytriton.client import ModelClient
    from trt_infer import query_llm
except:
    TRT_AVAILABLE = False

from config import LABEL_SET, LABEL_TO_ID, NEMO_PROMPT
from tqdm import tqdm
from typing import List
from collections import defaultdict
from joblib import Parallel, delayed

"""
This clss implements the inference of the model (including create the model).
"""


class Inference(object):

    def __init__(self, args):
        self.error_analysis = False
        self.args = args
        self.model = args.model
        self.nemo_cfg, self.nemo_trainer = None, None
        self.create_model()

    def create_model(self):
        """
        ChatGPT is a special case, we use the openai api to create the model.
        """

        if self.model not in ['chatgpt', 'gpt4']:
            import torch
            import os

            """
            Here you can add you own model.
            """

            if self.model == 'google/flan-t5-large':
                from transformers import T5Tokenizer, T5ForConditionalGeneration

                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model, device_map="cuda")
                self.pipe = T5ForConditionalGeneration.from_pretrained(self.model, device_map="cuda")

            elif self.model == 'EleutherAI/gpt-neox-20b':
                from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

                self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(
                    self.model, device_map="auto")
                self.pipe = GPTNeoXForCausalLM.from_pretrained(
                    self.model, device_map="auto", torch_dtype=torch.float16)

            # elif self.model.lower() == 'facebook/opt-66b':
            #     from transformers import AutoModelForCausalLM, AutoTokenizer

            #     # the fast tokenizer currently does not work correctly
            #     self.tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto", use_fast=False)
            #     self.pipe = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)

            elif self.model.lower() in ["llama-13b", "llama2-13b", 'llama2-13b-chat', 'llama2-7b', 'llama2-7b-chat']:

                from transformers import LlamaForCausalLM, LlamaTokenizer

                model_dir = os.path.join(self.args.model_dir, self.model)

                self.tokenizer = LlamaTokenizer.from_pretrained(
                    model_dir, device_map="auto")
                self.pipe = LlamaForCausalLM.from_pretrained(
                    model_dir, device_map="auto", torch_dtype=torch.float16)

            elif self.model.lower() in ["vicuna-13b", "vicuna-13b-v1.3"]:

                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_dir = os.path.join(self.args.model_dir, self.model)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir, device_map="auto", use_fast=False)
                self.pipe = AutoModelForCausalLM.from_pretrained(
                    model_dir, device_map="auto", torch_dtype=torch.float16)

            elif self.model == "google/flan-ul2":

                from transformers import T5ForConditionalGeneration, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.pipe = T5ForConditionalGeneration.from_pretrained(
                    self.model, torch_dtype=torch.bfloat16, device_map="auto")

            elif self.model == "tiiuae/falcon-40b-instruct":
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.pipe = AutoModelForCausalLM.from_pretrained(
                    self.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",)

            elif self.model == "cerebras/Cerebras-GPT-13B":
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model, device_map="auto")
                self.pipe = AutoModelForCausalLM.from_pretrained(
                    self.model, device_map="auto", torch_dtype=torch.float16)

            elif self.model == "databricks/dolly-v1-6b":
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    "databricks/dolly-v1-6b", device_map="auto", padding_side="left")
                self.pipe = AutoModelForCausalLM.from_pretrained(
                    "databricks/dolly-v1-6b", device_map="auto", torch_dtype=torch.float16)

            elif self.model == "nemo":
                if self.args.nemo_use_server:
                    if not TRT_AVAILABLE:
                        raise ImportError("TRT is not installed")

                    # the model is loaded on the server, we only need to send the request
                    return

                if not NEMO_AVAILABLE:
                    raise ImportError("NeMo is not installed")
                
                dir_path = os.path.dirname(os.path.realpath(__file__))
                cfg = os.path.abspath(f"{dir_path}/nemo_utils/nemo_cfgs/megatron_gpt_inference.yaml")

                cfg = OmegaConf.load(cfg)
                cfg.inference.tokens_to_generate = self.args.generate_len
                cfg.inference.batch_size = self.args.batch_size
                
                # update NeMo config if non None nemo-* args provided
                for arg_name in vars(self.args):
                    if arg_name.startswith("nemo_") and getattr(self.args, arg_name) is not None:
                        if arg_name == "nemo_model_path":
                            if self.args.nemo_model_path is None or not os.path.exists(self.args.nemo_model_path):
                                raise ValueError(f"NeMo model path {self.args.nemo_model_path} does not exist")
                            else:
                                cfg.gpt_model_file = self.args.nemo_model_path

                        elif arg_name == "nemo_devices":
                            cfg.trainer.devices = self.args.nemo_devices
                        else:
                            cfg.inference[arg_name] = getattr(self.args, arg_name)
                                
                self.nemo_cfg = cfg
                self.pipe, self.nemo_trainer = nemo_init_model(cfg)
            else:
                raise NotImplementedError("The model is not implemented!")

    def process_input(self, prompt, raw_data):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
            return self._process_cls_input(prompt, raw_data)
        elif self.args.dataset == "mmlu":
            return self._process_qa_input(prompt, raw_data)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_input(prompt, raw_data)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_input(prompt, raw_data)
        elif self.args.dataset == 'math':
            return self._process_math_input(prompt, raw_data)
        elif self.args.dataset == 'bool_logic':
            return self._process_bool_logic_input(prompt, raw_data)
        elif self.args.dataset == 'valid_parentheses':
            return self._process_valid_parentheses_input(prompt, raw_data)
        else:
            raise NotImplementedError("The dataset is not implemented!")

    def process_pred(self, pred):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
            return self._process_cls_pred(pred)
        elif self.args.dataset == "mmlu":
            return self._process_qa_pred(pred)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_pred(pred)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_pred(pred)
        elif self.args.dataset == 'math':
            return self._process_math_pred(pred)
        elif self.args.dataset == 'bool_logic':
            return self._process_bool_logic_pred(pred)
        elif self.args.dataset == 'valid_parentheses':
            return self._process_valid_parentheses_pred(pred)
        else:
            raise NotImplementedError("The dataset is not implemented!")

    def eval(self, preds, gts):

        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "mmlu", "bool_logic", "valid_parentheses"]:
            if self.args.dataset == "mmlu":
                preds = [pred.lower() for pred in preds]
                gts = [gt.lower() for gt in gts]

            if not isinstance(preds, list):
                preds = [preds]
                gts = [gts]

            return sum(a == b for a, b in zip(preds, gts)) / len(preds)

        elif self.args.dataset == "squad_v2":

            from metrics.squad_v2.squad_v2 import SquadV2
            metric = SquadV2()

            model_output = []

            for id, pred in zip(gts, preds):

                if pred == "unanswerable":
                    no_ans_prob = 1
                    pred = ""
                else:
                    no_ans_prob = 0

                model_output.append(
                    {"id": id, "prediction_text": pred, "no_answer_probability": no_ans_prob})

            references = self.args.data.get_reference()
            score = metric.compute(
                predictions=model_output, references=references)

            return score["f1"] / 100

        elif self.args.dataset in ['iwslt', 'un_multi']:

            from metrics.bleu.bleu import Bleu
            metric = Bleu()
            results = metric.compute(predictions=preds, references=gts)

            # it need to /100 to get the proper bleu score (in alignment with other dataset, e.g., glue)
            return results['bleu'] / 100

        elif self.args.dataset == 'math':

            processed_preds = []
            processed_gts = []
            for pred, gt in zip(preds, gts):
                if pred.lower() == "yes":
                    pred = "True"
                elif pred.lower() == "no":
                    pred = "False"

                gt = str(gt).lower()
                processed_preds.append(pred.lower())
                processed_gts.append(gt.lower())

            acc = sum(a == b for a, b in zip(processed_preds,
                      processed_gts)) / len(processed_gts)

            return acc

        else:
            raise NotImplementedError(
                "Eval this dataset {self.args.dataset} is not implemented!")

    def predict(self, prompt=None, max_samples=1000):
        assert self.args.data is not None, "Please load data first!"

        if self.model in ["chatgpt", "gpt4"]:
            results = self.predict_by_openai_api(self.model, prompt)
        else:
            results = self.predict_by_local_inference(self.model, prompt, max_samples)
        return results

    def predict_batch(self, prompt: List[str], max_samples=1000):
        assert self.args.data is not None, "Please load data first!"

        if self.model in ["chatgpt", "gpt4"]:
            raise NotImplementedError("Batch inference is not implemented for openai api, use predict() instead.")
        else:
            results = self.predict_by_local_inference_batch(self.model, prompt, max_samples)
        return results
    
    def predict_by_openai_api(self, model, prompt):
        data_len = len(self.args.data)
        if data_len > 1000:
            data_len = 1000

        score = 0
        check_correctness = 100
        preds = []
        gts = []

        for idx in tqdm(range(data_len)):

            raw_data = self.args.data.get_content_by_idx(
                idx, self.args.dataset)
            input_text, gt = self.process_input(prompt, raw_data)

            raw_pred = self.call_openai_api(model, input_text)
            pred = self.process_pred(raw_pred)

            preds.append(pred)
            gts.append(gt)

            if check_correctness > 0 and self.args.verbose:
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))

                check_correctness -= 1

        score = self.eval(preds, gts)
        return score


    def predict_by_local_inference(self, model, prompt, max_samples=1000):
        data_len = len(self.args.data)

        if data_len > max_samples:
            data_len = max_samples

        score = 0
        check_correctness = 100
        preds = []
        gts = []

        for idx in tqdm(range(data_len)):
            raw_data = self.args.data.get_content_by_idx(idx, self.args.dataset)
            input_text, gt = self.process_input(prompt, raw_data)

            raw_pred = self.pred_by_generation(input_text, model)[0]
            # non-open ai models return list of preds
            if isinstance(raw_pred, list) and len(list) == 1:
                raw_pred = raw_pred[0]
            pred = self.process_pred(raw_pred)
        
            preds.append(pred)
            gts.append(gt)
            if check_correctness > 0 and self.args.verbose:
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))
                check_correctness -= 1

        score = self.eval(preds, gts)
        return score

    def predict_by_local_inference_batch(self, model: str, prompts: List[str], max_samples: int=1000):
        data_len = len(self.args.data)
        if data_len > max_samples:
            data_len = max_samples

        scores = []
        # TODO: why are we re-doing this multiple times?
        raw_data = [self.args.data.get_content_by_idx(idx, self.args.dataset) for idx in range(data_len)]

        if isinstance(prompts, str):
            prompts = [prompts]

        # dataset is preprocessed differently for every prompt, we're combining multiple versions of the dataset
        # so that we can use batching across all prompts
        all_data = []
        for prompt in prompts:
            all_data.extend([self.process_input(prompt, raw_data[idx]) for idx in range(len(raw_data))])

        total_num_samples = len(all_data)
        raw_dataset_size = len(raw_data)
        assert total_num_samples == len(prompts) * raw_dataset_size

        gts = []
        preds = []
        num_iter = math.ceil(1.0 * total_num_samples / self.args.batch_size)
        for batch_id in tqdm(range(num_iter)):
            start_idx = batch_id * self.args.batch_size
            batch = all_data[start_idx : start_idx + self.args.batch_size]
            input_texts, batch_gts = zip(*batch)
            gts.extend(batch_gts)
            
            raw_batch_preds = self.pred_by_generation(input_text=input_texts, model=model)
            preds.extend([self.process_pred(raw_pred) for raw_pred in raw_batch_preds])

        assert len(preds) == total_num_samples
        # split preds and gts into lists of lists, where each sublist is the preds/gts for a single prompt
        preds = [preds[i:i+raw_dataset_size] for i in range(0, len(preds), raw_dataset_size)]
        gts = [gts[i:i+raw_dataset_size] for i in range(0, len(gts), raw_dataset_size)]
        # calculate scores for each prompt
        scores = [self.eval(prompt_preds, prompt_gts) for prompt_preds, prompt_gts in zip(preds, gts)]
        return scores
    
    def call_openai_api(self, model, prompt):
        import openai
        from config import OPENAI_API
        openai.api_key = OPENAI_API
        if model in ['chatgpt']:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=20,
                temperature=0
            )
            result = response['choices'][0]['text']
        else:
            response = openai.ChatCompletion.create(
            model='gpt-4-0613',
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        result = response['choices'][0]['message']['content']
        return result

    def pred_by_generation(self, input_text: List[str], model: str) -> List[str]:
        """
        Generates the output by the model based on input_text and returns only the model output [no context]

        Args:
            input_text (str or List[str]): the input text
            model (str): the model name
        """
        out = 'error!'

        if model == "nemo":
            if self.args.nemo_use_server and TRT_AVAILABLE:
                preds = query_llm(url=self.args.nemo_url, model_name=self.args.nemo_model_path,prompts=input_text,
                                  max_output_token=self.args.generate_len,
                                  top_k=self.args.nemo_top_k,
                                  top_p=self.args.nemo_top_p,
                                  temperature=self.args.nemo_temperature,
                                  init_timeout=self.args.nemo_init_timeout)
                return [p[0] for p in preds]
            else:
                out = nemo_generate(model=self.pipe, prompts=input_text,trainer=self.nemo_trainer,cfg=self.nemo_cfg,batch_size=self.args.batch_size)
                preds = []
                for pred in out:
                    preds.extend(pred["sentences"])
                
                assert len(preds) == len(input_text)

                # remove context from output
                for i in range(len(input_text)):
                    preds[i] = preds[i][len(input_text[i]):]
                    preds[i] = preds[i].replace("<extra_id_1>System", "").replace("<extra_id_1>system", "").replace("<extra_id_1>", "").strip()
                    preds[i] = preds[i].split("\n")[0].strip()
                return preds
        # pad to the longest sequence in the batch and truncate all the sequences to the max model's length
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        input_ids = self.tokenizer(input_text, padding="longest", truncation=True, return_tensors="pt").input_ids.to("cuda")
      
        if 't5' in model or 'ul2' in model:
            outputs = self.pipe.generate(input_ids, max_length=self.args.generate_len, early_stopping=True)
            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elif model == 'EleutherAI/gpt-neox-20b':
            outputs = self.pipe.generate(input_ids,
                                        #  do_sample=True,
                                        temperature=0.00001,
                                        #  max_length=50,
                                        max_new_tokens=self.args.generate_len,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.eos_token_id)

            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elif model == "facebook/opt-66b":
            outputs = self.pipe.generate(input_ids)
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif model in ["llama-13b", "llama2-13b", 'llama2-13b-chat', "vicuna-13b", "vicuna-13b-v1.3", "llama2-7b", "llama2-7b-chat"]:
            outputs = self.pipe.generate(input_ids,
                                        # temperature=1.0,
                                        max_new_tokens=self.args.generate_len,
                                        early_stopping=True)

            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elif model in ['databricks/dolly-v1-6b', 'cerebras/Cerebras-GPT-13B']:
            outputs = self.pipe.generate(input_ids,
                                        temperature=0,
                                        max_new_tokens=self.args.generate_len,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        early_stopping=True)

            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        elif model == "tiiuae/falcon-40b-instruct":
            outputs = self.pipe.generate(input_ids,
                                        temperature=0,
                                        max_new_tokens=self.args.generate_len,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                        early_stopping=True)

            out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return out

    def _process_valid_parentheses_input(self, prompt, raw_data):
        question, label = raw_data['question'], raw_data['answer']
        input_text = prompt + '\n'

        if self.args.shot > 0:
            input_text += "\n" + \
                self.args.data.get_few_shot_examples(raw_data['task'])

        input_text += ("Question: " + question)
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += '\nAnswer: '
        return input_text, label

    def _process_bool_logic_input(self, prompt, raw_data):
        question, label = raw_data['question'], raw_data['answer']
        input_text = prompt + '\n'

        if self.args.shot > 0:
            input_text += "\n" + \
                self.args.data.get_few_shot_examples(raw_data['task'])

        input_text += ("Question: " + question)
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += '\nAnswer: '
        return input_text, label

    def _process_math_input(self, prompt, raw_data):
        from config import MATH_QUESTION_TYPES
        question_type, question, label = MATH_QUESTION_TYPES[raw_data['task']
                                                             ], raw_data['question'], raw_data['answer']
        input_text = prompt.format(question_type) + '\n'

        if self.args.shot > 0:
            input_text += "\n" + \
                self.args.data.get_few_shot_examples(raw_data['task'])

        input_text += ("Question: " + question)
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += '\nAnswer: '
        return input_text, label

    def _process_trans_input(self, prompt, raw_data):
        from config import LANGUAGES
        source, target, task = raw_data['source'], raw_data['target'], raw_data['task']
        src_lang, des_lang = task.split('-')
        input_text = prompt.format(
            LANGUAGES[src_lang], LANGUAGES[des_lang]) + '\n'

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(task)

        input_text += content
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += '\nAnswer: '
        return input_text, target

    def _process_squad_v2_input(self, prompt, raw_data):
        id, content = raw_data["id"], raw_data["content"]
        input_text = prompt

        if self.args.shot > 0:
            input_text += "\n" + \
                self.args.data.get_few_shot_examples(self.args.dataset)

        input_text += content
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += "Answer: "
        return input_text, id

    def _process_qa_input(self, prompt, raw_data):
        task, content = raw_data["task"], raw_data["content"]
        label = raw_data["label"]

        input_text = prompt.format(task) + "\n"

        if self.args.shot > 0:
            input_text += "\n" + \
                self.args.data.get_few_shot_examples(task.replace(" ", "_"))

        input_text += content
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += "\nAnswer: "
        return input_text, label

    def _process_cls_input(self, prompt, raw_data):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt

        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset)
            input_text += "\n"+few_shot_examples
            if self.args.dataset == "sst2" or self.args.dataset == "cola":
                input_text += "Sentence: "

        input_text += content
        # TODO fix few shot examples for NeMo prompt
        if self.args.model == "nemo":
            input_text = NEMO_PROMPT.replace("{prompt}", input_text)
        else:
            input_text += ' Answer: '
        return input_text, label

    def _process_bool_logic_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.replace("<extra_id_1>", "") # for nemo
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_valid_parentheses_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_math_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_trans_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_squad_v2_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_cls_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")

        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        pred = pred.split(" ")[-1]
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        if pred in LABEL_SET[self.args.dataset]:
            pred = LABEL_TO_ID[self.args.dataset][pred]
        else:
            self.args.logger.debug(f"The raw_pred label ({raw_pred}) -> processed_pred {pred} is not in label set: {LABEL_TO_ID[self.args.dataset]}")
            pred = -1
        return pred

    def _process_qa_pred(self, raw_pred):
        pred = raw_pred.lower()

        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")

        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        pred = pred.split(" ")[-1]
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        if pred not in LABEL_SET[self.args.dataset]:
            self.args.logger.warn(
                "The original label : '{}'.".format(raw_pred))
            self.args.logger.warn(
                "The predicted label: '{}' is not in label set.".format(pred))
            pred = 'no_answer'

        return pred
