# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging

from config import *
from dataload import create_dataset
from inference import Inference
from prompt_attack.attack import create_attack
from prompt_attack.goal_function import create_goal_function
from config import MODEL_SET


def create_logger(log_path):

    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-large', choices=MODEL_SET,
                        help="model name. For LLAMA also specify `--model_dir`, and NeMo models: `--nemo_model_path` and `--nemo_infer_cfg`.")
    parser.add_argument('--dataset', type=str, default='bool_logic', choices=["sst2", "cola", "qqp",
                                                                              "mnli", "mnli_matched", "mnli_mismatched",
                                                                              "qnli", "wnli", "rte", "mrpc",
                                                                              "mmlu", "squad_v2", "un_multi", "iwslt", "math",
                                                                              "bool_logic", "valid_parentheses",
                                                                              ])

    parser.add_argument('--query_budget', type=float, default=float("inf"))
    parser.add_argument('--attack', type=str, default='deepwordbug', choices=[
        'textfooler',
        'textbugger',
        'bertattack',
        'deepwordbug',
        'checklist',
        'stresstest',
        'semantic',
        'no', 
        'noattack',
        'clean',
    ])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--model_dir', type=str, default=None, help="path to the model directory for LLAMA and NeMo models")
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--generate_len', type=int, default=4)
    parser.add_argument('--prompt_selection', action='store_true')
    parser.add_argument('--max_samples', type=int, default=1000, help="max number of samples to use from the dataset")
    parser.add_argument('--nemo_infer_cfg', type=str, default=None, help='path to NeMo inference config yaml')
    parser.add_argument('--nemo_model_path', type=str, default=None, help='path to .nemo model file')

    args = parser.parse_args()
    return args


def prompt_selection(logger, inference_model, prompts, max_samples=1000):
    """Select the top 3 prompts to attack based on the accuracy
    """
    prompt_dict = {}

    for prompt in prompts:
        acc = inference_model.predict(prompt, max_samples=max_samples)
        prompt_dict[prompt] = acc
        logger.info("{:.2f}, {}\n".format(acc*100, prompt))
    sorted_prompts = sorted(prompt_dict.items(),
                            key=lambda x: x[1], reverse=True)
    return sorted_prompts


def attack(args, inference_model, RESULTS_DIR):
    if args.attack == "semantic":
        from prompts.semantic_atk_prompts import SEMANTIC_ADV_PROMPT_SET

        prompts_dict = SEMANTIC_ADV_PROMPT_SET[args.dataset]

        for language in prompts_dict.keys():
            prompts = prompts_dict[language]

            for prompt in prompts:
                acc = inference_model.predict(prompt)
                args.logger.info("Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                    language, acc*100, prompt))

                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                        language, acc*100, prompt))
    elif args.attack in ['no', 'noattack', 'clean']:
        from config import PROMPT_SET_Promptbench_advglue as prompt_raw
        prompt = prompt_raw['clean'][args.dataset][0]
        acc = inference_model.predict(prompt)
        args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
        with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
            f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
    else:
        if args.shot == 0:
            from prompts.zero_shot.task_oriented import TASK_ORIENTED_PROMPT_SET
            from prompts.zero_shot.role_oriented import ROLE_ORIENTED_PROMPT_SET

        elif args.shot == 3:
            from prompts.three_shot.task_oriented import TASK_ORIENTED_PROMPT_SET
            from prompts.three_shot.role_oriented import ROLE_ORIENTED_PROMPT_SET

        else:
            raise NotImplementedError(
                "Currently we only implemented zero-shot and three-shot!")

        goal_function = create_goal_function(args, inference_model)
        attack = create_attack(args, goal_function)

        # each dataset has different predifiend prompts, the number of prompts can vary
        run_list = [
            TASK_ORIENTED_PROMPT_SET[args.dataset],
            ROLE_ORIENTED_PROMPT_SET[args.dataset],
        ]

        for prompts in run_list:
            # select attack prompts that give the highest accuracy
            sorted_prompts = prompt_selection(
                args.logger, inference_model, prompts, args.max_samples)
            if args.prompt_selection:
                for prompt, acc in sorted_prompts:
                    args.logger.info(
                        "Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
                    with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                        f.write("Prompt: {}, acc: {:.2f}%\n".format(
                            prompt, acc*100))

                continue

            for init_prompt, init_acc in sorted_prompts[:3]:
                if init_acc > 0:
                    print("Init prompt: {}".format(init_prompt))
                    import pdb; pdb.set_trace()
                    init_acc, attacked_prompt, attacked_acc, dropped_acc = attack.attack(init_prompt)
                    args.logger.info("Original prompt: {}".format(init_prompt))
                    args.logger.info("Attacked prompt: {}".format(attacked_prompt.encode('utf-8')))
                    args.logger.info("Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%".format(init_acc*100, attacked_acc*100, dropped_acc*100))
                    import pdb; pdb.set_trace()
                    with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                        f.write("Original prompt: {}\n".format(init_prompt))
                        f.write("Attacked prompt: {}\n".format(
                            attacked_prompt.encode('utf-8')))
                        f.write("Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%\n\n".format(
                            init_acc*100, attacked_acc*100, dropped_acc*100))
                else:
                    with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                        f.write("Init acc is 0, skip this prompt\n")
                        f.write("Original prompt: {}\n".format(init_prompt))
                        f.write("Original acc: {:.2f}% \n\n".format(
                            init_acc*100, init_prompt))


def main(args):
    save_dir = args.dataset

    if args.dataset == "iwslt" or args.dataset == "un_multi":
        from config import SUPPORTED_LANGUAGES
        supported_languages = SUPPORTED_LANGUAGES[args.model]

    save_dir += "/"

    LOGS_DIR = os.path.join(args.output_dir, "logs/" + save_dir)
    RESULTS_DIR = os.path.join(args.output_dir, "results/" + save_dir)

    for DIR in [LOGS_DIR, RESULTS_DIR]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    model_name = args.model.replace('/', '_')
    if args.model == "nemo":
        if args.nemo_model_path is None or not os.path.exists(args.nemo_model_path):
            raise ValueError("Please specify a valid .nemo path")
        model_name = os.path.basename(args.nemo_model_path).replace(".nemo", "")

    file_name = model_name + '_' + args.attack + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot"
    
    args.save_file_name = file_name

    if args.dataset in ["iwslt", "un_multi"]:
        data = create_dataset(args.dataset, supported_languages)
    else:
        data = create_dataset(args.dataset)

    inference_model = Inference(args)
    args.data = data

    logger = create_logger(LOGS_DIR+file_name+".log")
    logger.info(args)

    args.logger = logger

    attack(args, inference_model, RESULTS_DIR)


if __name__ == '__main__':
    args = get_args()
    main(args)
