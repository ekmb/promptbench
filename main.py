# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

from config import *
from config import MODEL_SET, NEMO_TRT_MODELS, TIME_LIMIT
from dataload import create_dataset
from inference import Inference
from nemo_utils.run_time_tracker import RunTimeTracker
from nemo_utils.state_manager import StateManager
from prompt_attack.attack import create_attack
from prompt_attack.goal_function import create_goal_function
from prompt_attack.utils import CLASS_REGISTRY
from utils import generate_semantic_attack_state, handle_timeout_error, serialize_args


def _add_nemo_args(parser):
    """Add NeMo arguments to update inference config, see promptbench/nemo_utils/nemo_cfgs/megatron_gpt_inference.yaml"""
    group = parser.add_argument_group(title="NeMo arguments")
    group.add_argument(
        "--nemo_model_path", type=str, default=None, help="path to .nemo model file"
    )
    group.add_argument(
        "--nemo_greedy",
        action="store_true",
        help="Whether or not to use sampling ; use greedy decoding otherwise",
    )
    group.add_argument(
        "--nemo_top_k",
        type=int,
        default=0,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    group.add_argument(
        "--nemo_top_p",
        type=float,
        default=0.9,
        help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    group.add_argument(
        "--nemo_temperature", type=float, default=1.0, help="sampling temperature"
    )
    group.add_argument(
        "--nemo_add_BOS",
        action="store_true",
        help="add the bos token at the begining of the prompt",
    )
    group.add_argument(
        "--nemo_all_probs",
        action="store_true",
        help="whether return the log prob for all the tokens in vocab",
    )
    group.add_argument(
        "--nemo_repetition_penalty",
        type=float,
        default=1.2,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    group.add_argument(
        "--nemo_devices",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    group.add_argument(
        "--nemo_url",
        type=str,
        default="localhost:8000",
        help="url for server inference",
    )
    group.add_argument(
        "--nemo_init_timeout",
        type=float,
        default=600.0,
        help="timeout for server inference",
    )
    group.add_argument(
        "--nemo_use_server", action="store_true", help="enable server inference"
    )
    group.add_argument(
        "--nemo_use_prompt",
        action="store_true",
        help="use NeMo prompt for aligned models",
    )
    group.add_argument(
        "--steerlm", action="store_true", help="use steerlm prompt for aligned models"
    )
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-large",
        choices=MODEL_SET,
        help="model name. For LLAMA also specify `--model_dir`, and NeMo models: `--nemo_model_path` and `--nemo_infer_cfg`.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bool_logic",
        choices=[
            "sst2",
            "cola",
            "qqp",
            "mnli",
            "mnli_matched",
            "mnli_mismatched",
            "qnli",
            "wnli",
            "rte",
            "mrpc",
            "mmlu",
            "squad_v2",
            "un_multi",
            "iwslt",
            "math",
            "bool_logic",
            "valid_parentheses",
        ],
    )

    parser.add_argument("--query_budget", type=float, default=float("inf"))
    parser.add_argument(
        "--attack",
        type=str,
        default="deepwordbug",
        choices=[
            "textfooler",
            "textbugger",
            "bertattack",
            "deepwordbug",
            "checklist",
            "stresstest",
            "semantic",
            "no",
            "noattack",
            "clean",
            "nemo",
            "flexible_attack",
        ],
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="path to the model directory for LLAMA and NeMo models",
    )
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--generate_len", type=int, default=4)
    parser.add_argument("--prompt_selection", action="store_true")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="max number of samples to use from the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for inference"
    )
    parser.add_argument(
        "--transforms",
        nargs="*",
        type=str,
        help=f'List of transformations for the flexible attack, list of available transformations: {CLASS_REGISTRY["transformations"]}',
        default=[],
    )
    parser.add_argument(
        "--constraints",
        nargs="*",
        type=str,
        help=f'List of constraints for the flexible attack, list of available constraints: {CLASS_REGISTRY["constraints"]}',
        default=[],
    )
    parser.add_argument(
        "--search_method",
        type=str,
        help=f'Search method for the flexible attack, list of available search methods: {CLASS_REGISTRY["search_methods"]}',
        default="",
    )

    parser = _add_nemo_args(parser)
    args = parser.parse_args()
    return args


def prompt_selection(logger, inference_model, prompts, max_samples):
    """Select the top 3 prompts to attack based on the accuracy"""

    if "predict_batch" in dir(inference_model):
        acc = inference_model.predict_batch(prompts, max_samples=max_samples)
        prompt_dict = {prompt: acc[idx] for idx, prompt in enumerate(prompts)}
    else:
        logger.warning(
            "The model does not support batch inference! Running sequentially..."
        )
        prompt_dict = {}
        for prompt in prompts:
            acc = inference_model.predict(prompt)
            prompt_dict[prompt] = acc
            logger.info("{:.2f}, {}\n".format(acc * 100, prompt))
    sorted_prompts = sorted(prompt_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_prompts


@handle_timeout_error(generate_semantic_attack_state)
def semantic_attack(language, inference_model, results_dir, dataset):
    state_manager = StateManager()
    from prompts.semantic_atk_prompts import SEMANTIC_ADV_PROMPT_SET

    prompts_dict = SEMANTIC_ADV_PROMPT_SET[dataset]
    prompts = prompts_dict[language]
    if "predict_batch" in dir(inference_model):
        acc = inference_model.predict_batch(prompts)
        for idx in range(len(prompts)):
            state_manager.logger.info(
                "Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                    language, acc[idx] * 100, prompts[idx]
                )
            )
        with open(results_dir + args.save_file_name + ".txt", "a+") as f:
            f.write(
                "Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                    language, acc * 100, prompt
                )
            )  # TODO: Fix this
    else:
        state_manager.logger.warninig(
            "The model does not support batch inference! Running sequentially..."
        )
        for prompt in prompts:
            acc = inference_model.predict(prompt)
            state_manager.logger.info(
                "Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                    language, acc * 100, prompt
                )
            )
            with open(results_dir + args.save_file_name + ".txt", "a+") as f:
                f.write(
                    "Language: {}, acc: {:.2f}%, prompt: {}\n".format(
                        language, acc * 100, prompt
                    )
                )


def attack(args, inference_model, RESULTS_DIR):
    state_manager = StateManager()

    if args.attack == "semantic":
        from prompts.semantic_atk_prompts import SEMANTIC_ADV_PROMPT_SET

        prompts_dict = SEMANTIC_ADV_PROMPT_SET[args.dataset]
        latest_language = state_manager.get("language", None)
        for language in prompts_dict.keys():
            if latest_language is not None and language < latest_language:
                continue
            semantic_attack(
                language=language,
                inference_model=inference_model,
                results_dir=RESULTS_DIR,
                dataset=args.dataset,
            )

    elif args.attack in [
        "no",
        "noattack",
        "clean",
    ]:  # Not adding saving state for this because it's done in one pass
        from config import PROMPT_SET_Promptbench_advglue as prompt_raw

        prompt = prompt_raw["clean"][args.dataset][0]
        acc = inference_model.predict(prompt)
        state_manager.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
        with open(RESULTS_DIR + args.save_file_name + ".txt", "a+") as f:
            f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc * 100))
    else:
        if args.shot == 0:
            from prompts.zero_shot.role_oriented import ROLE_ORIENTED_PROMPT_SET
            from prompts.zero_shot.task_oriented import TASK_ORIENTED_PROMPT_SET

        elif args.shot == 3:
            from prompts.three_shot.role_oriented import ROLE_ORIENTED_PROMPT_SET
            from prompts.three_shot.task_oriented import TASK_ORIENTED_PROMPT_SET

        else:
            raise NotImplementedError(
                "Currently we only implemented zero-shot and three-shot!"
            )

        goal_function = create_goal_function(args, inference_model)
        attack = create_attack(args, goal_function)

        # each dataset has different predifiend prompts, the number of prompts can vary
        run_list = [
            TASK_ORIENTED_PROMPT_SET[args.dataset],
            # ROLE_ORIENTED_PROMPT_SET[args.dataset],
        ]

        for prompts in run_list:  # TODO: add saving if needed
            # select attack prompts that give the highest accuracy
            if state_manager.state.get("sorted_prompts", None) is None:
                sorted_prompts = prompt_selection(
                    state_manager.logger, inference_model, prompts, args.max_samples
                )
                state_manager.add_to_state("sorted_prompts", sorted_prompts)
                state_manager.logger.info(f"Sorted prompts: {sorted_prompts}")
                state_manager.save_state()
            else:
                sorted_prompts = state_manager.state["sorted_prompts"]

            if args.prompt_selection:
                for prompt, acc in sorted_prompts:
                    state_manager.logger.info(
                        "Prompt: {}, acc: {:.2f}%\n".format(prompt, acc * 100)
                    )
                    with open(RESULTS_DIR + args.save_file_name + ".txt", "a+") as f:
                        f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc * 100))

                continue

            if state_manager.state.get("prompt_idx", None) is None:
                prompt_idx = 0
                state_manager.add_to_state("prompt_idx", prompt_idx)
                state_manager.save_state()
            else:
                prompt_idx = state_manager.state["prompt_idx"]

            for i, (init_prompt, init_acc) in enumerate(
                sorted_prompts[:3], start=prompt_idx
            ):
                if init_acc > 0:
                    state_manager.logger.info("Init prompt: {}".format(init_prompt))
                    (
                        init_acc,
                        attacked_prompt,
                        attacked_acc,
                        dropped_acc,
                    ) = attack.attack(init_prompt)
                    state_manager.logger.info("Original prompt: {}".format(init_prompt))
                    state_manager.logger.info(
                        "Attacked prompt: {}".format(attacked_prompt.encode("utf-8"))
                    )
                    state_manager.logger.info(
                        "Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%".format(
                            init_acc * 100, attacked_acc * 100, dropped_acc * 100
                        )
                    )

                    with open(RESULTS_DIR + args.save_file_name + ".txt", "a+") as f:
                        f.write("Original prompt: {}\n".format(init_prompt))
                        f.write(
                            "Attacked prompt: {}\n".format(
                                attacked_prompt.encode("utf-8")
                            )
                        )
                        f.write(
                            "Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%\n\n".format(
                                init_acc * 100, attacked_acc * 100, dropped_acc * 100
                            )
                        )
                else:
                    with open(RESULTS_DIR + args.save_file_name + ".txt", "a+") as f:
                        f.write("Init acc is 0, skip this prompt\n")
                        f.write("Original prompt: {}\n".format(init_prompt))
                        f.write(
                            "Original acc: {:.2f}% \n\n".format(
                                init_acc * 100, init_prompt
                            )
                        )
                state_manager.add_to_state("prompt_idx", i)
                state_manager.save_state()


def main(args):
    save_dir = args.dataset

    if args.dataset == "iwslt" or args.dataset == "un_multi":
        from config import SUPPORTED_LANGUAGES

        supported_languages = SUPPORTED_LANGUAGES[args.model]

    save_dir += "/"

    LOGS_DIR = os.path.join(args.output_dir, "logs/" + save_dir)
    RESULTS_DIR = os.path.join(args.output_dir, "results/" + save_dir)

    for DIR in [LOGS_DIR, RESULTS_DIR]:
        os.makedirs(DIR, exist_ok=True)

    log_model_name = args.model.replace("/", "_")
    if args.model == "nemo":
        if args.nemo_use_server:
            if args.nemo_model_path not in NEMO_TRT_MODELS:
                raise ValueError(
                    "Please specify a valid NeMo model for server inference!"
                )
        elif args.nemo_model_path is None or not os.path.exists(args.nemo_model_path):
            raise ValueError(f"{args.nemo_model_path} not found. Please specify a valid .nemo path")
        
        log_model_name = f"{args.nemo_model_path}_server" if args.nemo_use_server else os.path.basename(args.nemo_model_path).replace(".nemo", "")
        if args.nemo_use_prompt:
            log_model_name += "_prompt"

    file_name = log_model_name + '_' + args.attack + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot"
    
    if args.attack == "flexible_attack":
        for DIR in [LOGS_DIR, RESULTS_DIR]:
            os.makedirs(f"{DIR}/{file_name}", exist_ok=True)

        file_name += "/" + "_".join(args.transforms)

    args.save_file_name = file_name

    state_manager: StateManager = StateManager(
        LOGS_DIR + f"/benchmark_state_{serialize_args(args)}.pkl",
        LOGS_DIR + file_name + ".log",
    )
    restored_state = state_manager.restore_state()
    if restored_state:
        state_manager.update_state(restored_state)
    time_tracker = RunTimeTracker(TIME_LIMIT)

    if args.dataset in ["iwslt", "un_multi"]:
        data = create_dataset(args.dataset, supported_languages)
    else:
        data = create_dataset(args.dataset)

    inference_model = Inference(args)
    args.data = data
    args.logger = state_manager.logger

    state_manager.logger.info(args)
    with open(RESULTS_DIR + args.save_file_name + ".txt", "a+") as f:
        f.write(str(args) + "\n")
    try:
        attack(args, inference_model, RESULTS_DIR)
    except TimeoutError:
        state_manager.logger.info("Exiting due to time limit.")


if __name__ == "__main__":
    args = get_args()
    main(args)
