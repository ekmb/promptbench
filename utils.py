import hashlib

from nemo_utils.state_manager import StateManager


def serialize_args(args):
    """Serialize the argparse Namespace to a hash string."""
    args_str = "_".join([f"{key}_{value}" for key, value in vars(args).items()])
    # Create a hash of the arguments string for a shorter, fixed-length file name
    return hashlib.md5(args_str.encode()).hexdigest()


def handle_timeout_error(state_generator_func):
    """
    A decorator that handles TimeoutError exceptions by saving the state and logging a message.
    state_generator_func (function): A function that generates the state.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            state_manager = StateManager()
            try:
                return func(*args, **kwargs)
            except TimeoutError:
                if state_manager:
                    print("Saving state...")
                    state = state_generator_func(*args, **kwargs)
                    state_manager.save_state(state)
                    state_manager.logger.info("Exiting due to time limit.")
                raise

        return wrapper

    return decorator


def generate_predict_step(obj, batch_id, gts, preds, model, all_data=None):
    return {"batch_id": batch_id, "gts": gts, "preds": preds, "all_data": all_data}


def generate_semantic_attack_state(
    obj, language, inference_model, results_dir, dataset
):
    return {"language": language}
