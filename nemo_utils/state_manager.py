import logging
import pickle


def create_logger(log_path):
    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StateManager(metaclass=SingletonMeta):
    def __init__(self, file_path=None, log_dir=None):
        if not hasattr(self, "initialized"):
            if file_path is None or log_dir is None:
                raise ValueError(
                    "file_path and log_dir must be provided for the first initialization."
                )
            self.file_path = file_path
            self.state = {}
            self.logger = create_logger(log_dir)
            self.initialized = True
        if self.initialized:
            self.logger.info(
                f"StateManager retrieved from file_path={self.file_path} and log_dir={log_dir}"
            )
        else:
            self.logger.info(
                f"StateManager initialized with file_path={self.file_path} and log_dir={log_dir}"
            )

    def save_state(self, state=None):
        if state is not None:
            self.state.update(state)
        with open(self.file_path, "wb") as f:
            pickle.dump(self.state, f)
        self.logger.info(f"State saved to {self.file_path}")

    def restore_state(self):
        try:
            with open(self.file_path, "rb") as f:
                self.state = pickle.load(f)
                return self.state
        except FileNotFoundError:
            return None

    def add_to_state(self, key, value):
        """
        Add a key-value pair to the state.
        """
        self.state[key] = value

    def update_state(self, state):
        """
        Add a dictionary of key-value pairs to the state.
        """
        self.state.update(state)
