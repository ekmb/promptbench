import time

from nemo_utils.state_manager import SingletonMeta


class RunTimeTracker(metaclass=SingletonMeta):
    def __init__(self, time_limit_sec=None):
        if not hasattr(self, "initialized"):
            if time_limit_sec is None:
                raise ValueError(
                    "time_limit_sec must be provided for the first initialization."
                )
            self.start_time = time.time()
            self.time_limit = time_limit_sec
            self.initialized = True

    def elapsed_time(self):
        return time.time() - self.start_time

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def has_sufficient_time(self, buffer_time=30):
        """
        Check if there is sufficient time left before the time limit.
        :param buffer_time: Time in seconds to be reserved for saving state (default 10 minutes)
        :return: True if there is enough time left, False otherwise
        """
        return self.elapsed_time() < self.time_limit - buffer_time
