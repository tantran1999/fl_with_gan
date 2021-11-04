from .selection_strategy import SelectionStrategy
import random

class SelectNormalWorkers(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        normal_workers = [idx for idx in workers if idx not in poisoned_workers]
        random_workers = random.sample(normal_workers, kwargs["NUM_WORKERS_PER_ROUND"])
        return random_workers