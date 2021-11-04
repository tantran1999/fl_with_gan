from .selection_strategy import SelectionStrategy
import random

class SelectSingleAttacker(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        normal_workers = [idx for idx in workers if idx not in poisoned_workers]
        random_workers = random.sample(normal_workers, kwargs["NUM_WORKERS_PER_ROUND"] - 1)
        random_workers.append(poisoned_workers[0])
        return random_workers