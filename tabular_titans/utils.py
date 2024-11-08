import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

n_iterations = 5


def time_func(func) -> Callable[..., list[float]]:
    def wrap_func(*args, **kwargs) -> list[float]:
        results = []
        for i in range(n_iterations):
            t1 = time.time()
            logger.info(f'Starting iteration {i} of "{func.__name__}"')
            result = func(*args, **kwargs)
            t2 = time.time()
            logger.info(f'Function "{func.__name__}" executed in {(t2-t1):.4f}s')
            results.append(t2 - t1)
        return results

    return wrap_func
