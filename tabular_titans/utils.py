

import time
import logging

logger = logging.getLogger(__name__)

def time_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        logger.info(f'Starting run of "{func.__name__}"')
        result = func(*args, **kwargs)
        t2 = time.time()
        logger.info(f'Function "{func.__name__}" executed in {(t2-t1):.4f}s')
        return t2 - t1

    return wrap_func