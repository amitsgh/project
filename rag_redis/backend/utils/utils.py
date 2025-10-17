import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            if execution_time < 60:
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            else:
                minutes = execution_time / 60
                logger.info(f"{func.__name__} executed in {minutes:.2f} minutes")

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            if execution_time < 60:
                logger.error(
                    f"{func.__name__} executed in {execution_time:.2f} seconds"
                )
            else:
                minutes = execution_time / 60
                logger.error(f"{func.__name__} executed in {minutes:.2f} minutes")

    return wrapper
