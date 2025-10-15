import logging
import os
from colorlog import ColoredFormatter


def setup_logging():
    # Get log level from environment or default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create formatter
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    # Create handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure root logger to affect all loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(handler)
    root_logger.propagate = False

    # Set specific loggers to use the same configuration
    loggers_to_configure = [
        "backend.app",
        "backend.services.docker_service",
        "backend.services.redis_service",
        "backend.services.memory_service",
        "backend.services.llm_service",
        "backend.services.rag_service",
        "backend.services.cache_service",
        "backend.services.document_service",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level))
        logger.addHandler(handler)
        logger.propagate = False