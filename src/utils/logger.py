import logging
import os


def create_logger(log_path, log_level=logging.INFO):
    """
    Initialize the logger.
    """
    log_format = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    log_handlers = []

    if log_path:
        log_dir_path = os.path.dirname(log_path)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        file_handler = logging.FileHandler(filename=log_path, mode="a")
        log_handlers.append(file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)
