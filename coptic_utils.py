import logging
import os
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("log/coptic_data_processing.log")
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def get_home_path():
    return os.path.expanduser('~')