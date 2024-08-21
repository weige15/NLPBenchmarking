import logging
import os
 
class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.colors = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',   # Green
            'WARNING': '\033[93m', # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[41m'  # Red background
        }

    def format(self, record: logging.LogRecord) -> str:
        color = self.colors.get(record.levelname, '\033[0m')
        # Format the prefix including date-time and log level
        prefix = f"{self.formatTime(record, self.datefmt)} {record.levelname}"
        colored_prefix = f"{color}{prefix}\033[0m"
        message = super().format(record)
        find_record = message.find(record.levelname)
        left_over = message[find_record+len(record.levelname):]
        message = colored_prefix + left_over

        # Replace the entire prefix with the colored version
        return message
        
# Set all Seeds
def set_seeds(seed: int):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)



def create_logger(name: str) -> logging.Logger:
    # Check if .log folder exists if ot crea
    if not os.path.exists("logs/"):
        os.makedirs("logs/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"logs/{name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Create decorator with dates and reason saying why a function is deprecated
def deprecated(reason, date):
    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{func.__name__} has deprecated since {date} and will be removed in the future.\n"
                f"Reason: {reason}\n"
            )

        return wrapper

    return decorator

