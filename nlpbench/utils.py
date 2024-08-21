import logging
import os
from typing import Optional, Callable
import warnings


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



# Stole this from ragas, very nice
def deprecated(
    since: str,
    *,
    removal: Optional[str] = None,
    alternative: Optional[str] = None,
    addendum:Optional[str] = None,
    pending: bool = False,
):
    """
    Decorator to mark functions or classes as deprecated.

    Args:
        since: str
             The release at which this API became deprecated.
        removal: str, optional
            The expected removal version. Cannot be used with pending=True.
            Must be specified with pending=False.
        alternative: str, optional
            The alternative API or function to be used instead
            of the deprecated function.
        addendum: str, optional
            Additional text appended directly to the final message.
        pending: bool
            Whether the deprecation version is already scheduled or not.
            Cannot be used with removal.


    Examples
    --------

        .. code-block:: python

            @deprecated("0.1", removal="0.2", alternative="some_new_function")
            def some_old_function():
                print("This is an old function.")

    """

    def deprecate(func: Callable):
        def emit_warning(*args, **kwargs):
            if pending and removal:
                raise ValueError(
                    "A pending deprecation cannot have a scheduled removal"
                )

            message = f"The function {func.__name__} was deprecated in {since},"

            if not pending:
                if removal:
                    message += f" and will be removed in the {removal} release."
                else:
                    raise ValueError(
                        "A non-pending deprecation must have a scheduled removal."
                    )
            else:
                message += " and will be removed in a future release."

            if alternative:
                message += f" Use {alternative} instead."

            if addendum:
                message += f" {addendum}"

            warnings.warn(message, stacklevel=2, category=DeprecationWarning)
            return func(*args, **kwargs)

        return emit_warning

    return deprecate
