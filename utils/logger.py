# utils/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")
# PRODUCTION FIX: Increased from 150 to 10000 lines for better audit trail
MAX_LOG_LINES = 10000


class LineCappedFileHandler(RotatingFileHandler):
    """
    File handler that keeps only the last N lines in the log file
    and uses UTF-8 with replacement to avoid UnicodeEncodeError.
    """

    def __init__(self, filename, max_lines=MAX_LOG_LINES, **kwargs):
        self.max_lines = max_lines
        super().__init__(filename, **kwargs)

    def emit(self, record):
        super().emit(record)
        self._trim_file(record)

    def _trim_file(self, record):
        try:
            self.acquire()
            if self.stream:
                self.stream.flush()

            encoding = self.encoding or "utf-8"
            errors = getattr(self, "errors", "replace")
            with open(self.baseFilename, "r", encoding=encoding, errors=errors) as f:
                lines = f.readlines()

            if len(lines) <= self.max_lines:
                return

            trimmed = lines[-self.max_lines :]
            with open(self.baseFilename, "w", encoding=encoding, errors=errors) as f:
                f.writelines(trimmed)
        except Exception:
            # Defer to logging's standard error handling
            self.handleError(record)
        finally:
            self.release()


def get_logger(name: str = "app_logger"):
    """Returns a configured logger for the project with immediate flush."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # File Handler with UTF-8 encoding (prevents UnicodeEncodeError)
    file_handler = LineCappedFileHandler(
        LOG_FILE,
        max_lines=MAX_LOG_LINES,
        maxBytes=0,  # disable size-based rotation; we trim by lines
        backupCount=0,
        encoding="utf-8",
        errors="replace",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.flush()  # Flush on creation

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s %(funcName)s:%(lineno)d] -> %(message)s"
    )

    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
 