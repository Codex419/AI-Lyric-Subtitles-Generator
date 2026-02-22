import logging
import logging.handlers
import queue
import threading
import contextlib

# Global queue for logs
log_queue = queue.Queue()

class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def __init__(self):
        super().__init__()
        self.batch_id = "N/A"
        self.duration = "N/A"

    def filter(self, record):
        record.batch_id = self.batch_id
        record.duration = self.duration
        return True

# Singleton filter instance to be updated dynamically
context_filter = ContextFilter()

def setup_logging():
    """
    Sets up the logging configuration with QueueHandler for async logging
    and ContextFilter for dynamic context injection.
    """
    logger = logging.getLogger("CodexTranscriber")
    logger.setLevel(logging.DEBUG)

    # Formatter with context info
    formatter = logging.Formatter(
        '%(asctime)s | Batch:%(batch_id)s | Dur:%(duration)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Queue Handler (Non-blocking)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # Console Handler (Listener will write to this)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)

    # We return the listener so the main app can start/stop it
    listener = logging.handlers.QueueListener(log_queue, console_handler)
    return logger, listener

@contextlib.contextmanager
def log_context(batch_id, duration="0s"):
    """Context manager to temporarily set logging context."""
    old_batch = context_filter.batch_id
    old_duration = context_filter.duration
    context_filter.batch_id = batch_id
    context_filter.duration = duration
    try:
        yield
    finally:
        context_filter.batch_id = old_batch
        context_filter.duration = old_duration
