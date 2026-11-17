"""Write flushed, chronological CLI progress and human-readable sections to stderr."""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

logger = logging.getLogger("controller.progress")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class ProgressSection:
    """Group progress output in one atomic record.

    Parameters
    ----------
    title : str
        Record heading.
    details : tuple of str
        Record details.
    rationale : str or None, optional
        Rationale for the record.
    """

    title: str
    details: tuple[str, ...]
    rationale: str | None = None

    def __str__(self) -> str:
        lines = [_sanitize(self.title)]
        lines.extend(f"  {_sanitize(detail)}" for detail in self.details)
        if self.rationale is not None:
            lines.append(f"  Rationale: {_sanitize(self.rationale)}")
        return "\n".join(lines)


@contextmanager
def terminal_progress() -> Iterator[None]:
    """Enable elapsed-time progress on stderr for one CLI invocation, then restore logging."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ProgressFormatter())
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


class _ProgressFormatter(logging.Formatter):
    """Timestamp progress and preserve only the line breaks added by sections."""

    def __init__(self) -> None:
        super().__init__()
        self.started = perf_counter()

    def format(self, record: logging.LogRecord) -> str:
        minutes, seconds = divmod(int(perf_counter() - self.started), 60)
        if isinstance(record.msg, ProgressSection):
            message = str(record.msg)
        else:
            message = _sanitize(record.getMessage())
        return f"[{minutes:02d}:{seconds:02d}] {message}"


def _sanitize(message: str) -> str:
    # Proposal rationales and operator paths must not inject terminal controls or log lines.
    message = "".join(character if character.isprintable() else " " for character in message)
    return " ".join(message.split())
