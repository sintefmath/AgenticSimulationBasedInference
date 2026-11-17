"""Write flushed, chronological CLI progress and human-readable sections to stderr."""

import logging
import os
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

logger = logging.getLogger("controller.progress")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class ProgressSection:
    """Keep a heading, details, and optional rationale in one atomic progress record."""

    title: str
    details: tuple[str, ...]
    rationale: str | None = None

    def __str__(self) -> str:
        parts = (self.title, *self.details)
        if self.rationale is not None:
            parts += (f"Rationale: {self.rationale}",)
        return "; ".join(_sanitize(part) for part in parts)


@contextmanager
def terminal_progress() -> Iterator[None]:
    """Enable elapsed-time progress on stderr for one CLI invocation, then restore logging."""
    try:
        width = os.get_terminal_size(sys.stderr.fileno()).columns or 88
    except OSError:
        width = 88
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ProgressFormatter(width))
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
    """Render trusted sections without letting their values inject terminal layout or controls."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.started = perf_counter()
        self.width = width

    def format(self, record: logging.LogRecord) -> str:
        minutes, seconds = divmod(int(perf_counter() - self.started), 60)
        stamp = f"[{minutes:02d}:{seconds:02d}]"
        section = record.msg
        if not isinstance(section, ProgressSection):
            return f"{stamp} {_sanitize(record.getMessage())}"

        indent = " " * (len(stamp) + 1)
        heading = self.wrap(f"=== {section.title}", indent)
        heading[0] = f"{stamp} {heading[0].lstrip()}"
        heading[-1] = (heading[-1] + " ").ljust(self.width, "=").rstrip()
        title = "\n".join(heading)
        color = 36  # Cyan.
        if record.levelno >= logging.ERROR:
            color = 31  # Red.
        elif record.levelno >= logging.WARNING:
            color = 33  # Yellow.
        title = f"\x1b[1;{color}m{title}\x1b[0m"

        lines = ["", title]
        for detail in section.details:
            lines.extend(self.wrap(detail, indent))
        if section.rationale is not None:
            lines.extend(("", f"{indent}Rationale:"))
            lines.extend(self.wrap(section.rationale, indent + "  "))
        lines.append("")
        return "\n".join(lines)

    def wrap(self, message: str, indent: str) -> list[str]:
        return [
            indent + line
            for line in textwrap.wrap(
                _sanitize(message),
                width=max(1, self.width - len(indent)),
                break_on_hyphens=False,
            )
        ]


def _sanitize(message: str) -> str:
    # Proposal rationales and operator paths must not inject terminal controls or log lines.
    message = "".join(character if character.isprintable() else " " for character in message)
    return " ".join(message.split())
