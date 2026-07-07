from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class ProgressLog:
    """Small append-only progress logger for long-running jobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_outdir(cls, outdir: str | Path, filename: str = "progress.log") -> "ProgressLog":
        return cls(Path(outdir) / "logs" / filename)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

