from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, timezone

class TraceLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, step: str, message: str, stats: dict | None = None) -> None:
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": step,
            "message": message,
            "stats": stats or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
