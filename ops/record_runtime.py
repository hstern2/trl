#!/usr/bin/env python3
"""Record exact wall-clock timing for a long-running training service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def human_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def start(path: Path, started_at: datetime, *, force: bool = False) -> dict[str, Any]:
    if path.is_file() and not force:
        return read_json(path)
    payload: dict[str, Any] = {
        "completed_at_utc": None,
        "elapsed_human": None,
        "elapsed_seconds": None,
        "started_at_utc": iso_timestamp(started_at),
        "status": "running",
    }
    atomic_write_json(path, payload)
    return payload


def completion_from_journal(unit: str, started_at: datetime) -> datetime:
    result = subprocess.run(
        [
            "journalctl",
            "--user",
            "--unit",
            unit,
            "--since",
            iso_timestamp(started_at),
            "--output=json",
            "--no-pager",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed_at: datetime | None = None
    for line in result.stdout.splitlines():
        entry = json.loads(line)
        if str(entry.get("MESSAGE", "")).startswith("[done]"):
            micros = int(entry["__REALTIME_TIMESTAMP"])
            completed_at = datetime.fromtimestamp(micros / 1_000_000, UTC)
    if completed_at is None:
        raise RuntimeError(f"no [done] event found for {unit} after the recorded start")
    return completed_at


def finish(path: Path, completed_at: datetime) -> dict[str, Any]:
    payload = read_json(path)
    started_at = parse_timestamp(str(payload["started_at_utc"]))
    elapsed = (completed_at - started_at).total_seconds()
    if elapsed < 0:
        raise ValueError("completion timestamp precedes start timestamp")
    payload.update(
        {
            "completed_at_utc": iso_timestamp(completed_at),
            "elapsed_human": human_duration(elapsed),
            "elapsed_seconds": round(elapsed, 3),
            "status": "completed",
        }
    )
    atomic_write_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--output", type=Path, required=True)
    start_parser.add_argument("--started-at")
    start_parser.add_argument("--force", action="store_true")

    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--output", type=Path, required=True)
    finish_parser.add_argument("--journal-unit")

    args = parser.parse_args()
    if args.command == "start":
        started_at = parse_timestamp(args.started_at) if args.started_at else utc_now()
        payload = start(args.output, started_at, force=args.force)
    else:
        completed_at = utc_now()
        if args.journal_unit:
            started_at = parse_timestamp(str(read_json(args.output)["started_at_utc"]))
            completed_at = completion_from_journal(args.journal_unit, started_at)
        payload = finish(args.output, completed_at)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
