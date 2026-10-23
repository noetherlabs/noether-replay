"""Local, offline telemetry inspection. No hardware or network interfaces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .model import load_recording
from .report import build_report, markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Local AOT FITS recording or self-contained JSON fixture"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--output", type=Path, help="Create a new report file; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    try:
        if args.input.suffix.lower() in {".fits", ".fit"}:
            from .aot import read_aot

            recording = read_aot(args.input)
        elif args.input.suffix.lower() == ".json":
            recording = load_recording(args.input)
        else:
            raise ValueError("Input must be a local .fits, .fit or .json file")
        report = build_report(recording)
        output = (
            markdown_report(report)
            if args.format == "markdown"
            else json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if args.output:
            # Publish a complete file without replacing any existing evidence.
            # A same-directory hard link provides atomic no-clobber creation.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=args.output.parent,
                    prefix=".noether-replay-",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(output)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            sys.stdout.write(output)
    except (
        ValueError,
        OSError,
        ImportError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as exc:
        parser.exit(2, f"noether-replay: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
