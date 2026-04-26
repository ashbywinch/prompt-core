#!/usr/bin/env python3
"""Run a command with a hard timeout."""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command with timeout and propagate exit code"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Timeout in seconds before terminating the command",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        print("No command specified", file=sys.stderr)
        return 2

    try:
        completed = subprocess.run(command, check=False, timeout=args.timeout)
        return completed.returncode
    except subprocess.TimeoutExpired:
        print(
            f"Command timed out after {args.timeout} seconds: {' '.join(command)}",
            file=sys.stderr,
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
