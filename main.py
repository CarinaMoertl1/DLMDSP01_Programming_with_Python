"""Command-line entry point for the application."""

from __future__ import annotations

import argparse
import sys

from app import run
from ideal_functions.exceptions import IdealFunctionError


def main() -> None:
    """Parse command-line arguments and start the application."""
    parser = argparse.ArgumentParser(
        description="Select and map ideal functions."
    )

    parser.add_argument(
        "--data-dir",
        default="datasets",
        help="Folder containing the CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder for generated output.",
    )

    arguments = parser.parse_args()

    try:
        run(
            arguments.data_dir,
            arguments.output_dir,
        )
    except IdealFunctionError as exc:
        print(
            f"Input or processing error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except OSError as exc:
        print(
            f"File-system error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except Exception as exc:
        print(
            f"Unexpected error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()