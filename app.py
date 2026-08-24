"""Command-line entry point for the application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ideal_functions.analysis import IdealFunctionSelector
from ideal_functions.datasets import (
    IdealDataSet,
    TestDataSet,
    TrainingDataSet,
)
from ideal_functions.exceptions import IdealFunctionError
from ideal_functions.repository import SQLiteRepository
from ideal_functions.visualization import create_visualization


def run(
    data_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Load the data, perform the analysis, and create the outputs."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training = TrainingDataSet(
        data_dir / "train.csv"
    ).frame

    ideal = IdealDataSet(
        data_dir / "ideal.csv"
    ).frame

    test = TestDataSet(
        data_dir / "test.csv"
    ).frame

    selector = IdealFunctionSelector(
        training,
        ideal,
    )

    selections = selector.select()
    mappings = selector.map_test_data(test)

    database_path = (
        output_dir / "ideal_functions.sqlite"
    )

    with SQLiteRepository(database_path) as repository:
        repository.write_sources(
            training,
            ideal,
            test,
        )

        repository.write_mappings(mappings)

        selection_data = pd.DataFrame(
            selection.__dict__
            for selection in selections
        )

        repository.write_selections(
            selection_data
        )

    visualization_path = (
        output_dir / "visualization.html"
    )

    create_visualization(
        training,
        ideal,
        mappings,
        selections,
        visualization_path,
    )

    print("Selected ideal functions:")

    for selection in selections:
        print(
            f"  {selection.training_column}: "
            f"{selection.ideal_column} "
            f"(SSE={selection.sum_squared_error:.6g}, "
            f"max deviation={selection.max_deviation:.6g})"
        )

    assigned_count = mappings[
        "ideal_function"
    ].notna().sum()

    print(
        f"Mapped {assigned_count} "
        f"of {len(test)} test points."
    )

    print(f"Database: {database_path}")
    print(
        f"Visualization: {visualization_path}"
    )


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