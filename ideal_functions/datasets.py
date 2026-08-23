"""CSV data sets with common validation behaviour."""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import DataSchemaError


class CsvDataSet(ABC):
    """Base class for CSV-backed data sets."""

    required_columns: tuple[str, ...] = ("x",)
    allow_duplicate_x = False

    def __init__(self, path: str | Path) -> None:
        """Load a CSV file and validate its contents."""
        self.path = Path(path)

        try:
            self.frame = pd.read_csv(self.path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataSchemaError(
                f"Cannot read {self.path}: {exc}"
            ) from exc

        self._validate()

    def _validate(self) -> None:
        """Validate the structure and values of the data set."""
        self._validate_columns()
        self._validate_numeric_values()
        self._validate_finite_values()
        self._validate_x_values()

    def _validate_columns(self) -> None:
        """Check that the CSV contains exactly the required columns."""
        actual = set(self.frame.columns)
        expected = set(self.required_columns)

        if actual == expected:
            return

        missing = expected - actual
        unexpected = actual - expected

        details = []

        if missing:
            details.append(
                f"missing columns: {sorted(missing)}"
            )

        if unexpected:
            details.append(
                f"unexpected columns: {sorted(unexpected)}"
            )

        raise DataSchemaError(
            f"{self.path} has an invalid schema "
            f"({'; '.join(details)})"
        )

    def _validate_numeric_values(self) -> None:
        """Ensure all data values can be interpreted as numbers."""
        try:
            self.frame = self.frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise DataSchemaError(
                f"{self.path} contains non-numeric values"
            ) from exc

    def _validate_finite_values(self) -> None:
        """Reject missing, infinite, or otherwise non-finite values."""
        if not np.isfinite(self.frame.to_numpy()).all():
            raise DataSchemaError(
                f"{self.path} contains missing or non-finite values"
            )

    def _validate_x_values(self) -> None:
        """Check whether duplicate x-values are allowed."""
        if self.allow_duplicate_x:
            return

        if self.frame["x"].duplicated().any():
            raise DataSchemaError(
                f"{self.path} contains duplicate x values"
            )


class TrainingDataSet(CsvDataSet):
    """The four measured training functions."""

    required_columns = (
        "x",
        "y1",
        "y2",
        "y3",
        "y4",
    )


class IdealDataSet(CsvDataSet):
    """The candidate ideal functions."""

    required_columns = (
        "x",
        *(
            f"y{number}"
            for number in range(1, 51)
        ),
    )


class TestDataSet(CsvDataSet):
    """Unlabelled observations to assign to ideal functions."""

    required_columns = ("x", "y")
    allow_duplicate_x = True