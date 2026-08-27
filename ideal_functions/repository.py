"""SQLite persistence for source data and analysis results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .exceptions import DatabaseOperationError


class SQLiteRepository:
    """Store application data in a SQLite database."""

    MAPPING_COLUMN_NAMES = {
        "x": "X",
        "y": "Y",
        "deviation": "Delta_Y",
        "ideal_function": "Ideal_Function_No",
    }

    MAPPING_COLUMNS = ("X", "Y", "Delta_Y", "Ideal_Function_No")

    SELECTION_COLUMNS = (
        "training_column",
        "ideal_column",
        "sum_squared_error",
        "max_deviation",
    )

    SOURCE_TABLES = ("training_data", "ideal_functions", "test_data")

    def __init__(self, database_path: str | Path) -> None:
        """Create a repository connected to the given SQLite file."""
        self.database_path = Path(database_path)

        self.engine: Engine = create_engine(f"sqlite:///{self.database_path.resolve()}")

    def write_sources(
        self, training: pd.DataFrame, ideal: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """Store the three input data sets."""
        try:
            training.to_sql(
                "training_data", self.engine, if_exists="replace", index=False
            )

            ideal.to_sql(
                "ideal_functions", self.engine, if_exists="replace", index=False
            )

            test.to_sql("test_data", self.engine, if_exists="replace", index=False)

        except Exception as exc:
            raise DatabaseOperationError(
                f"Could not save source tables: {exc}"
            ) from exc

    def write_mappings(self, mappings: pd.DataFrame) -> None:
        """Store test mappings using the required database column names."""
        try:
            database_frame = mappings.rename(columns=self.MAPPING_COLUMN_NAMES).loc[
                :, self.MAPPING_COLUMNS
            ]

            database_frame.to_sql(
                "test_mappings", self.engine, if_exists="replace", index=False
            )

        except Exception as exc:
            raise DatabaseOperationError(
                f"Could not save test mappings: {exc}"
            ) from exc

    def write_selections(self, selections: pd.DataFrame) -> None:
        """Store the evidence for the selected ideal functions."""
        try:
            selection_frame = selections.loc[:, self.SELECTION_COLUMNS]

            selection_frame.to_sql(
                "selected_functions", self.engine, if_exists="replace", index=False
            )

        except Exception as exc:
            raise DatabaseOperationError(
                f"Could not save selected functions: {exc}"
            ) from exc

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read a table from the database."""
        try:
            return pd.read_sql_table(table_name, self.engine)

        except Exception as exc:
            raise DatabaseOperationError(
                f"Could not read table '{table_name}': {exc}"
            ) from exc

    def __enter__(self) -> "SQLiteRepository":
        """Return the repository for use in a with-statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the database connection when leaving the context."""
        self.close()

    def close(self) -> None:
        """Release the database connections."""
        self.engine.dispose()