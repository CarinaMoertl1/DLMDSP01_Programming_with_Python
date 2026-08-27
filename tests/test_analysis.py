"""Tests for analysis, validation, persistence, and visualization."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ideal_functions.analysis import IdealFunctionSelector
from ideal_functions.constants import X_DECIMALS
from ideal_functions.datasets import (
    IdealDataSet,
    TestDataSet,
    TrainingDataSet,
)
from ideal_functions.exceptions import (
    DataConsistencyError,
    DataSchemaError,
)
from ideal_functions.repository import SQLiteRepository
from ideal_functions.visualization import create_visualization


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_TEMPORARY_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "datasets"


class AnalysisTests(unittest.TestCase):
    """Test the main application behaviour with deterministic data."""

    def setUp(self) -> None:
        """Create a small set of training and ideal functions."""
        self.training = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y1": [0.1, 1.1, 2.1],
            "y2": [0.0, 2.0, 4.0],
            "y3": [1.0, 2.0, 3.0],
            "y4": [3.0, 3.0, 3.0],
        })

        self.ideal = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
        })

        for number in range(1, 51):
            self.ideal[f"y{number}"] = [
                100.0,
                100.0,
                100.0,
            ]

        self.ideal["y1"] = [0.0, 1.0, 2.0]
        self.ideal["y2"] = [0.0, 2.0, 4.0]
        self.ideal["y3"] = [1.0, 2.0, 3.0]
        self.ideal["y4"] = [3.0, 3.0, 3.0]

    def test_selects_least_squares_functions(self) -> None:
        """Select the closest ideal function for each training series."""
        selections = IdealFunctionSelector(
            self.training,
            self.ideal,
        ).select()

        self.assertEqual(
            [item.ideal_column for item in selections],
            ["y1", "y2", "y3", "y4"],
        )

    def test_mapping_obeys_sqrt_two_threshold(self) -> None:
        """Map points within the allowed deviation."""
        selector = IdealFunctionSelector(
            self.training,
            self.ideal,
        )
        selector.select()

        mapped = selector.map_test_data(
            pd.DataFrame({
                "x": [1.0, 2.0, 5.0],
                "y": [1.12, 20.0, 1.0],
            })
        )

        self.assertEqual(len(mapped), 3)
        self.assertEqual(
            mapped.iloc[0]["ideal_function"],
            "y1",
        )
        self.assertTrue(
            pd.isna(mapped.iloc[1]["ideal_function"])
        )
        self.assertTrue(
            pd.isna(mapped.iloc[2]["ideal_function"])
        )

    def test_constructor_rejects_missing_ideal_x(self) -> None:
        """Reject training data whose x values are absent from the ideal data."""
        training = pd.DataFrame({
            "x": [0.0, 1.0, 99.0],
            "y1": [0.1, 1.1, 2.1],
            "y2": [0.0, 2.0, 4.0],
            "y3": [1.0, 2.0, 3.0],
            "y4": [3.0, 3.0, 3.0],
        })

        with self.assertRaises(DataConsistencyError):
            IdealFunctionSelector(
                training,
                self.ideal,
            )

    def test_selects_distinct_ideal_functions(self) -> None:
        """Never assign the same ideal function to two training columns."""
        training = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y1": [0.01, 1.01, 2.01],
            "y2": [0.02, 1.02, 2.02],
            "y3": [100.0, 100.0, 100.0],
            "y4": [200.0, 200.0, 200.0],
        })

        ideal = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
        })

        for number in range(1, 51):
            ideal[f"y{number}"] = [
                1000.0,
                1000.0,
                1000.0,
            ]

        # Both y1 and y2 are individually closest to ideal y1.
        # If each column were matched independently, both would be assigned
        # to y1, leaving ideal y2 without a training column.
        ideal["y1"] = [0.0, 1.0, 2.0]
        ideal["y2"] = [0.0, 1.05, 2.0]
        ideal["y3"] = [100.0, 100.0, 100.0]
        ideal["y4"] = [200.0, 200.0, 200.0]

        selections = IdealFunctionSelector(
            training,
            ideal,
        ).select()

        ideal_columns = [
            item.ideal_column
            for item in selections
        ]

        self.assertEqual(
            len(ideal_columns),
            len(set(ideal_columns)),
        )

        self.assertEqual(
            ideal_columns,
            ["y1", "y2", "y3", "y4"],
        )

    def test_mapping_requires_selection(self) -> None:
        """Reject mapping when no functions have been selected."""
        selector = IdealFunctionSelector(
            self.training,
            self.ideal,
        )

        with self.assertRaises(DataConsistencyError):
            selector.map_test_data(
                pd.DataFrame({
                    "x": [],
                    "y": [],
                })
            )

    def test_mapping_accepts_value_at_threshold(self) -> None:
        """Accept a test point within the allowed deviation limit."""
        selector = IdealFunctionSelector(
            self.training,
            self.ideal,
        )
        selector.select()

        max_deviation = selector.selections[0].max_deviation
        threshold = np.sqrt(2) * max_deviation

        # Stay just inside the boundary to avoid floating-point equality issues.
        y = 1.0 + threshold - 1e-12

        mapped = selector.map_test_data(
            pd.DataFrame({
                "x": [1.0],
                "y": [y],
            })
        )

        self.assertEqual(
            mapped.iloc[0]["ideal_function"],
            "y1",
        )

    def test_mapping_chooses_closest_eligible_function(self) -> None:
        """Choose the closest function when several functions are eligible."""
        training = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y1": [0.1, 1.1, 2.1],
            "y2": [0.0, 1.2, 2.0],
            "y3": [10.0, 10.0, 10.0],
            "y4": [20.0, 20.0, 20.0],
        })

        ideal = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y1": [0.0, 1.0, 2.0],
            "y2": [0.0, 1.1, 2.0],
            "y3": [10.0, 10.0, 10.0],
            "y4": [20.0, 20.0, 20.0],
        })

        selector = IdealFunctionSelector(
            training,
            ideal,
        )
        selector.select()

        mapped = selector.map_test_data(
            pd.DataFrame({
                "x": [1.0],
                "y": [1.08],
            })
        )

        self.assertEqual(
            mapped.iloc[0]["ideal_function"],
            "y2",
        )

    def test_schema_validation_rejects_missing_column(self) -> None:
        """Reject a dataset with missing columns."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            path = Path(directory) / "bad.csv"

            pd.DataFrame({
                "x": [0],
                "y1": [1],
            }).to_csv(path, index=False)

            with self.assertRaises(DataSchemaError):
                TrainingDataSet(path)

    def test_schema_validation_rejects_non_numeric_values(self) -> None:
        """Reject non-numeric values in the dataset."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            path = Path(directory) / "non_numeric.csv"

            pd.DataFrame({
                "x": [0],
                "y1": ["not a number"],
                "y2": [1],
                "y3": [1],
                "y4": [1],
            }).to_csv(path, index=False)

            with self.assertRaises(DataSchemaError):
                TrainingDataSet(path)

    def test_schema_validation_rejects_non_finite_values(
        self,
    ) -> None:
        """Reject NaN and infinite values in the dataset."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            path = Path(directory) / "non_finite.csv"

            pd.DataFrame({
                "x": [0.0, 1.0, 2.0],
                "y1": [1.0, np.nan, 3.0],
                "y2": [1.0, 2.0, np.inf],
                "y3": [1.0, 2.0, 3.0],
                "y4": [1.0, 2.0, 3.0],
            }).to_csv(path, index=False)

            with self.assertRaises(DataSchemaError):
                TrainingDataSet(path)

    def test_schema_validation_rejects_duplicate_training_x(
        self,
    ) -> None:
        """Reject duplicate x values in training data."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            path = Path(directory) / "duplicate_x.csv"

            pd.DataFrame({
                "x": [0, 0],
                "y1": [1, 2],
                "y2": [1, 2],
                "y3": [1, 2],
                "y4": [1, 2],
            }).to_csv(path, index=False)

            with self.assertRaises(DataSchemaError):
                TrainingDataSet(path)

    def test_test_data_allows_duplicate_x(self) -> None:
        """Allow duplicate x values in test data."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            path = Path(directory) / "test.csv"

            pd.DataFrame({
                "x": [0, 0],
                "y": [1, 2],
            }).to_csv(path, index=False)

            dataset = TestDataSet(path)

            self.assertEqual(
                len(dataset.frame),
                2,
            )

    def test_repository_writes_required_mapping_columns(self) -> None:
        """Check the column names used for stored mappings."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            repo = SQLiteRepository(
                Path(directory) / "result.sqlite"
            )

            mappings = pd.DataFrame({
                "x": [1.0],
                "y": [2.0],
                "ideal_function": ["y1"],
                "deviation": [0.1],
            })

            repo.write_mappings(mappings)

            self.assertEqual(
                list(
                    repo.read_table(
                        "test_mappings"
                    ).columns
                ),
                [
                    "X",
                    "Y",
                    "Delta_Y",
                    "Ideal_Function_No",
                ],
            )

            repo.close()

    def test_repository_writes_selected_functions(self) -> None:
        """Check that selected-function evidence is persisted."""
        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            repo = SQLiteRepository(
                Path(directory) / "result.sqlite"
            )

            selections = pd.DataFrame({
                "training_column": ["y1", "y2"],
                "ideal_column": ["y13", "y24"],
                "sum_squared_error": [34.0807, 33.4518],
                "max_deviation": [0.499221, 0.499],
            })

            repo.write_selections(selections)

            stored = repo.read_table(
                "selected_functions"
            )

            self.assertEqual(
                list(stored.columns),
                [
                    "training_column",
                    "ideal_column",
                    "sum_squared_error",
                    "max_deviation",
                ],
            )

            self.assertEqual(
                stored["ideal_column"].tolist(),
                ["y13", "y24"],
            )

            repo.close()

    def test_out_of_domain_point_is_persisted_as_unassigned(
        self,
    ) -> None:
        """Keep points outside the available x-range unassigned."""
        selector = IdealFunctionSelector(
            self.training,
            self.ideal,
        )
        selector.select()

        mappings = selector.map_test_data(
            pd.DataFrame({
                "x": [99.0],
                "y": [1.0],
            })
        )

        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            repo = SQLiteRepository(
                Path(directory) / "result.sqlite"
            )

            repo.write_mappings(mappings)

            stored = repo.read_table(
                "test_mappings"
            )

            self.assertTrue(
                pd.isna(
                    stored.iloc[0]["Delta_Y"]
                )
            )

            self.assertTrue(
                pd.isna(
                    stored.iloc[0]["Ideal_Function_No"]
                )
            )

            repo.close()

    def test_visualization_writes_dashboard(self) -> None:
        """Create the visualization file with selected functions."""
        selector = IdealFunctionSelector(
            self.training,
            self.ideal,
        )

        selections = selector.select()

        mappings = selector.map_test_data(
            pd.DataFrame({
                "x": [1.0],
                "y": [1.1],
            })
        )

        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            dashboard = Path(directory) / "dashboard.html"

            create_visualization(
                self.training,
                self.ideal,
                mappings,
                selections,
                dashboard,
            )

            self.assertTrue(
                dashboard.is_file()
            )

            html = dashboard.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "Ideal Function Mapping",
                html,
            )

            for selection in selections:
                self.assertIn(
                    selection.ideal_column,
                    html,
                )

    def test_actual_dataset_visualization_contains_selected_functions(
        self,
    ) -> None:
        """Verify that the real dataset dashboard contains the selected functions."""
        training = TrainingDataSet(
            DATA_DIRECTORY / "train.csv"
        ).frame

        ideal = IdealDataSet(
            DATA_DIRECTORY / "ideal.csv"
        ).frame

        test = TestDataSet(
            DATA_DIRECTORY / "test.csv"
        ).frame

        selector = IdealFunctionSelector(
            training,
            ideal,
        )

        selections = selector.select()
        mappings = selector.map_test_data(test)

        self.assertEqual(
            [item.ideal_column for item in selections],
            ["y13", "y24", "y36", "y40"],
        )

        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            dashboard = Path(directory) / "dashboard.html"

            create_visualization(
                training,
                ideal,
                mappings,
                selections,
                dashboard,
            )

            self.assertTrue(
                dashboard.is_file()
            )

            html = dashboard.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "Ideal Function Mapping",
                html,
            )

            for function_name in (
                "y13",
                "y24",
                "y36",
                "y40",
            ):
                self.assertIn(
                    function_name,
                    html,
                )

    def test_actual_dataset_mapping_diagnostics(self) -> None:
        """Print the deviation of every real test point against each selection."""
        # Use the real project datasets, not tests/datasets.
        data_dir = DATA_DIRECTORY

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

        print("\n--- Mapping diagnostics ---")

        for selection in selections:
            threshold = (
                np.sqrt(2)
                * selection.max_deviation
            )

            print(
                f"{selection.training_column} -> "
                f"{selection.ideal_column}: "
                f"threshold={threshold:.6f}"
            )

        print("\nTest points:")

        for row in test.itertuples(index=False):
            x = round(float(row.x), X_DECIMALS)
            y = float(row.y)

            candidates = []

            if x in selector.ideal.index:
                for selection in selections:
                    ideal_y = float(
                        selector.ideal.at[
                            x,
                            selection.ideal_column,
                        ]
                    )

                    deviation = abs(
                        y - ideal_y
                    )

                    threshold = (
                        np.sqrt(2)
                        * selection.max_deviation
                    )

                    candidates.append(
                        (
                            deviation,
                            selection.ideal_column,
                            threshold,
                        )
                    )

            candidates.sort()

            if candidates:
                (
                    deviation,
                    ideal_function,
                    threshold,
                ) = candidates[0]

                status = (
                    "ASSIGNED"
                    if deviation <= threshold
                    else "UNASSIGNED"
                )

                print(
                    f"x={x:.6f}, "
                    f"y={y:.6f}, "
                    f"closest={ideal_function}, "
                    f"deviation={deviation:.6f}, "
                    f"threshold={threshold:.6f}, "
                    f"{status}"
                )

            else:
                print(
                    f"x={x:.6f}, "
                    f"y={y:.6f}, "
                    "NO IDEAL X, UNASSIGNED"
                )

    def test_application_run_creates_expected_outputs(self) -> None:
        """Run the complete application and verify its output files."""
        from app import run

        data_dir = DATA_DIRECTORY

        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            output_dir = Path(directory)

            run(
                data_dir=data_dir,
                output_dir=output_dir,
            )

            database_path = (
                output_dir
                / "ideal_functions.sqlite"
            )

            visualization_path = (
                output_dir
                / "visualization.html"
            )

            self.assertTrue(
                database_path.is_file()
            )

            self.assertTrue(
                visualization_path.is_file()
            )

            with SQLiteRepository(
                database_path
            ) as repository:
                mappings = repository.read_table(
                    "test_mappings"
                )

                selections = repository.read_table(
                    "selected_functions"
                )

            self.assertEqual(
                len(mappings),
                100,
            )

            self.assertEqual(
                mappings[
                    "Ideal_Function_No"
                ].notna().sum(),
                34,
            )

            self.assertEqual(
                len(selections),
                4,
            )

            self.assertEqual(
                selections[
                    "ideal_column"
                ].tolist(),
                [
                    "y13",
                    "y24",
                    "y36",
                    "y40",
                ],
            )

    def test_end_to_end_actual_dataset_result(self) -> None:
        """Verify the expected result for the complete provided dataset."""
        training = TrainingDataSet(
            DATA_DIRECTORY / "train.csv"
        ).frame

        ideal = IdealDataSet(
            DATA_DIRECTORY / "ideal.csv"
        ).frame

        test = TestDataSet(
            DATA_DIRECTORY / "test.csv"
        ).frame

        selector = IdealFunctionSelector(
            training,
            ideal,
        )

        selections = selector.select()

        selected_functions = [
            item.ideal_column
            for item in selections
        ]

        self.assertEqual(
            selected_functions,
            [
                "y13",
                "y24",
                "y36",
                "y40",
            ],
        )

        mappings = selector.map_test_data(
            test
        )

        self.assertEqual(
            len(mappings),
            100,
        )

        assigned = (
            mappings["ideal_function"]
            .notna()
            .sum()
        )

        unassigned = (
            mappings["ideal_function"]
            .isna()
            .sum()
        )

        self.assertEqual(
            assigned,
            34,
        )

        self.assertEqual(
            unassigned,
            66,
        )

        with TemporaryDirectory(
            dir=TEST_TEMPORARY_DIRECTORY
        ) as directory:
            repo = SQLiteRepository(
                Path(directory)
                / "result.sqlite"
            )

            repo.write_mappings(
                mappings
            )

            stored = repo.read_table(
                "test_mappings"
            )

            self.assertEqual(
                len(stored),
                100,
            )

            stored_assigned = (
                stored[
                    "Ideal_Function_No"
                ]
                .notna()
                .sum()
            )

            stored_unassigned = (
                stored[
                    "Ideal_Function_No"
                ]
                .isna()
                .sum()
            )

            self.assertEqual(
                stored_assigned,
                34,
            )

            self.assertEqual(
                stored_unassigned,
                66,
            )

            repo.close()