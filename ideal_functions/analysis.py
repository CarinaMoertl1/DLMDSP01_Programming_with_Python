"""Least-squares ideal-function selection and test-data mapping."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .exceptions import DataConsistencyError

MAPPING_FACTOR = np.sqrt(2)

# Round x values before using them as lookup keys.
# This prevents small floating-point differences between the
# training, ideal, and test CSV data from causing lookup errors.
X_DECIMALS = 9

# Number of closest competing candidates recorded per selection.
# These candidates help explain why an ideal function was chosen.
ALTERNATIVES_RECORDED = 6

@dataclass(frozen=True)
class CandidateScore:
    """Record one candidate ideal function considered for a selection."""

    ideal_column: str
    sum_squared_error: float


@dataclass(frozen=True)
class Selection:
    """Store the result of selecting an ideal function."""

    training_column: str
    ideal_column: str
    sum_squared_error: float
    max_deviation: float

    # Store the closest competing candidates so the dashboard can
    # explain why the selected ideal function was chosen.
    alternatives: tuple[CandidateScore, ...] = ()


class IdealFunctionSelector:
    """Select ideal functions and map test observations."""

    def __init__(self, training: pd.DataFrame, ideal: pd.DataFrame) -> None:
        """Prepare training and ideal data for comparison."""
        self.training = training.set_index("x").sort_index()
        self.training.index = self.training.index.round(X_DECIMALS)

        self.ideal = ideal.set_index("x").sort_index()
        self.ideal.index = self.ideal.index.round(X_DECIMALS)

        if not self.training.index.isin(self.ideal.index).all():
            raise DataConsistencyError(
                "Every training x value must occur in ideal data"
            )

        self.selections: list[Selection] = []

    def select(self) -> list[Selection]:
        """Select distinct ideal functions with minimum total SSE.

        Each training function is compared with every ideal function.
        The Hungarian algorithm then finds the combination with the
        lowest total squared error while keeping the selected ideal
        functions distinct.
        """
        aligned_ideal = self.ideal.loc[self.training.index]

        training_columns = list(self.training.columns)
        ideal_columns = list(self.ideal.columns)

        cost_matrix = np.empty((len(training_columns), len(ideal_columns)))

        for row, training_column in enumerate(training_columns):
            residuals = aligned_ideal.sub(self.training[training_column], axis=0)

            cost_matrix[row, :] = residuals.pow(2).sum(axis=0).to_numpy()

        row_indices, column_indices = linear_sum_assignment(cost_matrix)

        selected: list[Selection] = []

        for row, column in zip(row_indices, column_indices):
            training_column = training_columns[row]
            ideal_column = ideal_columns[column]

            residuals = aligned_ideal[ideal_column].sub(self.training[training_column])

            candidate_scores = pd.Series(
                cost_matrix[row, :], index=ideal_columns
            ).sort_values()

            top_candidates = candidate_scores.head(ALTERNATIVES_RECORDED)

            # The selected function may not be among the closest
            # candidates because another training function may have
            # already claimed one of those candidates.
            if ideal_column not in top_candidates.index:
                top_candidates = pd.concat(
                    [top_candidates, candidate_scores[[ideal_column]]]
                ).sort_values()

            alternatives = tuple(
                CandidateScore(
                    ideal_column=str(candidate_column), sum_squared_error=float(score)
                )
                for candidate_column, score in top_candidates.items()
            )

            selected.append(
                Selection(
                    training_column=training_column,
                    ideal_column=ideal_column,
                    sum_squared_error=float(cost_matrix[row, column]),
                    max_deviation=float(residuals.abs().max()),
                    alternatives=alternatives,
                )
            )

        self.selections = selected

        return selected

    def _calculate_deviation(self, x: float, y: float, selection: Selection) -> float:
        """Calculate absolute deviation from a selected ideal function."""
        ideal_value = float(self.ideal.at[x, selection.ideal_column])

        return abs(y - ideal_value)

    def _find_best_match(self, x: float, y: float) -> tuple[str | None, float | None]:
        """Return the closest eligible ideal function and its deviation."""
        candidates: list[tuple[float, Selection]] = []

        for selection in self.selections:
            deviation = self._calculate_deviation(x, y, selection)

            limit = MAPPING_FACTOR * selection.max_deviation

            if deviation <= limit:
                candidates.append((deviation, selection))

        if not candidates:
            return None, None

        deviation, selection = min(candidates, key=lambda item: item[0])

        return (selection.ideal_column, deviation)

    def map_test_data(self, test: pd.DataFrame) -> pd.DataFrame:
        """Assign test observations to eligible ideal functions."""
        if not self.selections:
            raise DataConsistencyError("Call select() before mapping test data")

        rows: list[dict[str, float | str | None]] = []

        for point in test.itertuples(index=False):
            x = round(float(point.x), X_DECIMALS)
            y = float(point.y)

            if x not in self.ideal.index:
                rows.append({"x": x, "y": y, "ideal_function": None, "deviation": None})
                continue

            ideal_function, deviation = self._find_best_match(x, y)

            rows.append(
                {
                    "x": x,
                    "y": y,
                    "ideal_function": ideal_function,
                    "deviation": deviation,
                }
            )

        return pd.DataFrame(rows, columns=["x", "y", "ideal_function", "deviation"])