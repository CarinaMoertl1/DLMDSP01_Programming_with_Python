"""Bokeh figure construction for the ideal-function dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, HoverTool, Label, Legend, LegendItem, Range1d
from bokeh.plotting import figure

from ..constants import X_DECIMALS
from .design import (
    ASSIGNED_COLOR,
    BORDER,
    CARD_BACKGROUND,
    FUNCTION_COLORS,
    GRID,
    MUTED_TEXT,
    TEXT,
    THRESHOLD_COLOR,
    TRAINING_COLOR,
    UNASSIGNED_COLOR,
)

# ============================================================================
# Data helpers
# ============================================================================


def _prepare_indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for x-based lookup.

    X values are rounded before being used as lookup keys.
    This absorbs floating-point noise introduced when the
    training, ideal, and test data are parsed separately.
    """
    result = frame.copy()

    result["x"] = (
        pd.to_numeric(result["x"], errors="coerce").astype(float).round(X_DECIMALS)
    )

    result = result.dropna(subset=["x"])

    result = result.set_index("x").sort_index()

    return result


def _safe_range(values, padding: float = 0.08) -> tuple[float, float]:
    """Return a visually useful numeric range."""
    array = np.asarray(values, dtype=float)

    array = array[np.isfinite(array)]

    if len(array) == 0:
        return -1.0, 1.0

    minimum = float(np.min(array))
    maximum = float(np.max(array))

    if minimum == maximum:
        margin = max(abs(minimum) * 0.1, 1.0)
        return minimum - margin, maximum + margin

    distance = maximum - minimum
    margin = distance * padding

    return minimum - margin, maximum + margin


def _clean_axis(plot: figure) -> None:
    """Apply consistent visual styling to a Bokeh figure."""
    plot.background_fill_color = CARD_BACKGROUND
    plot.border_fill_color = CARD_BACKGROUND

    plot.outline_line_color = BORDER
    plot.outline_line_width = 1

    plot.xgrid.grid_line_color = GRID
    plot.ygrid.grid_line_color = GRID

    plot.xgrid.grid_line_alpha = 0.65
    plot.ygrid.grid_line_alpha = 0.65

    plot.axis.axis_line_color = "#cbd5e1"
    plot.axis.major_tick_line_color = "#cbd5e1"
    plot.axis.minor_tick_line_color = None

    plot.axis.major_label_text_color = MUTED_TEXT
    plot.axis.axis_label_text_color = TEXT

    plot.axis.major_label_text_font_size = "10px"
    plot.axis.axis_label_text_font_size = "11px"

    plot.title.text_color = TEXT
    plot.title.text_font_size = "14px"
    plot.title.text_font_style = "bold"

    plot.toolbar.logo = None


def _add_legend_below(plot: figure, items: list[LegendItem]) -> None:
    """
    Place the legend below the corresponding plot.

    This avoids legends covering the actual diagrams.
    """
    legend = Legend(
        items=items,
        orientation="horizontal",
        spacing=10,
        padding=8,
        margin=4,
        label_text_font_size="10px",
        label_text_color=MUTED_TEXT,
        background_fill_color=CARD_BACKGROUND,
        background_fill_alpha=1.0,
        border_line_color=None,
    )

    plot.add_layout(legend, "below")


def _selected_test_points(
    mappings: pd.DataFrame, ideal: pd.DataFrame, ideal_column: str
) -> pd.DataFrame:
    """Return test points assigned to one selected ideal function."""
    if mappings.empty:
        return pd.DataFrame(columns=["x", "y", "predicted", "deviation"])

    result = mappings.copy()

    result["x"] = (
        pd.to_numeric(result["x"], errors="coerce").astype(float).round(X_DECIMALS)
    )

    result["y"] = pd.to_numeric(result["y"], errors="coerce")

    ideal_indexed = _prepare_indexed_frame(ideal)

    result["predicted"] = result["x"].map(ideal_indexed[ideal_column])

    result["deviation"] = (result["y"] - result["predicted"]).abs()

    result = result[result["ideal_function"] == ideal_column].copy()

    return result[["x", "y", "predicted", "deviation"]].dropna(
        subset=["x", "y", "predicted"]
    )


# ============================================================================
# Individual function plot
# ============================================================================


def _create_function_plot(
    training: pd.DataFrame,
    ideal: pd.DataFrame,
    mappings: pd.DataFrame,
    selection,
    function_index: int,
) -> figure:
    """
    Create one independent plot for one selected ideal function.

    The important point here is that the ideal function is always
    read directly from its own column. No function shares plotting
    data with another function.
    """

    training_indexed = _prepare_indexed_frame(training)
    ideal_indexed = _prepare_indexed_frame(ideal)

    training_column = selection.training_column
    ideal_column = selection.ideal_column

    function_color = FUNCTION_COLORS[function_index % len(FUNCTION_COLORS)]

    # ------------------------------------------------------------------
    # Get the exact data for this function.
    # ------------------------------------------------------------------

    ideal_data = pd.DataFrame(
        {
            "x": ideal_indexed.index.to_numpy(dtype=float),
            "y": pd.to_numeric(ideal_indexed[ideal_column], errors="coerce").to_numpy(
                dtype=float
            ),
        }
    )

    ideal_data = ideal_data.dropna()

    ideal_data = ideal_data.sort_values("x")

    training_data = pd.DataFrame(
        {
            "x": training_indexed.index.to_numpy(dtype=float),
            "y": pd.to_numeric(
                training_indexed[training_column], errors="coerce"
            ).to_numpy(dtype=float),
        }
    )

    training_data = training_data.dropna()

    training_data = training_data.sort_values("x")

    # ------------------------------------------------------------------
    # Assigned test points.
    # ------------------------------------------------------------------

    assigned = _selected_test_points(mappings, ideal, ideal_column)

    # ------------------------------------------------------------------
    # Unassigned test points that are inside this function's x-domain.
    # ------------------------------------------------------------------

    unassigned = mappings[mappings["ideal_function"].isna()].copy()

    if not unassigned.empty:
        unassigned["x"] = (
            pd.to_numeric(unassigned["x"], errors="coerce")
            .astype(float)
            .round(X_DECIMALS)
        )

        unassigned["y"] = pd.to_numeric(unassigned["y"], errors="coerce")

        unassigned["predicted"] = unassigned["x"].map(ideal_indexed[ideal_column])

        unassigned = unassigned.dropna(subset=["x", "y", "predicted"])
    else:
        unassigned = pd.DataFrame(columns=["x", "y", "predicted"])

    # ------------------------------------------------------------------
    # Deviation threshold.
    # ------------------------------------------------------------------

    threshold = np.sqrt(2) * float(selection.max_deviation)

    ideal_x = ideal_data["x"].to_numpy()
    ideal_y = ideal_data["y"].to_numpy()

    upper_y = ideal_y + threshold
    lower_y = ideal_y - threshold

    # ------------------------------------------------------------------
    # Determine plot range.
    #
    # Important:
    # Huge outlier test observations are intentionally NOT used
    # to determine the main y-axis range. Otherwise a few extreme
    # unassigned points can flatten the actual function visually.
    # ------------------------------------------------------------------

    visible_values = [ideal_y, training_data["y"].to_numpy(), upper_y, lower_y]

    if not assigned.empty:
        visible_values.append(assigned["y"].to_numpy())

    combined = np.concatenate(
        [
            np.asarray(values, dtype=float)
            for values in visible_values
            if len(values) > 0
        ]
    )

    y_min, y_max = _safe_range(combined, padding=0.10)

    x_min, x_max = _safe_range(ideal_x, padding=0.03)

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------

    plot = figure(
        title=(f"{training_column}  →  {ideal_column}"),
        width=760,
        height=440,
        x_axis_label="x",
        y_axis_label="Function value",
        x_range=Range1d(x_min, x_max),
        y_range=Range1d(y_min, y_max),
        toolbar_location="above",
        sizing_mode="stretch_width",
    )

    _clean_axis(plot)

    # ------------------------------------------------------------------
    # Ideal function.
    # ------------------------------------------------------------------

    ideal_source = ColumnDataSource(ideal_data)

    ideal_renderer = plot.line(
        x="x",
        y="y",
        source=ideal_source,
        line_color=function_color,
        line_width=3,
        alpha=0.95,
    )

    # ------------------------------------------------------------------
    # Training observations.
    # ------------------------------------------------------------------

    training_source = ColumnDataSource(training_data)

    training_renderer = plot.scatter(
        x="x",
        y="y",
        source=training_source,
        size=6,
        alpha=0.55,
        color=TRAINING_COLOR,
        line_color=CARD_BACKGROUND,
        line_width=1,
    )

    # ------------------------------------------------------------------
    # Deviation boundaries.
    # ------------------------------------------------------------------

    upper_renderer = plot.line(
        ideal_x,
        upper_y,
        line_color=THRESHOLD_COLOR,
        line_width=1.5,
        line_dash="dashed",
        alpha=0.85,
    )

    lower_renderer = plot.line(
        ideal_x,
        lower_y,
        line_color=THRESHOLD_COLOR,
        line_width=1.5,
        line_dash="dashed",
        alpha=0.85,
    )

    # ------------------------------------------------------------------
    # Assigned test observations.
    # ------------------------------------------------------------------

    assigned_renderer = None

    if not assigned.empty:
        assigned_source = ColumnDataSource(assigned)

        assigned_renderer = plot.scatter(
            x="x",
            y="y",
            source=assigned_source,
            size=9,
            alpha=0.95,
            color=ASSIGNED_COLOR,
            line_color=CARD_BACKGROUND,
            line_width=1.2,
        )

        plot.add_tools(
            HoverTool(
                renderers=[assigned_renderer],
                tooltips=[
                    ("x", "@x{0.000000}"),
                    ("Actual", "@y{0.000000}"),
                    ("Predicted", "@predicted{0.000000}"),
                    ("Deviation", "@deviation{0.000000}"),
                ],
            )
        )

    # ------------------------------------------------------------------
    # Unassigned observations.
    # ------------------------------------------------------------------

    unassigned_renderer = None

    if not unassigned.empty:
        unassigned_source = ColumnDataSource(unassigned[["x", "y", "predicted"]])

        unassigned_renderer = plot.scatter(
            x="x",
            y="y",
            source=unassigned_source,
            size=8,
            alpha=0.50,
            color=UNASSIGNED_COLOR,
            line_color=CARD_BACKGROUND,
            line_width=1,
        )

        plot.add_tools(
            HoverTool(
                renderers=[unassigned_renderer],
                tooltips=[
                    ("x", "@x{0.000000}"),
                    ("Actual", "@y{0.000000}"),
                    ("Predicted", "@predicted{0.000000}"),
                ],
            )
        )

    # ------------------------------------------------------------------
    # Threshold annotation.
    # ------------------------------------------------------------------

    threshold_label = Label(
        x=x_min,
        y=y_max,
        x_units="data",
        y_units="data",
        text=(f"Allowed deviation: ±{threshold:.4f}"),
        text_color=MUTED_TEXT,
        text_font_size="10px",
        background_fill_color=CARD_BACKGROUND,
        background_fill_alpha=0.90,
        border_line_color=BORDER,
        border_line_alpha=0.8,
        padding=7,
    )

    plot.add_layout(threshold_label)

    # ------------------------------------------------------------------
    # Legend BELOW the diagram.
    # ------------------------------------------------------------------

    legend_items = [
        LegendItem(
            label=f"Selected ideal function ({ideal_column})",
            renderers=[ideal_renderer],
        ),
        LegendItem(
            label=f"Training data ({training_column})", renderers=[training_renderer]
        ),
        LegendItem(label="Allowed deviation", renderers=[upper_renderer]),
    ]

    if assigned_renderer is not None:
        legend_items.append(
            LegendItem(label="Assigned test points", renderers=[assigned_renderer])
        )

    if unassigned_renderer is not None:
        legend_items.append(
            LegendItem(label="Unassigned test points", renderers=[unassigned_renderer])
        )

    _add_legend_below(plot, legend_items)

    return plot


# ============================================================================
# Predicted vs actual
# ============================================================================


def _create_prediction_plot(
    mappings: pd.DataFrame, ideal: pd.DataFrame, selections: list
) -> figure:
    """
    Create a clear predicted-versus-actual visualization.

    Every point represents ONE assigned test observation.

    X-axis:
        value predicted by the selected ideal function

    Y-axis:
        actual observed test value

    Perfect predictions lie on the diagonal.
    """

    prediction_frames = []

    for selection in selections:
        assigned = _selected_test_points(mappings, ideal, selection.ideal_column)

        if not assigned.empty:
            prediction_frames.append(assigned[["predicted", "y", "deviation"]])

    if prediction_frames:
        data = pd.concat(prediction_frames, ignore_index=True)
    else:
        data = pd.DataFrame(columns=["predicted", "y", "deviation"])

    data = data.dropna()

    # ------------------------------------------------------------------
    # Range.
    # ------------------------------------------------------------------

    if data.empty:
        minimum = 0.0
        maximum = 1.0
    else:
        minimum = min(data["predicted"].min(), data["y"].min())

        maximum = max(data["predicted"].max(), data["y"].max())

    if minimum == maximum:
        margin = max(abs(minimum) * 0.1, 1.0)
    else:
        margin = (maximum - minimum) * 0.08

    lower = minimum - margin
    upper = maximum + margin

    plot = figure(
        title="Predicted versus actual values",
        width=760,
        height=470,
        x_axis_label="Predicted value",
        y_axis_label="Actual value",
        x_range=Range1d(lower, upper),
        y_range=Range1d(lower, upper),
        toolbar_location="above",
        sizing_mode="stretch_width",
    )

    _clean_axis(plot)

    # ------------------------------------------------------------------
    # Perfect prediction line.
    # ------------------------------------------------------------------

    perfect_line = plot.line(
        [lower, upper],
        [lower, upper],
        line_color="#475569",
        line_width=2,
        line_dash="dashed",
        alpha=0.85,
    )

    # ------------------------------------------------------------------
    # Assigned observations.
    # ------------------------------------------------------------------

    if not data.empty:
        source = ColumnDataSource(data)

        scatter = plot.scatter(
            x="predicted",
            y="y",
            source=source,
            size=9,
            alpha=0.85,
            color=ASSIGNED_COLOR,
            line_color=CARD_BACKGROUND,
            line_width=1,
        )

        plot.add_tools(
            HoverTool(
                renderers=[scatter],
                tooltips=[
                    ("Predicted", "@predicted{0.000000}"),
                    ("Actual", "@y{0.000000}"),
                    ("Absolute deviation", "@deviation{0.000000}"),
                ],
            )
        )
    else:
        scatter = None

    # ------------------------------------------------------------------
    # Legend below.
    # ------------------------------------------------------------------

    legend_items = [LegendItem(label="Perfect prediction", renderers=[perfect_line])]

    if scatter is not None:
        legend_items.append(
            LegendItem(label="Assigned test observations", renderers=[scatter])
        )

    _add_legend_below(plot, legend_items)

    # ------------------------------------------------------------------
    # Explanation.
    # ------------------------------------------------------------------

    if not data.empty:
        mean_deviation = float(data["deviation"].mean())

        label = Label(
            x=lower,
            y=upper,
            x_units="data",
            y_units="data",
            text=(f"Mean absolute deviation: " f"{mean_deviation:.4f}"),
            text_color=MUTED_TEXT,
            text_font_size="10px",
            background_fill_color=CARD_BACKGROUND,
            background_fill_alpha=0.90,
            border_line_color=BORDER,
            border_line_alpha=0.8,
            padding=7,
        )

        plot.add_layout(label)

    return plot
