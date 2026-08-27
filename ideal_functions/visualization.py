"""Interactive visualization for ideal-function selection and test mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bokeh.embed import file_html
from bokeh.layouts import column, row, Spacer
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Label,
    Legend,
    LegendItem,
    Range1d,
    Span,
)
from bokeh.plotting import figure
from bokeh.resources import CDN


# ============================================================================
# Design
# ============================================================================

BACKGROUND = "#f5f7fa"
CARD_BACKGROUND = "#ffffff"
TEXT = "#1f2937"
MUTED_TEXT = "#64748b"
GRID = "#e5e7eb"
BORDER = "#dbe1e8"

TRAINING_COLOR = "#64748b"
ASSIGNED_COLOR = "#16a34a"
UNASSIGNED_COLOR = "#dc2626"
THRESHOLD_COLOR = "#94a3b8"

FUNCTION_COLORS = [
    "#2563eb",
    "#7c3aed",
    "#0891b2",
    "#ea580c",
]


# ============================================================================
# Data helpers
# ============================================================================


def _prepare_indexed_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a dataframe for x-based lookup.

    X values are rounded before being used as lookup keys.
    This absorbs floating-point noise introduced when the
    training, ideal, and test data are parsed separately.
    """
    result = frame.copy()

    result["x"] = (
        pd.to_numeric(result["x"], errors="coerce")
        .astype(float)
        .round(9)
    )

    result = result.dropna(subset=["x"])

    result = (
        result
        .set_index("x")
        .sort_index()
    )

    return result


def _safe_range(
    values,
    padding: float = 0.08,
) -> tuple[float, float]:
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


def _add_legend_below(
    plot: figure,
    items: list[LegendItem],
) -> None:
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
    mappings: pd.DataFrame,
    ideal: pd.DataFrame,
    ideal_column: str,
) -> pd.DataFrame:
    """Return test points assigned to one selected ideal function."""
    if mappings.empty:
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "predicted",
                "deviation",
            ]
        )

    result = mappings.copy()

    result["x"] = (
        pd.to_numeric(result["x"], errors="coerce")
        .astype(float)
        .round(9)
    )

    result["y"] = pd.to_numeric(
        result["y"],
        errors="coerce",
    )

    ideal_indexed = _prepare_indexed_frame(ideal)

    result["predicted"] = result["x"].map(
        ideal_indexed[ideal_column]
    )

    result["deviation"] = (
        result["y"]
        - result["predicted"]
    ).abs()

    result = result[
        result["ideal_function"] == ideal_column
    ].copy()

    return result[
        [
            "x",
            "y",
            "predicted",
            "deviation",
        ]
    ].dropna(
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

    function_color = FUNCTION_COLORS[
        function_index % len(FUNCTION_COLORS)
    ]

    # ------------------------------------------------------------------
    # Get the exact data for this function.
    # ------------------------------------------------------------------

    ideal_data = pd.DataFrame({
        "x": ideal_indexed.index.to_numpy(dtype=float),
        "y": pd.to_numeric(
            ideal_indexed[ideal_column],
            errors="coerce",
        ).to_numpy(dtype=float),
    })

    ideal_data = ideal_data.dropna()

    ideal_data = ideal_data.sort_values("x")

    training_data = pd.DataFrame({
        "x": training_indexed.index.to_numpy(dtype=float),
        "y": pd.to_numeric(
            training_indexed[training_column],
            errors="coerce",
        ).to_numpy(dtype=float),
    })

    training_data = training_data.dropna()

    training_data = training_data.sort_values("x")

    # ------------------------------------------------------------------
    # Assigned test points.
    # ------------------------------------------------------------------

    assigned = _selected_test_points(
        mappings,
        ideal,
        ideal_column,
    )

    # ------------------------------------------------------------------
    # Unassigned test points that are inside this function's x-domain.
    # ------------------------------------------------------------------

    unassigned = mappings[
        mappings["ideal_function"].isna()
    ].copy()

    if not unassigned.empty:
        unassigned["x"] = (
            pd.to_numeric(
                unassigned["x"],
                errors="coerce",
            )
            .astype(float)
            .round(9)
        )

        unassigned["y"] = pd.to_numeric(
            unassigned["y"],
            errors="coerce",
        )

        unassigned["predicted"] = unassigned[
            "x"
        ].map(
            ideal_indexed[ideal_column]
        )

        unassigned = unassigned.dropna(
            subset=[
                "x",
                "y",
                "predicted",
            ]
        )
    else:
        unassigned = pd.DataFrame(
            columns=[
                "x",
                "y",
                "predicted",
            ]
        )

    # ------------------------------------------------------------------
    # Deviation threshold.
    # ------------------------------------------------------------------

    threshold = (
        np.sqrt(2)
        * float(selection.max_deviation)
    )

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

    visible_values = [
        ideal_y,
        training_data["y"].to_numpy(),
        upper_y,
        lower_y,
    ]

    if not assigned.empty:
        visible_values.append(
            assigned["y"].to_numpy()
        )

    combined = np.concatenate(
        [
            np.asarray(values, dtype=float)
            for values in visible_values
            if len(values) > 0
        ]
    )

    y_min, y_max = _safe_range(
        combined,
        padding=0.10,
    )

    x_min, x_max = _safe_range(
        ideal_x,
        padding=0.03,
    )

    # ------------------------------------------------------------------
    # Figure.
    # ------------------------------------------------------------------

    plot = figure(
        title=(
            f"{training_column}  →  {ideal_column}"
        ),
        width=760,
        height=440,
        x_axis_label="x",
        y_axis_label="Function value",
        x_range=Range1d(
            x_min,
            x_max,
        ),
        y_range=Range1d(
            y_min,
            y_max,
        ),
        toolbar_location="above",
        sizing_mode="stretch_width",
    )

    _clean_axis(plot)

    # ------------------------------------------------------------------
    # Ideal function.
    # ------------------------------------------------------------------

    ideal_source = ColumnDataSource(
        ideal_data
    )

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

    training_source = ColumnDataSource(
        training_data
    )

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
        assigned_source = ColumnDataSource(
            assigned
        )

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
                renderers=[
                    assigned_renderer
                ],
                tooltips=[
                    (
                        "x",
                        "@x{0.000000}",
                    ),
                    (
                        "Actual",
                        "@y{0.000000}",
                    ),
                    (
                        "Predicted",
                        "@predicted{0.000000}",
                    ),
                    (
                        "Deviation",
                        "@deviation{0.000000}",
                    ),
                ],
            )
        )

    # ------------------------------------------------------------------
    # Unassigned observations.
    # ------------------------------------------------------------------

    unassigned_renderer = None

    if not unassigned.empty:
        unassigned_source = ColumnDataSource(
            unassigned[
                [
                    "x",
                    "y",
                    "predicted",
                ]
            ]
        )

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
                renderers=[
                    unassigned_renderer
                ],
                tooltips=[
                    (
                        "x",
                        "@x{0.000000}",
                    ),
                    (
                        "Actual",
                        "@y{0.000000}",
                    ),
                    (
                        "Predicted",
                        "@predicted{0.000000}",
                    ),
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
        text=(
            f"Allowed deviation: ±{threshold:.4f}"
        ),
        text_color=MUTED_TEXT,
        text_font_size="10px",
        background_fill_color=CARD_BACKGROUND,
        background_fill_alpha=0.90,
        border_line_color=BORDER,
        border_line_alpha=0.8,
        padding=7,
    )

    plot.add_layout(
        threshold_label
    )

    # ------------------------------------------------------------------
    # Legend BELOW the diagram.
    # ------------------------------------------------------------------

    legend_items = [
        LegendItem(
            label=f"Selected ideal function ({ideal_column})",
            renderers=[
                ideal_renderer
            ],
        ),
        LegendItem(
            label=f"Training data ({training_column})",
            renderers=[
                training_renderer
            ],
        ),
        LegendItem(
            label="Allowed deviation",
            renderers=[
                upper_renderer
            ],
        ),
    ]

    if assigned_renderer is not None:
        legend_items.append(
            LegendItem(
                label="Assigned test points",
                renderers=[
                    assigned_renderer
                ],
            )
        )

    if unassigned_renderer is not None:
        legend_items.append(
            LegendItem(
                label="Unassigned test points",
                renderers=[
                    unassigned_renderer
                ],
            )
        )

    _add_legend_below(
        plot,
        legend_items,
    )

    return plot


# ============================================================================
# Predicted vs actual
# ============================================================================


def _create_prediction_plot(
    mappings: pd.DataFrame,
    ideal: pd.DataFrame,
    selections: list,
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
        assigned = _selected_test_points(
            mappings,
            ideal,
            selection.ideal_column,
        )

        if not assigned.empty:
            prediction_frames.append(
                assigned[
                    [
                        "predicted",
                        "y",
                        "deviation",
                    ]
                ]
            )

    if prediction_frames:
        data = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    else:
        data = pd.DataFrame(
            columns=[
                "predicted",
                "y",
                "deviation",
            ]
        )

    data = data.dropna()

    # ------------------------------------------------------------------
    # Range.
    # ------------------------------------------------------------------

    if data.empty:
        minimum = 0.0
        maximum = 1.0
    else:
        minimum = min(
            data["predicted"].min(),
            data["y"].min(),
        )

        maximum = max(
            data["predicted"].max(),
            data["y"].max(),
        )

    if minimum == maximum:
        margin = max(
            abs(minimum) * 0.1,
            1.0,
        )
    else:
        margin = (
            maximum - minimum
        ) * 0.08

    lower = minimum - margin
    upper = maximum + margin

    plot = figure(
        title="Predicted versus actual values",
        width=760,
        height=470,
        x_axis_label="Predicted value",
        y_axis_label="Actual value",
        x_range=Range1d(
            lower,
            upper,
        ),
        y_range=Range1d(
            lower,
            upper,
        ),
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
        source = ColumnDataSource(
            data
        )

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
                renderers=[
                    scatter
                ],
                tooltips=[
                    (
                        "Predicted",
                        "@predicted{0.000000}",
                    ),
                    (
                        "Actual",
                        "@y{0.000000}",
                    ),
                    (
                        "Absolute deviation",
                        "@deviation{0.000000}",
                    ),
                ],
            )
        )
    else:
        scatter = None

    # ------------------------------------------------------------------
    # Legend below.
    # ------------------------------------------------------------------

    legend_items = [
        LegendItem(
            label="Perfect prediction",
            renderers=[
                perfect_line
            ],
        )
    ]

    if scatter is not None:
        legend_items.append(
            LegendItem(
                label="Assigned test observations",
                renderers=[
                    scatter
                ],
            )
        )

    _add_legend_below(
        plot,
        legend_items,
    )

    # ------------------------------------------------------------------
    # Explanation.
    # ------------------------------------------------------------------

    if not data.empty:
        mean_deviation = float(
            data["deviation"].mean()
        )

        label = Label(
            x=lower,
            y=upper,
            x_units="data",
            y_units="data",
            text=(
                f"Mean absolute deviation: "
                f"{mean_deviation:.4f}"
            ),
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


# ============================================================================
# Summary cards
# ============================================================================


def _create_summary_card(
    total_points: int,
    assigned_points: int,
    unassigned_points: int,
    selections: list,
) -> str:
    """Create the dashboard summary cards."""

    assignment_rate = (
        assigned_points
        / total_points
        * 100
        if total_points
        else 0.0
    )

    function_items = "".join(
        f"""
        <div class="function-item">
            <span class="function-training">
                {selection.training_column}
            </span>

            <span class="function-arrow">
                →
            </span>

            <span class="function-ideal">
                {selection.ideal_column}
            </span>
        </div>
        """
        for selection in selections
    )

    return f"""
    <section class="summary-grid">

        <div class="metric-card">
            <div class="metric-label">
                Test observations
            </div>

            <div class="metric-value">
                {total_points}
            </div>

            <div class="metric-description">
                Observations evaluated by the mapping procedure
            </div>
        </div>

        <div class="metric-card assigned-card">
            <div class="metric-label">
                Assigned
            </div>

            <div class="metric-value">
                {assigned_points}
            </div>

            <div class="metric-description">
                {assignment_rate:.1f}% of all test observations
            </div>
        </div>

        <div class="metric-card unassigned-card">
            <div class="metric-label">
                Unassigned
            </div>

            <div class="metric-value">
                {unassigned_points}
            </div>

            <div class="metric-description">
                Observations exceeding the allowed deviation
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Selected functions
            </div>

            <div class="function-list">
                {function_items}
            </div>
        </div>

    </section>
    """


# ============================================================================
# Closest competing candidates
# ============================================================================


def _create_alternatives_card(
    selection,
    accent_color: str,
) -> str:
    """Create a small table of the closest competing ideal functions.

    The selected ideal function is highlighted so the reader can see
    at a glance how close the runner-up candidates were.
    """
    rows = "".join(
        f"""
        <tr class="{
            'alt-row-selected'
            if candidate.ideal_column == selection.ideal_column
            else ''
        }">
            <td>{candidate.ideal_column}</td>
            <td>{candidate.sum_squared_error:.6g}</td>
        </tr>
        """
        for candidate in selection.alternatives
    )

    return f"""
    <div class="alt-card" style="border-top-color: {accent_color};">

        <div class="alt-card-title">
            <span class="alt-training">
                {selection.training_column}
            </span>

            <span class="alt-arrow">
                →
            </span>

            <span class="alt-ideal" style="color: {accent_color};">
                {selection.ideal_column}
            </span>
        </div>

        <table class="alt-table">
            <thead>
                <tr>
                    <th>Candidate</th>
                    <th>Sum of squared errors</th>
                </tr>
            </thead>

            <tbody>
                {rows}
            </tbody>
        </table>

    </div>
    """


def _create_alternatives_section(
    selections: list,
) -> str:
    """Create the section listing competing candidates per selection."""
    cards = "".join(
        _create_alternatives_card(
            selection,
            FUNCTION_COLORS[index % len(FUNCTION_COLORS)],
        )
        for index, selection in enumerate(selections)
    )

    return f"""
    <div class="section">

        <div class="section-header">

            <h2 class="section-title">
                Closest competing candidates
            </h2>

            <div class="section-description">
                For each selected ideal function, the table lists the
                ideal functions with the lowest sum of squared errors
                against the corresponding training function. The
                selected function is highlighted; the remaining rows
                show how close the next-best candidates were and give
                additional evidence for the selection.
            </div>

        </div>

        <div class="alt-grid">
            {cards}
        </div>

    </div>
    """


# ============================================================================
# Main visualization
# ============================================================================


def create_visualization(
    training: pd.DataFrame,
    ideal: pd.DataFrame,
    mappings: pd.DataFrame,
    selections: list,
    output_path: str | Path,
) -> None:
    """Create the complete interactive thesis visualization."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_points = len(mappings)

    assigned_points = int(
        mappings[
            "ideal_function"
        ].notna().sum()
    )

    unassigned_points = (
        total_points
        - assigned_points
    )

    # ------------------------------------------------------------------
    # Create the four independent function plots.
    # ------------------------------------------------------------------

    function_plots = []

    for index, selection in enumerate(
        selections
    ):
        function_plots.append(
            _create_function_plot(
                training,
                ideal,
                mappings,
                selection,
                index,
            )
        )

    # ------------------------------------------------------------------
    # Arrange plots manually.
    #
    # We intentionally do NOT use gridplot(..., spacing=...).
    # A manual row/column layout gives us predictable spacing and
    # avoids Bokeh-version compatibility issues.
    # ------------------------------------------------------------------

    if len(function_plots) >= 4:

        top_row = row(
            function_plots[0],
            function_plots[1],
            sizing_mode="stretch_width",
        )

        bottom_row = row(
            function_plots[2],
            function_plots[3],
            sizing_mode="stretch_width",
        )

        function_dashboard = column(
            top_row,
            Spacer(height=55),
            bottom_row,
            sizing_mode="stretch_width",
        )

    elif len(function_plots) == 3:

        top_row = row(
            function_plots[0],
            function_plots[1],
            sizing_mode="stretch_width",
        )

        bottom_row = row(
            function_plots[2],
            sizing_mode="stretch_width",
        )

        function_dashboard = column(
            top_row,
            Spacer(height=55),
            bottom_row,
            sizing_mode="stretch_width",
        )

    else:

        function_dashboard = column(
            *function_plots,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Prediction plot.
    # ------------------------------------------------------------------

    prediction_plot = _create_prediction_plot(
        mappings,
        ideal,
        selections,
    )

    # ------------------------------------------------------------------
    # Complete Bokeh dashboard.
    # ------------------------------------------------------------------

    dashboard = column(
        function_dashboard,
        Spacer(height=65),
        prediction_plot,
        sizing_mode="stretch_width",
    )

    dashboard_html = file_html(
        dashboard,
        CDN,
        "Ideal Function Analysis",
    )

    # Extract Bokeh body.
    body_start = dashboard_html.find(
        "<body>"
    )

    body_end = dashboard_html.find(
        "</body>"
    )

    if (
        body_start != -1
        and body_end != -1
    ):
        bokeh_body = dashboard_html[
            body_start + len("<body>"):
            body_end
        ]
    else:
        bokeh_body = dashboard_html

    # Extract Bokeh head content.
    head_content = ""

    if "<head>" in dashboard_html:
        head_content = (
            dashboard_html
            .split(
                "<head>",
                1,
            )[1]
            .split(
                "</head>",
                1,
            )[0]
        )

    # ------------------------------------------------------------------
    # Modern page styling.
    # ------------------------------------------------------------------

    style = """
    <style>

        :root {
            --background: #f5f7fa;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #64748b;
            --border: #e5e7eb;
            --green: #16a34a;
            --red: #dc2626;
        }

        * {
            box-sizing: border-box;
        }

        html {
            background: var(--background);
        }

        body {
            margin: 0;
            padding: 0;
            background: var(--background);
            color: var(--text);
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .page {
            width: min(
                1550px,
                calc(100% - 56px)
            );

            margin: 0 auto;

            padding:
                48px
                0
                70px;
        }

        .header {
            margin-bottom: 34px;
        }

        .eyebrow {
            margin-bottom: 10px;

            font-size: 12px;
            font-weight: 700;

            letter-spacing: 0.12em;
            text-transform: uppercase;

            color: #64748b;
        }

        h1 {
            margin: 0;

            font-size: 36px;
            line-height: 1.15;

            letter-spacing: -0.025em;

            color: var(--text);
        }

        .subtitle {
            max-width: 900px;

            margin-top: 13px;

            font-size: 15px;
            line-height: 1.65;

            color: var(--muted);
        }

        .summary-grid {
            display: grid;

            grid-template-columns:
                repeat(
                    4,
                    minmax(0, 1fr)
                );

            gap: 16px;

            margin-bottom: 38px;
        }

        .metric-card {
            min-height: 145px;

            padding: 22px;

            background: var(--card);

            border:
                1px solid
                var(--border);

            border-radius: 14px;

            box-shadow:
                0 2px 8px
                rgba(
                    15,
                    23,
                    42,
                    0.04
                );
        }

        .metric-label {
            margin-bottom: 10px;

            font-size: 12px;
            font-weight: 700;

            letter-spacing: 0.07em;
            text-transform: uppercase;

            color: #64748b;
        }

        .metric-value {
            font-size: 30px;
            font-weight: 750;
            line-height: 1;

            color: var(--text);
        }

        .metric-description {
            margin-top: 10px;

            font-size: 12px;
            line-height: 1.45;

            color: var(--muted);
        }

        .assigned-card {
            border-top:
                3px solid
                var(--green);
        }

        .unassigned-card {
            border-top:
                3px solid
                var(--red);
        }

        .function-list {
            display: flex;

            flex-direction: column;

            gap: 7px;
        }

        .function-item {
            display: flex;

            align-items: center;

            gap: 7px;

            font-size: 14px;
        }

        .function-training {
            font-weight: 700;
            color: var(--text);
        }

        .function-arrow {
            color: #94a3b8;
        }

        .function-ideal {
            font-weight: 700;
            color: #2563eb;
        }

        .alt-grid {
            display: grid;

            grid-template-columns:
                repeat(
                    2,
                    minmax(0, 1fr)
                );

            gap: 16px;
        }

        .alt-card {
            padding: 18px 20px;

            background: var(--card);

            border:
                1px solid
                var(--border);

            border-top: 3px solid #2563eb;

            border-radius: 14px;

            box-shadow:
                0 2px 8px
                rgba(
                    15,
                    23,
                    42,
                    0.04
                );
        }

        .alt-card-title {
            margin-bottom: 12px;

            font-size: 14px;
        }

        .alt-training {
            font-weight: 700;
            color: var(--text);
        }

        .alt-arrow {
            margin: 0 6px;
            color: #94a3b8;
        }

        .alt-ideal {
            font-weight: 700;
        }

        .alt-table {
            width: 100%;

            border-collapse: collapse;

            font-size: 12px;
        }

        .alt-table th {
            padding: 6px 8px;

            text-align: left;

            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;

            color: var(--muted);

            border-bottom: 1px solid var(--border);
        }

        .alt-table td {
            padding: 6px 8px;

            color: var(--text);

            border-bottom: 1px solid var(--border);
        }

        .alt-row-selected td {
            font-weight: 700;
            color: var(--green);
            background: #f0fdf4;
        }

        .section {
            margin-bottom: 38px;
        }

        .section-header {
            margin-bottom: 16px;
        }

        .section-title {
            margin: 0;

            font-size: 21px;
            font-weight: 750;

            letter-spacing: -0.01em;
        }

        .section-description {
            max-width: 950px;

            margin-top: 7px;

            font-size: 13px;
            line-height: 1.6;

            color: var(--muted);
        }

        .plot-card {
            padding: 22px;

            background: var(--card);

            border:
                1px solid
                var(--border);

            border-radius: 16px;

            box-shadow:
                0 2px 10px
                rgba(
                    15,
                    23,
                    42,
                    0.04
                );
        }

        .methodology {
            margin-top: 40px;

            padding:
                25px
                28px;

            background: #eef2ff;

            border:
                1px solid
                #dbeafe;

            border-radius: 14px;
        }

        .methodology h2 {
            margin:
                0
                0
                9px;

            font-size: 17px;
        }

        .methodology p {
            max-width: 1050px;

            margin: 0;

            font-size: 13px;
            line-height: 1.75;

            color: #475569;
        }

        .footer {
            margin-top: 35px;

            padding-top: 20px;

            border-top:
                1px solid
                var(--border);

            font-size: 11px;
            line-height: 1.6;

            color: #94a3b8;
        }

        @media (
            max-width: 1100px
        ) {

            .summary-grid {
                grid-template-columns:
                    repeat(
                        2,
                        minmax(0, 1fr)
                    );
            }

        }

        @media (
            max-width: 800px
        ) {

            .page {
                width:
                    calc(100% - 28px);

                padding-top: 30px;
            }

            .summary-grid {
                grid-template-columns: 1fr;
            }

            .alt-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 29px;
            }

        }

    </style>
    """

    # ------------------------------------------------------------------
    # Header.
    # ------------------------------------------------------------------

    header = """
    <div class="header">

        <div class="eyebrow">
            Numerical Analysis · Ideal Function Selection
        </div>

        <h1>
            Ideal Function Mapping Analysis
        </h1>

        <div class="subtitle">
            Comparison of the selected ideal functions with their
            corresponding training data and the observed test data.
            Test observations are classified according to the
            maximum permitted deviation defined by the selection
            procedure.
        </div>

    </div>
    """

    # ------------------------------------------------------------------
    # Function section.
    # ------------------------------------------------------------------

    function_section = """
    <div class="section">

        <div class="section-header">

            <h2 class="section-title">
                Selected ideal functions and test observations
            </h2>

            <div class="section-description">
                Each diagram represents one independent training-to-ideal
                function assignment. The blue, purple, teal, and orange
                curves therefore represent four different ideal functions.
                Training observations are shown as grey points. Green
                test observations satisfy the deviation criterion, while
                red observations remain unassigned. The dashed lines
                indicate the permitted deviation around the selected
                ideal function.
            </div>

        </div>

        <div class="plot-card">
            __FUNCTION_DASHBOARD__
        </div>

    </div>
    """

    function_section = function_section.replace(
        "__FUNCTION_DASHBOARD__",
        bokeh_body,
    )

    # ------------------------------------------------------------------
    # Closest competing candidates.
    # ------------------------------------------------------------------

    alternatives_section = _create_alternatives_section(
        selections
    )

    # ------------------------------------------------------------------
    # Prediction section.
    # ------------------------------------------------------------------

    prediction_explanation = """
    <div class="section">

        <div class="section-header">

            <h2 class="section-title">
                Predicted versus actual values
            </h2>

            <div class="section-description">
                This diagram evaluates the quality of the mapping for
                assigned test observations. Each point represents one
                assigned test observation. Its horizontal position is
                the value predicted by the selected ideal function,
                while its vertical position is the actual observed
                test value. A point on the dashed diagonal represents
                a perfect prediction. The closer a point is to the
                diagonal, the smaller its prediction error.
            </div>

        </div>

        <!-- The prediction plot is already contained in the Bokeh
             dashboard above. This explanatory section is intentionally
             kept separate for readability. -->

    </div>
    """

    # ------------------------------------------------------------------
    # Methodology.
    # ------------------------------------------------------------------

    methodology = """
    <div class="methodology">

        <h2>
            Interpretation of the mapping criterion
        </h2>

        <p>
            For each training function, the ideal function with the
            smallest sum of squared errors is selected. For a selected
            ideal function, a test observation is assigned when its
            absolute deviation from the corresponding ideal-function
            value does not exceed √2 times the maximum deviation
            observed between that ideal function and its associated
            training function. Observations exceeding this threshold
            remain unassigned.
        </p>

    </div>
    """

    # ------------------------------------------------------------------
    # Footer.
    # ------------------------------------------------------------------

    footer = """
    <div class="footer">

        Interactive visualization generated from the supplied
        training, ideal, and test datasets. Hover over observations
        to inspect their numerical values and deviations.

    </div>
    """

    # ------------------------------------------------------------------
    # Final HTML.
    # ------------------------------------------------------------------

    complete_page = f"""
    <!DOCTYPE html>

    <html lang="en">

    <head>

        <meta charset="utf-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>
            Ideal Function Analysis
        </title>

        {style}

        {head_content}

    </head>

    <body>

        <main class="page">

            {header}

            {_create_summary_card(
                total_points,
                assigned_points,
                unassigned_points,
                selections,
            )}

            {function_section}

            {alternatives_section}

            {prediction_explanation}

            {methodology}

            {footer}

        </main>

    </body>

    </html>
    """

    output_path.write_text(
        complete_page,
        encoding="utf-8",
    )