"""Modern Bokeh visualization dashboard for fitted functions and test assignments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from bokeh.layouts import column
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    Div,
    FactorRange,
    HoverTool,
    Label,
    Span,
    TabPanel,
    Tabs,
)
from bokeh.plotting import figure, output_file, save

from .analysis import Selection


# ============================================================================
# Layout
# ============================================================================

PLOT_WIDTH = 920
CURVE_HEIGHT = 410
RESIDUAL_HEIGHT = 330
OVERVIEW_HEIGHT = 460
CANDIDATE_HEIGHT = 260


# ============================================================================
# Design system
# ============================================================================

COLORS = {
    "page": "#F4F6F8",
    "surface": "#FFFFFF",

    "ink": "#172B4D",
    "muted": "#6B778C",
    "subtle": "#9AA7B5",

    "border": "#E2E8F0",
    "grid": "#EDF1F5",

    "training": "#4C78A8",
    "ideal": "#E76F51",

    "accepted": "#16A085",
    "outside": "#D95D5D",
    "rejected": "#98A2B3",

    "candidate_selected": "#16A085",
    "candidate_reserved": "#E5A62B",
    "candidate_rejected": "#D95D5D",

    "success_background": "#EAF7F3",
    "warning_background": "#FFF4E5",
    "neutral_background": "#F1F4F7",

    "success_text": "#087F67",
    "warning_text": "#946200",
    "neutral_text": "#52616B",
}


# Each selected ideal function receives one stable color.
ASSIGNMENT_COLORS = [
    "#16A085",
    "#4C78A8",
    "#8C6BB1",
    "#E5A62B",
    "#D95D5D",
    "#3A9D9A",
    "#6C8EBF",
    "#A66A9C",
]

# ============================================================================
# Shared plot styling
# ============================================================================

def _style_plot(plot) -> None:
    """Apply the common visual style to a Bokeh plot."""

    plot.background_fill_color = COLORS["surface"]
    plot.border_fill_color = COLORS["surface"]

    plot.outline_line_color = COLORS["border"]
    plot.outline_line_width = 1

    # Title
    plot.title.text_color = COLORS["ink"]
    plot.title.text_font = "Arial"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "14pt"
    plot.title.align = "left"

    # Grid
    plot.xgrid.grid_line_color = COLORS["grid"]
    plot.ygrid.grid_line_color = COLORS["grid"]

    plot.xgrid.grid_line_alpha = 0.8
    plot.ygrid.grid_line_alpha = 0.8

    # Axes
    plot.xaxis.axis_line_color = COLORS["border"]
    plot.yaxis.axis_line_color = COLORS["border"]

    plot.xaxis.major_tick_line_color = COLORS["border"]
    plot.yaxis.major_tick_line_color = COLORS["border"]

    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None

    # Axis labels
    plot.xaxis.major_label_text_color = COLORS["muted"]
    plot.yaxis.major_label_text_color = COLORS["muted"]

    plot.xaxis.axis_label_text_color = COLORS["muted"]
    plot.yaxis.axis_label_text_color = COLORS["muted"]

    plot.xaxis.major_label_text_font = "Arial"
    plot.yaxis.major_label_text_font = "Arial"

    plot.xaxis.axis_label_text_font = "Arial"
    plot.yaxis.axis_label_text_font = "Arial"

    plot.xaxis.major_label_text_font_size = "10pt"
    plot.yaxis.major_label_text_font_size = "10pt"

    plot.xaxis.axis_label_text_font_size = "11pt"
    plot.yaxis.axis_label_text_font_size = "11pt"

    # Toolbar
    plot.toolbar.logo = None
    plot.toolbar.autohide = True

# ============================================================================
# Legend components
# ============================================================================

def _legend_item(
    color: str,
    label: str,
    marker: str = "circle",
) -> str:
    """Return a small HTML legend item."""

    if marker == "line":
        symbol = f"""
            <span style="
                display:inline-block;
                width:22px;
                height:3px;
                background:{color};
                vertical-align:middle;
                margin-right:7px;
                border-radius:2px;
            "></span>
        """

    elif marker == "diamond":
        symbol = f"""
            <span style="
                display:inline-block;
                width:9px;
                height:9px;
                background:{color};
                transform:rotate(45deg);
                margin:0 9px 1px 5px;
                border-radius:2px;
            "></span>
        """

    elif marker == "cross":
        symbol = f"""
            <span style="
                display:inline-block;
                width:18px;
                margin-right:6px;
                color:{color};
                font-size:19px;
                font-weight:700;
                line-height:12px;
                text-align:center;
            ">×</span>
        """

    else:
        symbol = f"""
            <span style="
                display:inline-block;
                width:9px;
                height:9px;
                background:{color};
                border-radius:50%;
                margin-right:8px;
            "></span>
        """

    return f"""
        <span style="
            display:inline-flex;
            align-items:center;
            margin-right:20px;
            margin-bottom:5px;
            color:{COLORS["muted"]};
            font-size:12px;
        ">
            {symbol}
            {label}
        </span>
    """


def _create_plot_key(
    items: list[tuple[str, str, str]],
) -> Div:
    """Create a clean legend below a plot."""

    content = "".join(
        _legend_item(
            color=color,
            label=label,
            marker=marker,
        )
        for color, label, marker in items
    )

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            background:{COLORS["surface"]};
            border-left:1px solid {COLORS["border"]};
            border-right:1px solid {COLORS["border"]};
            border-bottom:1px solid {COLORS["border"]};
            border-radius:0 0 8px 8px;
            padding:10px 16px 7px;
            margin-top:-1px;
        ">
            {content}
        </div>
        """,
        width=PLOT_WIDTH,
    )

# ============================================================================
# Header
# ============================================================================

def _create_header(
    assigned_count: int,
    rejected_count: int,
    function_count: int,
) -> Div:
    """Create the dashboard header."""

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            padding:12px 4px 22px;
        ">

            <div style="
                margin-bottom:22px;
            ">

                <div style="
                    font-size:28px;
                    font-weight:700;
                    color:{COLORS["ink"]};
                    letter-spacing:-0.5px;
                ">
                    Ideal Function Assignment
                </div>

                <div style="
                    margin-top:7px;
                    font-size:14px;
                    color:{COLORS["muted"]};
                ">
                    Least-squares fitting and
                    <b>√2 maximum-deviation</b>
                    test-data classification
                </div>

            </div>


            <div style="
                display:flex;
                gap:12px;
                flex-wrap:wrap;
            ">

                <div style="
                    flex:1;
                    min-width:180px;
                    background:{COLORS["surface"]};
                    border:1px solid {COLORS["border"]};
                    border-radius:9px;
                    padding:15px 18px;
                ">

                    <div style="
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:0.8px;
                        text-transform:uppercase;
                        color:{COLORS["success_text"]};
                    ">
                        Assigned
                    </div>

                    <div style="
                        margin-top:4px;
                        font-size:26px;
                        font-weight:700;
                        color:{COLORS["ink"]};
                    ">
                        {assigned_count}
                    </div>

                    <div style="
                        font-size:12px;
                        color:{COLORS["muted"]};
                    ">
                        test points
                    </div>

                </div>


                <div style="
                    flex:1;
                    min-width:180px;
                    background:{COLORS["surface"]};
                    border:1px solid {COLORS["border"]};
                    border-radius:9px;
                    padding:15px 18px;
                ">

                    <div style="
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:0.8px;
                        text-transform:uppercase;
                        color:{COLORS["muted"]};
                    ">
                        Unassigned
                    </div>

                    <div style="
                        margin-top:4px;
                        font-size:26px;
                        font-weight:700;
                        color:{COLORS["ink"]};
                    ">
                        {rejected_count}
                    </div>

                    <div style="
                        font-size:12px;
                        color:{COLORS["muted"]};
                    ">
                        test points
                    </div>

                </div>


                <div style="
                    flex:1;
                    min-width:180px;
                    background:{COLORS["surface"]};
                    border:1px solid {COLORS["border"]};
                    border-radius:9px;
                    padding:15px 18px;
                ">

                    <div style="
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:0.8px;
                        text-transform:uppercase;
                        color:{COLORS["warning_text"]};
                    ">
                        Selected functions
                    </div>

                    <div style="
                        margin-top:4px;
                        font-size:26px;
                        font-weight:700;
                        color:{COLORS["ink"]};
                    ">
                        {function_count}
                    </div>

                    <div style="
                        font-size:12px;
                        color:{COLORS["muted"]};
                    ">
                        ideal functions
                    </div>

                </div>

            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )

# ============================================================================
# Explanatory components
# ============================================================================

def _create_method_note() -> Div:
    """Explain the assignment method."""

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            background:{COLORS["surface"]};
            border:1px solid {COLORS["border"]};
            border-radius:9px;
            padding:15px 18px;
            margin-bottom:16px;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                color:{COLORS["ink"]};
                text-transform:uppercase;
                letter-spacing:0.7px;
                margin-bottom:6px;
            ">
                How the assignment works
            </div>

            <div style="
                font-size:13px;
                line-height:1.6;
                color:{COLORS["muted"]};
            ">
                Each training function is compared with the available ideal
                functions using least-squares error. A test point is assigned
                when its deviation from the selected ideal function is no
                greater than <b>√2 × maximum training deviation</b>.
            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )


def _create_mapping_header(
    selection: Selection,
) -> Div:
    """Create the header for an individual mapping."""

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            padding:2px 0 14px;
        ">

            <div style="
                font-size:19px;
                font-weight:700;
                color:{COLORS["ink"]};
            ">

                {selection.training_column}

                <span style="
                    color:{COLORS["subtle"]};
                    padding:0 8px;
                ">
                    →
                </span>

                {selection.ideal_column}

            </div>

            <div style="
                margin-top:5px;
                font-size:13px;
                color:{COLORS["muted"]};
            ">
                Best-fit ideal function for this training function
            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )

def _create_fit_evidence(
    selection: Selection,
    threshold: float,
) -> Div:
    """Create compact fit statistics."""

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            background:{COLORS["surface"]};
            border:1px solid {COLORS["border"]};
            border-radius:9px;
            padding:14px 18px;
            margin-bottom:16px;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                color:{COLORS["ink"]};
                text-transform:uppercase;
                letter-spacing:0.7px;
                margin-bottom:11px;
            ">
                Fit evidence
            </div>


            <div style="
                display:flex;
                gap:10px;
                flex-wrap:wrap;
            ">

                <div style="
                    flex:1;
                    min-width:160px;
                    background:{COLORS["neutral_background"]};
                    border-radius:7px;
                    padding:10px 12px;
                ">
                    <div style="
                        font-size:11px;
                        color:{COLORS["muted"]};
                    ">
                        SSE
                    </div>

                    <div style="
                        margin-top:3px;
                        font-size:16px;
                        font-weight:700;
                        color:{COLORS["ink"]};
                    ">
                        {selection.sum_squared_error:.6g}
                    </div>
                </div>


                <div style="
                    flex:1;
                    min-width:160px;
                    background:{COLORS["neutral_background"]};
                    border-radius:7px;
                    padding:10px 12px;
                ">
                    <div style="
                        font-size:11px;
                        color:{COLORS["muted"]};
                    ">
                        Max training deviation
                    </div>

                    <div style="
                        margin-top:3px;
                        font-size:16px;
                        font-weight:700;
                        color:{COLORS["ink"]};
                    ">
                        {selection.max_deviation:.6g}
                    </div>
                </div>


                <div style="
                    flex:1;
                    min-width:160px;
                    background:{COLORS["success_background"]};
                    border-radius:7px;
                    padding:10px 12px;
                ">
                    <div style="
                        font-size:11px;
                        color:{COLORS["success_text"]};
                    ">
                        √2 acceptance limit
                    </div>

                    <div style="
                        margin-top:3px;
                        font-size:16px;
                        font-weight:700;
                        color:{COLORS["success_text"]};
                    ">
                        {threshold:.6g}
                    </div>
                </div>

            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )

# ============================================================================
# Candidate comparison ("why this function")
# ============================================================================

def _create_comparison_note() -> Div:
    """Explain the candidate comparison chart."""

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            background:{COLORS["surface"]};
            border:1px solid {COLORS["border"]};
            border-radius:9px;
            padding:14px 18px;
            margin-bottom:16px;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                color:{COLORS["ink"]};
                text-transform:uppercase;
                letter-spacing:0.7px;
                margin-bottom:6px;
            ">
                Why this function
            </div>

            <div style="
                font-size:13px;
                line-height:1.6;
                color:{COLORS["muted"]};
            ">
                Bars show the sum of squared errors between the training
                data and each of the closest candidate ideal functions, on
                a <b>log scale</b> since candidates can differ by several
                orders of magnitude. <b>Green</b> is the selected function.
                <b>Red</b> candidates simply had a higher error. <b>Amber</b>
                candidates had a lower error than the selection but were
                already reserved for a different training function, since
                each ideal function may only be used once.
            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )


def _create_comparison_chart(
    selection: Selection,
    claimed_by: dict[str, str],
) -> tuple[object, Div]:
    """Create a bar chart comparing the selection against close candidates."""

    categories = [
        candidate.ideal_column
        for candidate in selection.alternatives
    ]

    colors = []
    statuses = []

    for candidate in selection.alternatives:
        if candidate.ideal_column == selection.ideal_column:
            colors.append(COLORS["candidate_selected"])
            statuses.append("Selected")
            continue

        reserved_for = claimed_by.get(candidate.ideal_column)

        if reserved_for is not None:
            colors.append(COLORS["candidate_reserved"])
            statuses.append(
                f"Reserved for {reserved_for}"
            )
        else:
            colors.append(COLORS["candidate_rejected"])
            statuses.append("Higher error")

    source = ColumnDataSource(
        data={
            "ideal_column": categories,
            "sum_squared_error": [
                candidate.sum_squared_error
                for candidate in selection.alternatives
            ],
            "color": colors,
            "status": statuses,
        }
    )

    chart = figure(
        title="Closest candidate ideal functions",
        x_range=FactorRange(*categories),
        x_axis_label="Ideal function",
        y_axis_label="Sum of squared errors (log scale)",
        y_axis_type="log",
        width=PLOT_WIDTH,
        height=CANDIDATE_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
    )

    _style_plot(chart)

    # A logarithmic y-axis cannot display bars starting at zero.
    # Since the candidate errors can range from very small values to billions, the bars start from a small positive value instead.
    smallest_error = min(
        candidate.sum_squared_error
        for candidate in selection.alternatives
    )

    bar_floor = max(smallest_error / 10, 1e-9)

    bars = chart.vbar(
        x="ideal_column",
        top="sum_squared_error",
        bottom=bar_floor,
        width=0.6,
        source=source,
        color="color",
        line_color=None,
    )

    chart.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("Ideal function", "@ideal_column"),
                (
                    "Sum of squared errors",
                    "@sum_squared_error{0.0000}",
                ),
                ("Status", "@status"),
            ],
        )
    )

    chart.xgrid.grid_line_color = None

    key_items = [
        (
            COLORS["candidate_selected"],
            "Selected",
            "circle",
        ),
        (
            COLORS["candidate_rejected"],
            "Higher error",
            "circle",
        ),
    ]

    if any(status.startswith("Reserved") for status in statuses):
        key_items.append(
            (
                COLORS["candidate_reserved"],
                "Reserved for another training function",
                "circle",
            )
        )

    chart_key = _create_plot_key(key_items)

    return chart, chart_key


def _create_deviation_result(
    accepted_count: int,
    total_count: int,
) -> Div:
    """Create the conclusion below the deviation chart."""

    if total_count == 0:
        background = COLORS["neutral_background"]
        text_color = COLORS["neutral_text"]
        result = "No test points are assigned to this ideal function."

    elif accepted_count == total_count:
        background = COLORS["success_background"]
        text_color = COLORS["success_text"]
        result = (
            f"✓ {accepted_count} / {total_count} assigned test points "
            "are within the acceptance limit."
        )

    else:
        background = COLORS["warning_background"]
        text_color = COLORS["warning_text"]
        result = (
            f"{accepted_count} / {total_count} assigned test points "
            "are within the acceptance limit."
        )

    return Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            margin:12px 0 18px;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:0.7px;
                color:{COLORS["ink"]};
                margin-bottom:7px;
            ">
                Test-point deviation
            </div>

            <div style="
                font-size:12px;
                color:{COLORS["muted"]};
                margin-bottom:10px;
            ">
                Lower deviation means the test point lies closer to
                the selected ideal function.
                Points below the dashed line satisfy the acceptance criterion.
            </div>

            <div style="
                background:{background};
                color:{text_color};
                border-radius:7px;
                padding:10px 13px;
                font-size:13px;
                font-weight:700;
            ">
                {result}
            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )


# ============================================================================
# Main visualization
# ============================================================================

def create_visualization(
    training: pd.DataFrame,
    ideal: pd.DataFrame,
    mappings: pd.DataFrame,
    selections: list[Selection],
    output_path: str | Path,
) -> None:
    """Create and save the interactive visualization dashboard."""

    output_file(
        Path(output_path),
        title="Ideal Function Mapping",
    )

    panels = []

    # Shows which training column was assigned to each ideal function.
    # This helps explain why a candidate with a lower error may not have been selected: 
    # its ideal function may have already been assigned to another training column.
    claimed_by = {
        selection.ideal_column: selection.training_column
        for selection in selections
    }

    # ------------------------------------------------------------------------
    # Individual function tabs
    # ------------------------------------------------------------------------

    for selection_index, selection in enumerate(selections):

        assignment_color = ASSIGNMENT_COLORS[
            selection_index % len(ASSIGNMENT_COLORS)
        ]

        assigned = mappings[
            mappings["ideal_function"]
            == selection.ideal_column
        ].copy()

        threshold = (
            selection.max_deviation
            * 2**0.5
        )

        # ====================================================================
        # Main function chart
        # ====================================================================

        curve = figure(
            title="Training data and selected ideal function",
            x_axis_label="x",
            y_axis_label="y",
            width=PLOT_WIDTH,
            height=CURVE_HEIGHT,
            tools="pan,wheel_zoom,box_zoom,reset",
            toolbar_location="above",
        )

        _style_plot(curve)

        # Training data
        training_source = ColumnDataSource(
            data={
                "x": training["x"],
                "y": training[
                    selection.training_column
                ],
            }
        )

        training_renderer = curve.scatter(
            "x",
            "y",
            source=training_source,
            color=COLORS["training"],
            size=6,
            alpha=0.55,
        )

        # Ideal function
        ideal_source = ColumnDataSource(
            data={
                "x": ideal["x"],
                "y": ideal[
                    selection.ideal_column
                ],
            }
        )

        ideal_renderer = curve.line(
            "x",
            "y",
            source=ideal_source,
            color=COLORS["ideal"],
            line_width=3,
            alpha=0.95,
        )

        # Assigned test points
        if not assigned.empty:

            assigned_source = ColumnDataSource(
                assigned
            )

            test_renderer = curve.scatter(
                "x",
                "y",
                source=assigned_source,
                color=assignment_color,
                marker="diamond",
                size=10,
                alpha=0.95,
            )

            curve.add_tools(
                HoverTool(
                    renderers=[test_renderer],
                    tooltips=[
                        ("x", "@x{0.000}"),
                        ("y", "@y{0.000}"),
                        (
                            "Deviation",
                            "@deviation{0.0000}",
                        ),
                    ],
                )
            )

        curve.add_tools(
            HoverTool(
                renderers=[training_renderer],
                tooltips=[
                    ("x", "@x{0.000}"),
                    ("y", "@y{0.000}"),
                    ("Type", "Training data"),
                ],
            )
        )

        curve.add_tools(
            HoverTool(
                renderers=[ideal_renderer],
                tooltips=[
                    ("x", "$x{0.000}"),
                    ("y", "$y{0.000}"),
                    (
                        "Type",
                        "Selected ideal function",
                    ),
                ],
            )
        )

        curve_key = _create_plot_key(
            [
                (
                    COLORS["training"],
                    "Training data",
                    "circle",
                ),
                (
                    COLORS["ideal"],
                    "Selected ideal function",
                    "line",
                ),
                (
                    assignment_color,
                    "Assigned test data",
                    "diamond",
                ),
            ]
        )

        # ====================================================================
        # Deviation chart
        # ====================================================================

        residual = figure(
            title="Deviation of assigned test points",
            x_axis_label="x",
            y_axis_label="Absolute deviation",
            width=PLOT_WIDTH,
            height=RESIDUAL_HEIGHT,
            tools="pan,wheel_zoom,box_zoom,reset",
            toolbar_location="above",
        )

        _style_plot(residual)

        # Accepted region
        residual.add_layout(
            BoxAnnotation(
                bottom=0,
                top=threshold,
                fill_color=COLORS["accepted"],
                fill_alpha=0.07,
                line_alpha=0,
            )
        )

        # Acceptance threshold
        residual.add_layout(
            Span(
                location=threshold,
                dimension="width",
                line_color=COLORS["ideal"],
                line_dash="dashed",
                line_width=2,
            )
        )

        # Direct threshold label
        residual.add_layout(
            Label(
                x=0,
                y=threshold,
                x_units="screen",
                y_units="data",
                x_offset=12,
                y_offset=6,
                text=f"√2 limit = {threshold:.6g}",
                text_color=COLORS["ideal"],
                text_font="Arial",
                text_font_size="10pt",
                text_font_style="bold",
                background_fill_color=COLORS["surface"],
                background_fill_alpha=0.92,
                border_line_color=COLORS["border"],
                border_line_alpha=0.8,
                padding=5,
            )
        )

        # --------------------------------------------------------------------
        # Determine accepted/outside points
        # --------------------------------------------------------------------

        if not assigned.empty:

            assigned["deviation"] = pd.to_numeric(
                assigned["deviation"],
                errors="coerce",
            )

            accepted = assigned[
                assigned["deviation"] <= threshold
            ]

            outside_limit = assigned[
                assigned["deviation"] > threshold
            ]

            accepted_count = len(accepted)
            total_count = len(assigned)

            # Accepted points
            if not accepted.empty:

                accepted_source = ColumnDataSource(
                    accepted
                )

                accepted_renderer = residual.scatter(
                    "x",
                    "deviation",
                    source=accepted_source,
                    color=COLORS["accepted"],
                    marker="diamond",
                    size=9,
                    alpha=0.95,
                )

                residual.add_tools(
                    HoverTool(
                        renderers=[
                            accepted_renderer
                        ],
                        tooltips=[
                            ("x", "@x{0.000}"),
                            (
                                "Deviation",
                                "@deviation{0.0000}",
                            ),
                            (
                                "Status",
                                "Accepted",
                            ),
                        ],
                    )
                )

            # Outside-limit points
            if not outside_limit.empty:

                outside_source = ColumnDataSource(
                    outside_limit
                )

                outside_renderer = residual.scatter(
                    "x",
                    "deviation",
                    source=outside_source,
                    color=COLORS["outside"],
                    marker="circle",
                    size=10,
                    alpha=0.95,
                )

                residual.add_tools(
                    HoverTool(
                        renderers=[
                            outside_renderer
                        ],
                        tooltips=[
                            ("x", "@x{0.000}"),
                            (
                                "Deviation",
                                "@deviation{0.0000}",
                            ),
                            (
                                "Status",
                                "Outside limit",
                            ),
                        ],
                    )
                )

        else:
            accepted_count = 0
            total_count = 0

        residual_key = _create_plot_key(
            [
                (
                    COLORS["accepted"],
                    "Within acceptance limit",
                    "diamond",
                ),
                (
                    COLORS["outside"],
                    "Outside acceptance limit",
                    "circle",
                ),
                (
                    COLORS["ideal"],
                    "√2 acceptance limit",
                    "line",
                ),
            ]
        )

        deviation_result = _create_deviation_result(
            accepted_count=accepted_count,
            total_count=total_count,
        )

        # ====================================================================
        # Candidate comparison
        # ====================================================================

        comparison_chart, comparison_key = _create_comparison_chart(
            selection,
            claimed_by,
        )

        # ====================================================================
        # Individual tab
        # ====================================================================

        panels.append(
            TabPanel(
                title=(
                    f"{selection.training_column}"
                    f" → "
                    f"{selection.ideal_column}"
                ),
                child=column(
                    _create_mapping_header(
                        selection
                    ),

                    _create_fit_evidence(
                        selection,
                        threshold,
                    ),

                    _create_comparison_note(),
                    comparison_chart,
                    comparison_key,

                    curve,
                    curve_key,

                    deviation_result,

                    residual,
                    residual_key,

                    sizing_mode="fixed",
                ),
            )
        )

    # ------------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------------

    rejected = mappings[
        mappings["ideal_function"].isna()
    ]

    assigned_count = (
        len(mappings) - len(rejected)
    )

    overview = figure(
        title="Test-point assignments",
        x_axis_label="x",
        y_axis_label="y",
        width=PLOT_WIDTH,
        height=OVERVIEW_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
    )

    _style_plot(overview)

    overview_legend_items = []

    # ------------------------------------------------------------------------
    # Assigned test points
    # ------------------------------------------------------------------------

    for selection_index, selection in enumerate(
        selections
    ):

        assigned = mappings[
            mappings["ideal_function"]
            == selection.ideal_column
        ]

        if assigned.empty:
            continue

        assignment_color = ASSIGNMENT_COLORS[
            selection_index
            % len(ASSIGNMENT_COLORS)
        ]

        source = ColumnDataSource(
            assigned
        )

        renderer = overview.scatter(
            "x",
            "y",
            source=source,
            size=9,
            alpha=0.88,
            color=assignment_color,
            marker="diamond",
        )

        overview.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("x", "@x{0.000}"),
                    ("y", "@y{0.000}"),
                    (
                        "Assigned to",
                        selection.ideal_column,
                    ),
                ],
            )
        )

        overview_legend_items.append(
            (
                assignment_color,
                (
                    f"{selection.training_column}"
                    f" → "
                    f"{selection.ideal_column}"
                ),
                "diamond",
            )
        )

    # ------------------------------------------------------------------------
    # Unassigned test points
    # ------------------------------------------------------------------------

    if not rejected.empty:

        rejected_source = ColumnDataSource(
            rejected
        )

        rejected_renderer = overview.scatter(
            "x",
            "y",
            source=rejected_source,
            size=11,
            color=COLORS["rejected"],
            marker="x",
            line_width=2,
            alpha=0.9,
        )

        overview.add_tools(
            HoverTool(
                renderers=[
                    rejected_renderer
                ],
                tooltips=[
                    ("x", "@x{0.000}"),
                    ("y", "@y{0.000}"),
                    (
                        "Status",
                        "Not assigned",
                    ),
                ],
            )
        )

        overview_legend_items.append(
            (
                COLORS["rejected"],
                "Not assigned",
                "cross",
            )
        )

    overview_key = _create_plot_key(
        overview_legend_items
    )

    # ------------------------------------------------------------------------
    # Overview explanation
    # ------------------------------------------------------------------------

    overview_explanation = Div(
        text=f"""
        <div style="
            font-family:Arial, sans-serif;
            background:{COLORS["surface"]};
            border:1px solid {COLORS["border"]};
            border-radius:9px;
            padding:14px 18px;
            margin-top:16px;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:0.7px;
                color:{COLORS["ink"]};
                margin-bottom:6px;
            ">
                Reading the overview
            </div>

            <div style="
                font-size:13px;
                line-height:1.6;
                color:{COLORS["muted"]};
            ">
                Each color represents one training-to-ideal-function
                assignment. Diamond markers are assigned test points;
                grey crosses are points that could not be assigned.
                Hover over a point to see its coordinates and assignment.
            </div>

        </div>
        """,
        width=PLOT_WIDTH,
    )

    overview_panel = TabPanel(
        title="Overview",
        child=column(
            _create_method_note(),
            overview,
            overview_key,
            overview_explanation,
            sizing_mode="fixed",
        ),
    )

    # ------------------------------------------------------------------------
    # Final dashboard
    # ------------------------------------------------------------------------

    dashboard = column(
        _create_header(
            assigned_count=assigned_count,
            rejected_count=len(rejected),
            function_count=len(selections),
        ),

        Tabs(
            tabs=[
                overview_panel,
                *panels,
            ],
            sizing_mode="fixed",
        ),

        sizing_mode="fixed",
    )

    save(dashboard)