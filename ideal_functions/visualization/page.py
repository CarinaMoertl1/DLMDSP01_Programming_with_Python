"""Assembles the complete HTML dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bokeh.embed import file_html
from bokeh.layouts import Spacer, column, row
from bokeh.resources import CDN

from .cards import _create_alternatives_section, _create_summary_card
from .plots import _create_function_plot, _create_prediction_plot

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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_points = len(mappings)

    assigned_points = int(mappings["ideal_function"].notna().sum())

    unassigned_points = total_points - assigned_points

    # ------------------------------------------------------------------
    # Create the four independent function plots.
    # ------------------------------------------------------------------

    function_plots = []

    for index, selection in enumerate(selections):
        function_plots.append(
            _create_function_plot(training, ideal, mappings, selection, index)
        )

    # ------------------------------------------------------------------
    # Arrange plots manually.
    #
    # We intentionally do NOT use gridplot(..., spacing=...).
    # A manual row/column layout gives us predictable spacing and
    # avoids Bokeh-version compatibility issues.
    # ------------------------------------------------------------------

    if len(function_plots) >= 4:

        top_row = row(function_plots[0], function_plots[1], sizing_mode="stretch_width")

        bottom_row = row(
            function_plots[2], function_plots[3], sizing_mode="stretch_width"
        )

        function_dashboard = column(
            top_row, Spacer(height=55), bottom_row, sizing_mode="stretch_width"
        )

    elif len(function_plots) == 3:

        top_row = row(function_plots[0], function_plots[1], sizing_mode="stretch_width")

        bottom_row = row(function_plots[2], sizing_mode="stretch_width")

        function_dashboard = column(
            top_row, Spacer(height=55), bottom_row, sizing_mode="stretch_width"
        )

    else:

        function_dashboard = column(*function_plots, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Prediction plot.
    # ------------------------------------------------------------------

    prediction_plot = _create_prediction_plot(mappings, ideal, selections)

    # ------------------------------------------------------------------
    # Complete Bokeh dashboard.
    # ------------------------------------------------------------------

    dashboard = column(
        function_dashboard,
        Spacer(height=65),
        prediction_plot,
        sizing_mode="stretch_width",
    )

    dashboard_html = file_html(dashboard, CDN, "Ideal Function Analysis")

    # Extract Bokeh body.
    body_start = dashboard_html.find("<body>")

    body_end = dashboard_html.find("</body>")

    if body_start != -1 and body_end != -1:
        bokeh_body = dashboard_html[body_start + len("<body>") : body_end]
    else:
        bokeh_body = dashboard_html

    # Extract Bokeh head content.
    head_content = ""

    if "<head>" in dashboard_html:
        head_content = dashboard_html.split("<head>", 1)[1].split("</head>", 1)[0]

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

    function_section = function_section.replace("__FUNCTION_DASHBOARD__", bokeh_body)

    # ------------------------------------------------------------------
    # Closest competing candidates.
    # ------------------------------------------------------------------

    alternatives_section = _create_alternatives_section(selections)

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

    output_path.write_text(complete_page, encoding="utf-8")
