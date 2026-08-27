"""HTML "card" snippets used in the dashboard page (no Bokeh involved)."""

from __future__ import annotations

from .design import FUNCTION_COLORS

# ============================================================================
# Summary cards
# ============================================================================


def _create_summary_card(
    total_points: int, assigned_points: int, unassigned_points: int, selections: list
) -> str:
    """Create the dashboard summary cards."""

    assignment_rate = assigned_points / total_points * 100 if total_points else 0.0

    function_items = "".join(f"""
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
        """ for selection in selections)

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


def _create_alternatives_card(selection, accent_color: str) -> str:
    """Create a small table of the closest competing ideal functions.

    The selected ideal function is highlighted so the reader can see
    at a glance how close the runner-up candidates were.
    """
    rows = "".join(f"""
        <tr class="{
            'alt-row-selected'
            if candidate.ideal_column == selection.ideal_column
            else ''
        }">
            <td>{candidate.ideal_column}</td>
            <td>{candidate.sum_squared_error:.6g}</td>
        </tr>
        """ for candidate in selection.alternatives)

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


def _create_alternatives_section(selections: list) -> str:
    """Create the section listing competing candidates per selection."""
    cards = "".join(
        _create_alternatives_card(
            selection, FUNCTION_COLORS[index % len(FUNCTION_COLORS)]
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
