"""
Chart Suggestion Engine — Recommend visualizations based on data shape.

Analyzes column types and semantic classification to recommend
appropriate chart types with axis mappings.

Supports: bar, line, scatter, pie, histogram, heatmap, box
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def suggest_charts(
    df: pd.DataFrame,
    column_profiles: Optional[list[dict]] = None,
    max_suggestions: int = 5,
) -> list[dict]:
    """
    Recommend chart types based on the dataset's structure.

    Each suggestion includes:
    - chart_type: bar, line, scatter, pie, histogram, heatmap, box
    - title: Descriptive chart title
    - x_axis, y_axis: Column mappings
    - aggregation: sum, mean, count, etc.
    - reason: Why this chart was suggested
    - priority: 1 (highest) to 5 (lowest)

    Uses column semantic types (from profiler) if available,
    otherwise infers from pandas dtypes.
    """
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Also try to detect datetime columns stored as strings
    for col in categorical_cols[:]:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0.7 * len(df):
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    # ── Chart 1: Line chart for time series ──
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        for num_col in numeric_cols[:2]:
            suggestions.append({
                "chart_type": "line",
                "title": f"{num_col} Over Time",
                "x_axis": date_col,
                "y_axis": num_col,
                "aggregation": "sum",
                "reason": f"Time-series trend visualization for {num_col}",
                "priority": 1,
            })

    # ── Chart 2: Bar chart for categorical × numeric ──
    if categorical_cols and numeric_cols:
        # Pick categorical with reasonable cardinality
        for cat_col in categorical_cols:
            n_unique = df[cat_col].nunique()
            if 2 <= n_unique <= 20:
                for num_col in numeric_cols[:1]:
                    suggestions.append({
                        "chart_type": "bar",
                        "title": f"{num_col} by {cat_col}",
                        "x_axis": cat_col,
                        "y_axis": num_col,
                        "aggregation": "sum",
                        "reason": f"Compare {num_col} across {n_unique} {cat_col} categories",
                        "priority": 2,
                    })
                break

    # ── Chart 3: Pie chart for composition ──
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols:
            n_unique = df[cat_col].nunique()
            if 2 <= n_unique <= 8:
                suggestions.append({
                    "chart_type": "pie",
                    "title": f"{numeric_cols[0]} Distribution by {cat_col}",
                    "x_axis": cat_col,
                    "y_axis": numeric_cols[0],
                    "aggregation": "sum",
                    "reason": f"Show composition of {numeric_cols[0]} across {cat_col} ({n_unique} segments)",
                    "priority": 3,
                })
                break

    # ── Chart 4: Scatter plot for correlation ──
    if len(numeric_cols) >= 2:
        suggestions.append({
            "chart_type": "scatter",
            "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
            "x_axis": numeric_cols[0],
            "y_axis": numeric_cols[1],
            "aggregation": None,
            "color_by": categorical_cols[0] if categorical_cols else None,
            "reason": f"Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}",
            "priority": 3,
        })

    # ── Chart 5: Histogram for distribution ──
    if numeric_cols:
        suggestions.append({
            "chart_type": "histogram",
            "title": f"Distribution of {numeric_cols[0]}",
            "x_axis": numeric_cols[0],
            "y_axis": "count",
            "aggregation": "count",
            "bins": min(30, max(10, int(np.sqrt(len(df))))),
            "reason": f"Understand the distribution of {numeric_cols[0]} values",
            "priority": 4,
        })

    # ── Chart 6: Box plot for outlier detection ──
    if numeric_cols and categorical_cols:
        for cat_col in categorical_cols:
            if 2 <= df[cat_col].nunique() <= 10:
                suggestions.append({
                    "chart_type": "box",
                    "title": f"{numeric_cols[0]} by {cat_col} (Box Plot)",
                    "x_axis": cat_col,
                    "y_axis": numeric_cols[0],
                    "aggregation": None,
                    "reason": f"Compare {numeric_cols[0]} distributions and outliers across {cat_col}",
                    "priority": 4,
                })
                break

    # ── Chart 7: Heatmap for correlations ──
    if len(numeric_cols) >= 3:
        suggestions.append({
            "chart_type": "heatmap",
            "title": "Correlation Heatmap",
            "columns": numeric_cols[:10],
            "aggregation": None,
            "reason": f"Visualize correlations between {len(numeric_cols[:10])} numeric columns",
            "priority": 5,
        })

    # Sort by priority and limit
    suggestions.sort(key=lambda x: x["priority"])
    suggestions = suggestions[:max_suggestions]

    logger.info(f"Suggested {len(suggestions)} charts for dataset with {len(df.columns)} columns")
    return suggestions
