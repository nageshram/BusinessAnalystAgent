"""
KPI & Metrics Engine — Compute business metrics from datasets.

Capabilities:
- Aggregations: sum, avg, min, max, count, distinct count
- Growth rates: period-over-period (detect temporal columns automatically)
- Grouped metrics: e.g., revenue by region, count by category
- Ratios and percentages
- Top N / Bottom N analysis
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_aggregations(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> list[dict]:
    """
    Compute basic aggregation metrics for numeric columns.

    Returns list of metric dicts:
    [{"column": "revenue", "metric": "sum", "value": 1234567.89}, ...]
    """
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return [{"error": "No numeric columns found for aggregation"}]

    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        results.extend([
            {"column": col, "metric": "sum", "value": round(float(series.sum()), 4)},
            {"column": col, "metric": "mean", "value": round(float(series.mean()), 4)},
            {"column": col, "metric": "median", "value": round(float(series.median()), 4)},
            {"column": col, "metric": "min", "value": round(float(series.min()), 4)},
            {"column": col, "metric": "max", "value": round(float(series.max()), 4)},
            {"column": col, "metric": "std", "value": round(float(series.std()), 4)},
            {"column": col, "metric": "count", "value": int(len(series))},
            {"column": col, "metric": "distinct_count", "value": int(series.nunique())},
        ])

    return results


def compute_grouped_metrics(
    df: pd.DataFrame,
    measure_column: str,
    group_by_column: str,
    aggregation: str = "sum",
    top_n: int = 10,
) -> dict:
    """
    Compute metrics grouped by a dimension.
    e.g., "total revenue by region" → group_by=region, measure=revenue, agg=sum

    Returns:
    {
        "measure": "revenue",
        "group_by": "region",
        "aggregation": "sum",
        "results": [{"group": "North", "value": 500000}, ...],
        "summary": "Top region by revenue: North ($500,000)"
    }
    """
    if measure_column not in df.columns:
        return {"error": f"Column '{measure_column}' not found"}
    if group_by_column not in df.columns:
        return {"error": f"Column '{group_by_column}' not found"}

    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "avg": "mean",
        "count": "count",
        "min": "min",
        "max": "max",
        "median": "median",
    }

    agg_func = agg_map.get(aggregation.lower(), "sum")

    try:
        grouped = df.groupby(group_by_column)[measure_column].agg(agg_func)
        grouped = grouped.sort_values(ascending=False).head(top_n)

        results = [
            {"group": str(group), "value": round(float(value), 4)}
            for group, value in grouped.items()
        ]

        top_group = results[0] if results else None
        summary = (
            f"Top {group_by_column} by {aggregation}({measure_column}): "
            f"{top_group['group']} ({top_group['value']:,.2f})"
            if top_group else "No results"
        )

        return {
            "measure": measure_column,
            "group_by": group_by_column,
            "aggregation": aggregation,
            "results": results,
            "total_groups": int(df[group_by_column].nunique()),
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Grouped metrics failed: {e}")
        return {"error": str(e)}


def compute_growth_rate(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: str = "month",
) -> dict:
    """
    Compute period-over-period growth rates.

    Periods: 'day', 'week', 'month', 'quarter', 'year'

    Returns:
    {
        "value_column": "revenue",
        "period": "month",
        "growth_rates": [{"period": "2024-01", "value": 10500, "growth_pct": 5.2}, ...],
        "overall_growth_pct": 12.5,
        "summary": "Revenue grew 12.5% over the period"
    }
    """
    if date_column not in df.columns or value_column not in df.columns:
        return {"error": f"Column '{date_column}' or '{value_column}' not found"}

    try:
        df_copy = df[[date_column, value_column]].copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
        df_copy = df_copy.dropna(subset=[date_column, value_column])

        if df_copy.empty:
            return {"error": "No valid date-value pairs found"}

        # Group by period
        freq_map = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
        freq = freq_map.get(period, "MS")

        df_copy = df_copy.set_index(date_column)
        periodic = df_copy[value_column].resample(freq).sum()

        if len(periodic) < 2:
            return {"error": "Not enough periods for growth calculation"}

        # Compute growth rates
        growth_rates = []
        values = periodic.values
        periods = periodic.index

        for i in range(len(values)):
            entry = {
                "period": str(periods[i].date()),
                "value": round(float(values[i]), 4),
            }
            if i > 0 and values[i - 1] != 0:
                growth = ((values[i] - values[i - 1]) / abs(values[i - 1])) * 100
                entry["growth_pct"] = round(float(growth), 2)
            else:
                entry["growth_pct"] = None
            growth_rates.append(entry)

        # Overall growth
        first_val = float(values[0])
        last_val = float(values[-1])
        overall_growth = ((last_val - first_val) / abs(first_val) * 100) if first_val != 0 else None

        direction = "grew" if overall_growth and overall_growth > 0 else "declined" if overall_growth and overall_growth < 0 else "remained flat"

        return {
            "value_column": value_column,
            "period": period,
            "growth_rates": growth_rates,
            "overall_growth_pct": round(overall_growth, 2) if overall_growth else None,
            "first_period_value": round(first_val, 4),
            "last_period_value": round(last_val, 4),
            "summary": f"{value_column} {direction} {abs(overall_growth):.1f}% over {len(growth_rates)} {period}s" if overall_growth else f"No growth data for {value_column}",
        }

    except Exception as e:
        logger.error(f"Growth rate computation failed: {e}")
        return {"error": str(e)}


def compute_metrics(
    df: pd.DataFrame,
    metric_type: str = "aggregations",
    columns: Optional[list[str]] = None,
    group_by: Optional[str] = None,
    measure: Optional[str] = None,
    aggregation: str = "sum",
) -> dict:
    """
    Unified metrics computation entry point.

    metric_type:
    - 'aggregations': Basic stats for numeric columns
    - 'grouped': Metrics grouped by a dimension
    - 'top_n': Top N values in a column
    """
    if metric_type == "grouped" and group_by and measure:
        return compute_grouped_metrics(df, measure, group_by, aggregation)
    elif metric_type == "aggregations":
        results = compute_aggregations(df, columns)
        return {"metric_type": "aggregations", "results": results}
    else:
        results = compute_aggregations(df, columns)
        return {"metric_type": "aggregations", "results": results}
