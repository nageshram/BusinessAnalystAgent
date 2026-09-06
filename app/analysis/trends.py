"""
Trend Detection Engine — Time-series analysis for business data.

Capabilities:
- Linear regression (slope, R², p-value)
- Period comparison (this period vs. last period)
- Moving averages (SMA)
- Direction classification (increasing, decreasing, flat)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def detect_trends(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    period: str = "month",
) -> dict:
    """
    Detect trends in time-series data using linear regression.

    Returns:
    {
        "column": "revenue",
        "direction": "increasing",
        "slope": 1234.56,
        "r_squared": 0.89,
        "p_value": 0.001,
        "is_significant": True,
        "period_comparison": {...},
        "moving_averages": [...],
        "summary": "Revenue shows a significant increasing trend (R²=0.89, p<0.01)"
    }
    """
    if date_column not in df.columns:
        return {"error": f"Date column '{date_column}' not found"}
    if value_column not in df.columns:
        return {"error": f"Value column '{value_column}' not found"}

    try:
        df_copy = df[[date_column, value_column]].copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
        df_copy = df_copy.dropna(subset=[date_column, value_column])
        df_copy = df_copy.sort_values(date_column)

        if len(df_copy) < 3:
            return {"error": "Need at least 3 data points for trend detection"}

        # Aggregate by period
        freq_map = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
        freq = freq_map.get(period, "MS")

        df_copy = df_copy.set_index(date_column)
        periodic = df_copy[value_column].resample(freq).sum().dropna()

        if len(periodic) < 3:
            return {"error": f"Not enough {period}s for trend detection (need ≥3)"}

        # Linear regression
        x = np.arange(len(periodic))
        y = periodic.values.astype(float)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Direction
        if p_value > 0.05:
            direction = "flat (not statistically significant)"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Period comparison (last period vs previous)
        period_comparison = None
        if len(periodic) >= 2:
            last_val = float(periodic.iloc[-1])
            prev_val = float(periodic.iloc[-2])
            change = last_val - prev_val
            change_pct = (change / abs(prev_val) * 100) if prev_val != 0 else None

            period_comparison = {
                "current_period": str(periodic.index[-1].date()),
                "previous_period": str(periodic.index[-2].date()),
                "current_value": round(last_val, 4),
                "previous_value": round(prev_val, 4),
                "change": round(change, 4),
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
            }

        # Moving averages (3-period SMA)
        moving_averages = []
        if len(periodic) >= 3:
            sma = periodic.rolling(window=3).mean().dropna()
            for date, value in sma.items():
                moving_averages.append({
                    "period": str(date.date()),
                    "sma_3": round(float(value), 4),
                })

        # Summary
        sig_str = f"R²={r_squared:.2f}, p={'<0.01' if p_value < 0.01 else f'={p_value:.3f}'}"
        summary = f"{value_column} shows a {'significant ' if p_value < 0.05 else ''}{direction} trend ({sig_str}) over {len(periodic)} {period}s"

        return {
            "column": value_column,
            "date_column": date_column,
            "period": period,
            "direction": direction,
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r_squared), 4),
            "p_value": round(float(p_value), 6),
            "std_error": round(float(std_err), 6),
            "is_significant": bool(p_value < 0.05),
            "data_points": len(periodic),
            "period_comparison": period_comparison,
            "moving_averages": moving_averages[-10:],  # Last 10 for brevity
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Trend detection failed: {e}", exc_info=True)
        return {"error": str(e)}


def detect_all_trends(
    df: pd.DataFrame,
    date_column: str,
    value_columns: Optional[list[str]] = None,
    period: str = "month",
) -> list[dict]:
    """
    Detect trends for multiple numeric columns.
    If value_columns is None, detects trends for all numeric columns.
    """
    if value_columns is None:
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in value_columns:
        trend = detect_trends(df, date_column, col, period)
        results.append(trend)

    # Sort by significance (most significant first)
    results.sort(key=lambda x: x.get("p_value", 1.0))
    return results
