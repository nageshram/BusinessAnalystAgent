"""
Anomaly Detection Engine — Find outliers in business data.

Methods:
- IQR (Interquartile Range): Classic box-plot method
- Z-Score: Standard deviation-based detection
- Value range: Min/max boundary violations
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def detect_iqr_anomalies(
    series: pd.Series,
    multiplier: float = 1.5,
) -> dict:
    """
    Detect anomalies using IQR method.

    IQR = Q3 - Q1
    Lower bound = Q1 - multiplier * IQR
    Upper bound = Q3 + multiplier * IQR

    multiplier=1.5 for outliers, 3.0 for extreme outliers.
    """
    clean = series.dropna()
    if clean.empty or len(clean) < 4:
        return {"method": "iqr", "anomalies": [], "count": 0}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (clean < lower) | (clean > upper)
    anomalies = clean[mask]

    return {
        "method": "iqr",
        "multiplier": multiplier,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "count": int(mask.sum()),
        "percentage": round(float(mask.sum()) / len(clean) * 100, 2),
        "anomalies": [
            {"index": int(idx), "value": round(float(val), 4)}
            for idx, val in anomalies.head(20).items()  # Limit to 20
        ],
    }


def detect_zscore_anomalies(
    series: pd.Series,
    threshold: float = 3.0,
) -> dict:
    """
    Detect anomalies using Z-score method.
    Values with |z-score| > threshold are anomalies.
    """
    clean = series.dropna()
    if clean.empty or len(clean) < 3:
        return {"method": "zscore", "anomalies": [], "count": 0}

    mean = float(clean.mean())
    std = float(clean.std())

    if std == 0:
        return {"method": "zscore", "anomalies": [], "count": 0, "note": "Zero variance"}

    z_scores = ((clean - mean) / std).abs()
    mask = z_scores > threshold
    anomalies = clean[mask]

    return {
        "method": "zscore",
        "threshold": threshold,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "count": int(mask.sum()),
        "percentage": round(float(mask.sum()) / len(clean) * 100, 2),
        "anomalies": [
            {"index": int(idx), "value": round(float(val), 4), "z_score": round(float(z_scores[idx]), 2)}
            for idx, val in anomalies.head(20).items()
        ],
    }


def detect_anomalies(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    methods: Optional[list[str]] = None,
) -> dict:
    """
    Run anomaly detection across numeric columns.

    Returns per-column anomaly reports with multiple detection methods.
    """
    methods = methods or ["iqr", "zscore"]

    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "No numeric columns found for anomaly detection", "results": []}

    results = []
    total_anomalies = 0

    for col in numeric_cols:
        series = df[col]
        col_result = {"column": col, "detections": []}

        if "iqr" in methods:
            iqr_result = detect_iqr_anomalies(series)
            col_result["detections"].append(iqr_result)
            total_anomalies += iqr_result["count"]

        if "zscore" in methods:
            zscore_result = detect_zscore_anomalies(series)
            col_result["detections"].append(zscore_result)

        # Has anomalies in any method?
        col_result["has_anomalies"] = any(
            d["count"] > 0 for d in col_result["detections"]
        )

        results.append(col_result)

    # Sort: columns with anomalies first
    results.sort(key=lambda x: x["has_anomalies"], reverse=True)

    cols_with_anomalies = sum(1 for r in results if r["has_anomalies"])

    return {
        "total_columns_checked": len(numeric_cols),
        "columns_with_anomalies": cols_with_anomalies,
        "summary": (
            f"Found anomalies in {cols_with_anomalies}/{len(numeric_cols)} numeric columns "
            f"using {', '.join(methods)} methods"
        ),
        "results": results,
    }
