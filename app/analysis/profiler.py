"""
Dataset Profiler — Automated EDA at upload time.

Generates a comprehensive profile of a dataset:
- Shape (rows × columns)
- Column types (numeric, categorical, datetime, text)
- Missing values (count + percentage per column)
- Basic distributions (mean, median, std, quartiles for numeric)
- Top categories for categorical columns
- Memory usage
- Duplicate detection
- Semantic type classification (measure, dimension, temporal, identifier)
"""

import json
import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def classify_semantic_type(series: pd.Series, col_name: str) -> str:
    """
    Classify a column's semantic type based on its name and data.

    Categories:
    - 'temporal': Date/time columns
    - 'identifier': IDs, codes, keys
    - 'measure': Numeric values suitable for aggregation
    - 'dimension': Categorical values for grouping
    - 'text': Free-form text
    """
    name_lower = col_name.lower().strip()

    # Temporal indicators
    temporal_keywords = {"date", "time", "timestamp", "year", "month", "day", "created", "updated", "period", "quarter"}
    if any(kw in name_lower for kw in temporal_keywords):
        return "temporal"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"

    # Identifier indicators
    id_keywords = {"id", "key", "code", "uuid", "sku", "number", "num", "no", "index"}
    if any(name_lower == kw or name_lower.endswith(f"_{kw}") or name_lower.startswith(f"{kw}_") for kw in id_keywords):
        return "identifier"

    # Numeric → measure or identifier
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(len(series), 1)
        # If very high cardinality numeric (>90% unique), likely an ID
        if unique_ratio > 0.9 and series.nunique() > 100:
            return "identifier"
        return "measure"

    # Categorical
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        avg_len = series.dropna().astype(str).str.len().mean()
        unique_ratio = series.nunique() / max(len(series), 1)

        # Long text strings → text
        if avg_len > 100:
            return "text"
        # Low cardinality → dimension
        if unique_ratio < 0.5 or series.nunique() < 50:
            return "dimension"
        return "text"

    return "dimension"


def _numeric_stats(series: pd.Series) -> dict:
    """Compute statistics for a numeric column."""
    clean = series.dropna()
    if clean.empty:
        return {}

    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "q25": round(float(clean.quantile(0.25)), 4),
        "q75": round(float(clean.quantile(0.75)), 4),
        "skewness": round(float(clean.skew()), 4),
        "kurtosis": round(float(clean.kurtosis()), 4),
        "zeros": int((clean == 0).sum()),
        "negatives": int((clean < 0).sum()),
    }


def _categorical_stats(series: pd.Series, top_n: int = 10) -> dict:
    """Compute statistics for a categorical column."""
    clean = series.dropna()
    if clean.empty:
        return {}

    value_counts = clean.value_counts()
    top_values = [
        {"value": str(val), "count": int(cnt), "percentage": round(cnt / len(clean) * 100, 2)}
        for val, cnt in value_counts.head(top_n).items()
    ]

    return {
        "unique_count": int(clean.nunique()),
        "top_values": top_values,
        "avg_length": round(float(clean.astype(str).str.len().mean()), 2),
    }


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive dataset profile.

    Returns a dictionary with:
    - shape: {rows, columns}
    - memory_usage_mb: float
    - duplicate_rows: int
    - columns: list of column profiles
    - summary: human-readable summary string
    """
    rows, cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    duplicate_count = int(df.duplicated().sum())

    columns = []
    for idx, col_name in enumerate(df.columns):
        series = df[col_name]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / max(rows, 1) * 100, 2)
        unique_count = int(series.nunique())
        dtype = str(series.dtype)

        # Semantic classification
        semantic_type = classify_semantic_type(series, col_name)

        # Type-specific stats
        if pd.api.types.is_numeric_dtype(series):
            stats = _numeric_stats(series)
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            stats = {
                "min": str(clean.min()) if not clean.empty else None,
                "max": str(clean.max()) if not clean.empty else None,
                "range_days": int((clean.max() - clean.min()).days) if len(clean) > 1 else 0,
            }
            col_type = "datetime"
        else:
            stats = _categorical_stats(series)
            col_type = "categorical"

        columns.append({
            "column_name": col_name,
            "column_index": idx,
            "dtype": dtype,
            "column_type": col_type,
            "semantic_type": semantic_type,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "stats": stats,
        })

    # Build summary
    numeric_cols = [c for c in columns if c["column_type"] == "numeric"]
    categorical_cols = [c for c in columns if c["column_type"] == "categorical"]
    datetime_cols = [c for c in columns if c["column_type"] == "datetime"]
    measures = [c for c in columns if c["semantic_type"] == "measure"]
    dimensions = [c for c in columns if c["semantic_type"] == "dimension"]

    total_nulls = sum(c["null_count"] for c in columns)
    total_cells = rows * cols
    overall_completeness = round((1 - total_nulls / max(total_cells, 1)) * 100, 2)

    summary = (
        f"Dataset has {rows:,} rows × {cols} columns ({memory_mb} MB). "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
        f"{len(datetime_cols)} datetime columns. "
        f"{len(measures)} measures, {len(dimensions)} dimensions. "
        f"Data completeness: {overall_completeness}%. "
        f"{duplicate_count} duplicate rows found."
    )

    profile = {
        "shape": {"rows": rows, "columns": cols},
        "memory_usage_mb": memory_mb,
        "duplicate_rows": duplicate_count,
        "overall_completeness_pct": overall_completeness,
        "column_type_counts": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "datetime": len(datetime_cols),
        },
        "semantic_type_counts": {
            "measure": len(measures),
            "dimension": len(dimensions),
            "temporal": len(datetime_cols),
            "identifier": len([c for c in columns if c["semantic_type"] == "identifier"]),
            "text": len([c for c in columns if c["semantic_type"] == "text"]),
        },
        "columns": columns,
        "summary": summary,
    }

    logger.info(f"Profiled dataset: {rows}x{cols}, completeness={overall_completeness}%")
    return profile
