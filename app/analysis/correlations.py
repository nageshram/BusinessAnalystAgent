"""
Correlation Engine — Statistical correlation analysis.

Capabilities:
- Pearson correlation (linear relationships)
- Spearman rank correlation (monotonic relationships)
- Top N strongest correlations
- Correlation matrix as a structured dict
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_correlations(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict:
    """
    Compute correlation matrix and find strongest relationships.

    method: 'pearson' or 'spearman'

    Returns:
    {
        "method": "pearson",
        "top_correlations": [{"col_a": "price", "col_b": "quantity", "correlation": -0.85}, ...],
        "matrix": {"col_a": {"col_b": 0.85, ...}, ...},
        "summary": "Strongest correlation: price ↔ quantity (r=-0.85)"
    }
    """
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "error": "Need at least 2 numeric columns for correlation analysis",
            "top_correlations": [],
        }

    try:
        corr_matrix = df[numeric_cols].corr(method=method)

        # Extract top correlations (excluding self-correlations)
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                corr_val = corr_matrix.loc[col_a, col_b]

                if pd.notna(corr_val):
                    # Classify strength
                    abs_corr = abs(corr_val)
                    if abs_corr >= 0.8:
                        strength = "very strong"
                    elif abs_corr >= 0.6:
                        strength = "strong"
                    elif abs_corr >= 0.4:
                        strength = "moderate"
                    elif abs_corr >= 0.2:
                        strength = "weak"
                    else:
                        strength = "negligible"

                    direction = "positive" if corr_val > 0 else "negative"

                    pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": round(float(corr_val), 4),
                        "abs_correlation": round(abs_corr, 4),
                        "strength": strength,
                        "direction": direction,
                    })

        # Sort by absolute correlation (strongest first)
        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        top_pairs = pairs[:top_n]

        # Build matrix dict
        matrix = {}
        for col in numeric_cols:
            matrix[col] = {
                other_col: round(float(corr_matrix.loc[col, other_col]), 4)
                for other_col in numeric_cols
                if pd.notna(corr_matrix.loc[col, other_col])
            }

        # Summary
        if top_pairs:
            strongest = top_pairs[0]
            summary = (
                f"Strongest {method} correlation: {strongest['col_a']} ↔ {strongest['col_b']} "
                f"(r={strongest['correlation']}, {strongest['strength']} {strongest['direction']})"
            )
        else:
            summary = "No significant correlations found"

        # Count significant correlations
        significant = [p for p in pairs if p["abs_correlation"] >= 0.4]

        return {
            "method": method,
            "columns_analyzed": len(numeric_cols),
            "total_pairs": len(pairs),
            "significant_correlations": len(significant),
            "top_correlations": top_pairs,
            "matrix": matrix,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}", exc_info=True)
        return {"error": str(e), "top_correlations": []}
