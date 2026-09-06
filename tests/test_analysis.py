"""Tests for the analysis engines — profiler, metrics, trends, anomalies, correlations, charts."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.profiler import profile_dataset
from app.analysis.metrics import compute_aggregations, compute_grouped_metrics, compute_growth_rate
from app.analysis.trends import detect_trends
from app.analysis.anomalies import detect_anomalies
from app.analysis.correlations import compute_correlations
from app.analysis.charts import suggest_charts


# ──────────────────────────────────────────────
# Test Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a sample business dataset for testing."""
    np.random.seed(42)
    n = 100

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(1000, 5000, n).round(2),
        "quantity": np.random.randint(10, 100, n),
        "price": np.random.uniform(10, 100, n).round(2),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "product": np.random.choice(["Widget A", "Widget B", "Widget C"], n),
        "customer_id": np.arange(n),
    })


@pytest.fixture
def df_with_nulls():
    """Dataset with missing values."""
    df = pd.DataFrame({
        "revenue": [100, 200, None, 400, 500],
        "region": ["North", None, "South", "East", None],
    })
    return df


@pytest.fixture
def df_with_anomalies():
    """Dataset with known outliers."""
    values = list(np.random.normal(100, 10, 97))
    values.extend([500, 600, -200])  # Outliers
    return pd.DataFrame({"value": values})


# ──────────────────────────────────────────────
# Profiler Tests
# ──────────────────────────────────────────────

class TestProfiler:
    def test_basic_profile(self, sample_df):
        profile = profile_dataset(sample_df)
        assert profile["shape"]["rows"] == 100
        assert profile["shape"]["columns"] == 7
        assert profile["duplicate_rows"] >= 0
        assert "columns" in profile
        assert len(profile["columns"]) == 7

    def test_column_types(self, sample_df):
        profile = profile_dataset(sample_df)
        col_types = {c["column_name"]: c["column_type"] for c in profile["columns"]}
        assert col_types["revenue"] == "numeric"
        assert col_types["region"] == "categorical"

    def test_null_detection(self, df_with_nulls):
        profile = profile_dataset(df_with_nulls)
        rev_col = next(c for c in profile["columns"] if c["column_name"] == "revenue")
        assert rev_col["null_count"] == 1

    def test_summary_string(self, sample_df):
        profile = profile_dataset(sample_df)
        assert "summary" in profile
        assert "100" in profile["summary"]


# ──────────────────────────────────────────────
# Metrics Tests
# ──────────────────────────────────────────────

class TestMetrics:
    def test_aggregations(self, sample_df):
        results = compute_aggregations(sample_df, ["revenue", "quantity"])
        assert len(results) > 0
        metrics = {(r["column"], r["metric"]): r["value"] for r in results}
        assert ("revenue", "sum") in metrics
        assert ("revenue", "mean") in metrics

    def test_grouped_metrics(self, sample_df):
        result = compute_grouped_metrics(sample_df, "revenue", "region", "sum")
        assert "results" in result
        assert len(result["results"]) <= 4  # 4 regions

    def test_growth_rate(self, sample_df):
        result = compute_growth_rate(sample_df, "date", "revenue", "month")
        assert "growth_rates" in result
        assert len(result["growth_rates"]) >= 1


# ──────────────────────────────────────────────
# Trend Tests
# ──────────────────────────────────────────────

class TestTrends:
    def test_trend_detection(self, sample_df):
        result = detect_trends(sample_df, "date", "revenue", "month")
        assert "direction" in result
        assert "slope" in result
        assert "r_squared" in result
        assert "p_value" in result

    def test_missing_column(self, sample_df):
        result = detect_trends(sample_df, "date", "nonexistent", "month")
        assert "error" in result


# ──────────────────────────────────────────────
# Anomaly Tests
# ──────────────────────────────────────────────

class TestAnomalies:
    def test_anomaly_detection(self, df_with_anomalies):
        result = detect_anomalies(df_with_anomalies)
        assert result["total_columns_checked"] == 1
        assert result["columns_with_anomalies"] >= 1

    def test_iqr_method(self, df_with_anomalies):
        result = detect_anomalies(df_with_anomalies, methods=["iqr"])
        detections = result["results"][0]["detections"]
        iqr_result = next(d for d in detections if d["method"] == "iqr")
        assert iqr_result["count"] > 0


# ──────────────────────────────────────────────
# Correlation Tests
# ──────────────────────────────────────────────

class TestCorrelations:
    def test_pearson_correlation(self, sample_df):
        result = compute_correlations(sample_df, method="pearson")
        assert "top_correlations" in result
        assert "matrix" in result

    def test_spearman_correlation(self, sample_df):
        result = compute_correlations(sample_df, method="spearman")
        assert "top_correlations" in result


# ──────────────────────────────────────────────
# Chart Tests
# ──────────────────────────────────────────────

class TestCharts:
    def test_chart_suggestions(self, sample_df):
        suggestions = suggest_charts(sample_df)
        assert len(suggestions) > 0
        for s in suggestions:
            assert "chart_type" in s
            assert "title" in s
            assert "reason" in s
