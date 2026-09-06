"""
Business Analyst Agent Tools — 7 tools with production-grade error handling.

EVERY tool follows the same pattern from SupportRAGAgent:
1. Validate inputs
2. Execute with timeout
3. Return result string on success
4. Return error string on failure (NEVER raise — let the LLM handle errors gracefully)
"""

import json
import logging
from typing import Optional
from functools import wraps

from langchain_core.tools import tool

from app.config import get_settings
from app.circuit_breaker import rag_circuit_breaker, CircuitBreakerError

logger = logging.getLogger(__name__)
settings = get_settings()


def safe_tool(func):
    """
    Decorator that wraps tool functions with error handling.
    On any exception, returns a user-friendly error string
    instead of crashing the entire agent.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for tool '{func.__name__}': {e}")
            return f"⚠️ Service temporarily unavailable ({e.service_name}). Please try again in {e.time_remaining:.0f} seconds."
        except TimeoutError:
            logger.error(f"Tool '{func.__name__}' timed out")
            return f"⚠️ Tool '{func.__name__}' timed out. Please try a simpler query."
        except Exception as e:
            logger.error(f"Tool '{func.__name__}' failed: {e}", exc_info=True)
            return f"⚠️ Tool '{func.__name__}' encountered an error: {str(e)[:200]}. Please try rephrasing your request."
    return wrapper


@tool
@safe_tool
def profile_dataset(dataset_id: str) -> str:
    """Profile a dataset — analyze structure, types, missing values, and distributions.
    Use this when the user uploads a new dataset or asks about its structure.

    Args:
        dataset_id: The ID of the dataset to profile
    """
    from app.data.manager import DatasetManager
    from app.analysis.profiler import profile_dataset as run_profiler
    from app.db.crud import update_dataset_profile, save_dataset_columns, get_dataset

    # Load dataset
    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found. Please upload a dataset first."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)

    # Run profiler
    profile = run_profiler(df)

    # Save profile to database
    update_dataset_profile(
        dataset_id=dataset_id,
        row_count=profile["shape"]["rows"],
        column_count=profile["shape"]["columns"],
        profile_json=json.dumps(profile),
    )

    # Save column metadata
    save_dataset_columns(dataset_id, profile["columns"])

    # Format for LLM
    output = [f"📊 **Dataset Profile: {ds.name}**\n"]
    output.append(f"**Shape:** {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns")
    output.append(f"**Memory:** {profile['memory_usage_mb']} MB")
    output.append(f"**Data Completeness:** {profile['overall_completeness_pct']}%")
    output.append(f"**Duplicate Rows:** {profile['duplicate_rows']}")

    output.append(f"\n**Column Types:** {profile['column_type_counts']}")
    output.append(f"**Semantic Types:** {profile['semantic_type_counts']}")

    output.append("\n**Column Details:**")
    output.append("| Column | Type | Semantic | Nulls | Unique |")
    output.append("|:-------|:-----|:---------|:------|:-------|")
    for col in profile["columns"]:
        output.append(
            f"| {col['column_name']} | {col['dtype']} | {col['semantic_type']} | "
            f"{col['null_count']} ({col['null_percentage']}%) | {col['unique_count']} |"
        )

    return "\n".join(output)


@tool
@safe_tool
def query_data(dataset_id: str, operation: str) -> str:
    """Execute a data query operation on a dataset.
    Use this to explore data: see rows, get value counts, filter, sort, or group data.

    Args:
        dataset_id: The ID of the dataset to query
        operation: The operation to perform. Options:
            - 'head' — First 10 rows
            - 'tail' — Last 10 rows
            - 'describe' — Statistical summary
            - 'info' — Column types and memory
            - 'shape' — Row × column count
            - 'columns' — List of column names
            - 'value_counts:<column>' — Value counts for a column
            - 'groupby:<group_col>:<agg_col>:<agg_func>' — Group and aggregate
    """
    from app.data.manager import DatasetManager
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    file_path = ds.file_path if ds else None

    return DatasetManager.query_dataset(dataset_id, operation, file_path)


@tool
@safe_tool
def compute_metrics(
    dataset_id: str,
    metric_type: str = "aggregations",
    columns: Optional[str] = None,
    group_by: Optional[str] = None,
    measure: Optional[str] = None,
    aggregation: str = "sum",
) -> str:
    """Compute business metrics and KPIs from a dataset.
    Use this for aggregations, grouped metrics, or growth rate calculations.

    Args:
        dataset_id: The ID of the dataset
        metric_type: Type of metric: 'aggregations' or 'grouped'
        columns: Comma-separated column names to compute metrics for (optional)
        group_by: Column to group by (for grouped metrics)
        measure: Column to measure/aggregate (for grouped metrics)
        aggregation: Aggregation function: sum, mean, count, min, max
    """
    from app.data.manager import DatasetManager
    from app.analysis.metrics import compute_metrics as run_metrics
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)

    col_list = [c.strip() for c in columns.split(",")] if columns else None

    result = run_metrics(
        df,
        metric_type=metric_type,
        columns=col_list,
        group_by=group_by,
        measure=measure,
        aggregation=aggregation,
    )

    if "error" in result:
        return f"⚠️ {result['error']}"

    # Format output
    if metric_type == "grouped" and "results" in result:
        output = [f"📈 **{result.get('summary', 'Grouped Metrics')}**\n"]
        output.append(f"| {result['group_by']} | {result['aggregation']}({result['measure']}) |")
        output.append("|:---|:---|")
        for r in result["results"]:
            output.append(f"| {r['group']} | {r['value']:,.2f} |")
        return "\n".join(output)
    else:
        output = ["📈 **Metrics Summary:**\n"]
        output.append("| Column | Metric | Value |")
        output.append("|:-------|:-------|:------|")
        for r in result.get("results", []):
            if isinstance(r, dict) and "column" in r:
                output.append(f"| {r['column']} | {r['metric']} | {r['value']:,.4f} |")
        return "\n".join(output)


@tool
@safe_tool
def detect_trends(
    dataset_id: str,
    date_column: str,
    value_column: str,
    period: str = "month",
) -> str:
    """Detect trends in time-series data using linear regression.
    Use this when the user asks about trends, growth, or time-based patterns.

    Args:
        dataset_id: The ID of the dataset
        date_column: Name of the date/time column
        value_column: Name of the numeric value column to analyze
        period: Aggregation period: 'day', 'week', 'month', 'quarter', 'year'
    """
    from app.data.manager import DatasetManager
    from app.analysis.trends import detect_trends as run_trends
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)
    result = run_trends(df, date_column, value_column, period)

    if "error" in result:
        return f"⚠️ {result['error']}"

    output = [f"📉 **Trend Analysis: {value_column}**\n"]
    output.append(f"**Direction:** {result['direction']}")
    output.append(f"**Slope:** {result['slope']}")
    output.append(f"**R² (fit):** {result['r_squared']}")
    output.append(f"**p-value:** {result['p_value']} {'✅ significant' if result['is_significant'] else '❌ not significant'}")
    output.append(f"**Data points:** {result['data_points']} {period}s")

    if result.get("period_comparison"):
        pc = result["period_comparison"]
        change_str = f"{pc['change_pct']}%" if pc['change_pct'] is not None else "N/A"
        output.append(f"\n**Latest vs Previous {period}:**")
        output.append(f"  Current: {pc['current_value']:,.2f} | Previous: {pc['previous_value']:,.2f} | Change: {change_str}")

    output.append(f"\n**Summary:** {result['summary']}")
    return "\n".join(output)


@tool
@safe_tool
def detect_anomalies(dataset_id: str, columns: Optional[str] = None) -> str:
    """Detect anomalies and outliers in numeric columns using IQR and Z-score methods.
    Use this when the user asks about outliers, unusual values, or data quality.

    Args:
        dataset_id: The ID of the dataset
        columns: Comma-separated column names to check (optional, checks all numeric if not specified)
    """
    from app.data.manager import DatasetManager
    from app.analysis.anomalies import detect_anomalies as run_anomalies
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)
    col_list = [c.strip() for c in columns.split(",")] if columns else None

    result = run_anomalies(df, col_list)

    if "error" in result:
        return f"⚠️ {result['error']}"

    output = [f"🔍 **Anomaly Detection Report**\n"]
    output.append(f"{result['summary']}\n")

    for col_result in result["results"]:
        if col_result["has_anomalies"]:
            output.append(f"**{col_result['column']}** — anomalies found:")
            for detection in col_result["detections"]:
                if detection["count"] > 0:
                    output.append(
                        f"  - {detection['method'].upper()}: {detection['count']} outliers "
                        f"({detection['percentage']}%)"
                    )
                    if detection.get("lower_bound"):
                        output.append(
                            f"    Bounds: [{detection['lower_bound']:.2f}, {detection['upper_bound']:.2f}]"
                        )

    return "\n".join(output)


@tool
@safe_tool
def compute_correlations(dataset_id: str, method: str = "pearson") -> str:
    """Compute correlations between numeric columns to find relationships.
    Use this when the user asks about relationships between variables.

    Args:
        dataset_id: The ID of the dataset
        method: Correlation method: 'pearson' (linear) or 'spearman' (monotonic)
    """
    from app.data.manager import DatasetManager
    from app.analysis.correlations import compute_correlations as run_correlations
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)
    result = run_correlations(df, method=method)

    if "error" in result:
        return f"⚠️ {result['error']}"

    output = [f"🔗 **Correlation Analysis ({method})**\n"]
    output.append(f"{result['summary']}\n")
    output.append(f"**Columns analyzed:** {result['columns_analyzed']}")
    output.append(f"**Significant correlations (|r| ≥ 0.4):** {result['significant_correlations']}\n")

    if result["top_correlations"]:
        output.append("**Top Correlations:**")
        output.append("| Column A | Column B | Correlation | Strength |")
        output.append("|:---------|:---------|:------------|:---------|")
        for pair in result["top_correlations"]:
            output.append(
                f"| {pair['col_a']} | {pair['col_b']} | {pair['correlation']:.4f} | "
                f"{pair['strength']} {pair['direction']} |"
            )

    return "\n".join(output)


@tool
@safe_tool
def suggest_charts(dataset_id: str) -> str:
    """Suggest the best chart types for visualizing this dataset.
    Use this when the user asks for visualization recommendations.

    Args:
        dataset_id: The ID of the dataset
    """
    from app.data.manager import DatasetManager
    from app.analysis.charts import suggest_charts as run_charts
    from app.db.crud import get_dataset

    ds = get_dataset(dataset_id)
    if not ds:
        return f"⚠️ Dataset '{dataset_id}' not found."

    df = DatasetManager.load_dataset(dataset_id, ds.file_path)
    suggestions = run_charts(df)

    if not suggestions:
        return "⚠️ Could not generate chart suggestions for this dataset."

    output = ["📊 **Recommended Charts:**\n"]
    for i, chart in enumerate(suggestions, 1):
        output.append(f"**{i}. {chart['title']}** ({chart['chart_type']})")
        output.append(f"   - X-axis: {chart.get('x_axis', 'N/A')}")
        output.append(f"   - Y-axis: {chart.get('y_axis', 'N/A')}")
        if chart.get("color_by"):
            output.append(f"   - Color by: {chart['color_by']}")
        if chart.get("aggregation"):
            output.append(f"   - Aggregation: {chart['aggregation']}")
        output.append(f"   - Why: {chart['reason']}\n")

    return "\n".join(output)


@tool
@safe_tool
def search_knowledge_base(query: str, thread_id: str) -> str:
    """Search through uploaded business documents using hybrid retrieval (vector + BM25).
    Use this when the user asks about business definitions, glossary terms, or documentation.

    Args:
        query: The search query
        thread_id: The conversation thread ID
    """
    from app.rag.retriever import rag_retriever

    def _search():
        return rag_retriever(query=query, thread_id=thread_id)

    result = rag_circuit_breaker.call(_search)

    if not result or result == "no relevant document content found":
        return "No relevant content found in uploaded documents. The user may not have uploaded any business documents yet."

    return f"📄 **Knowledge Base Results:**\n\n{result}"


def get_tools() -> list:
    """Return all available tools for the business analyst agent."""
    return [
        profile_dataset,
        query_data,
        compute_metrics,
        detect_trends,
        detect_anomalies,
        compute_correlations,
        suggest_charts,
        search_knowledge_base,
    ]
