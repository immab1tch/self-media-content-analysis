# -*- coding: utf-8 -*-
"""
analyzer 模块单元测试。

覆盖：各分析函数返回格式、run_analysis 统一调度入口。
注意：分析函数只返回字符串文本，不返回图表对象。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analyzer import (  # noqa: E402
    compute_basic_stats,
    correlation_analysis,
    content_type_analysis,
    describe_statistics,
    distribution_analysis,
    run_analysis,
    top_content_analysis,
    trend_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """构造测试数据（已预处理过的形态）。"""
    return pd.DataFrame({
        "发布日期": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-04", "2025-01-05",
        ]),
        "内容标题": ["标题A", "标题B", "标题C", "标题D", "标题E"],
        "内容类型": ["短视频", "图文", "短视频", "长视频", "图文"],
        "播放量": [1000, 2000, 3000, 4000, 5000],
        "点赞数": [100, 200, 300, 400, 500],
        "评论数": [10, 20, 30, 40, 50],
        "转发数": [5, 10, 15, 20, 25],
        "收藏数": [50, 60, 70, 80, 90],
    })


# ---------------------------------------------------------------------------
# 返回格式通用校验
# ---------------------------------------------------------------------------

def _assert_analysis_text(result, expected_prefix):
    """断言分析结果为非空字符串且以指定前缀开头。"""
    assert isinstance(result, str), f"返回值应为 str，实际为 {type(result)}"
    assert len(result) > 0, "返回值不应为空"
    assert result.startswith(expected_prefix), (
        f"返回值应以 '{expected_prefix}' 开头，实际开头为：{result[:20]}"
    )


# ---------------------------------------------------------------------------
# describe_statistics
# ---------------------------------------------------------------------------

class TestDescribeStatistics:
    """描述性统计。"""

    def test_returns_str(self, df):
        result = describe_statistics(df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_starts_with_prefix(self, df):
        result = describe_statistics(df)
        assert result.startswith("[描述性统计")

    def test_contains_numeric_columns(self, df):
        result = describe_statistics(df)
        assert "播放量" in result


# ---------------------------------------------------------------------------
# correlation_analysis
# ---------------------------------------------------------------------------

class TestCorrelationAnalysis:
    """相关性分析。"""

    def test_returns_str(self, df):
        result = correlation_analysis(df)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, df):
        result = correlation_analysis(df)
        assert result.startswith("[相关性分析")

    def test_contains_correlation_keyword(self, df):
        result = correlation_analysis(df)
        # 应包含相关系数或相关性的描述
        assert ("相关" in result) or ("系数" in result)


# ---------------------------------------------------------------------------
# trend_analysis
# ---------------------------------------------------------------------------

class TestTrendAnalysis:
    """趋势分析。"""

    def test_returns_str(self, df):
        result = trend_analysis(df)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, df):
        result = trend_analysis(df)
        assert result.startswith("[趋势分析")

    def test_custom_metric(self, df):
        """指定点赞数作为指标应正常工作。"""
        result = trend_analysis(df, metric="点赞数")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_date_col(self, df):
        """自定义日期列。"""
        result = trend_analysis(df, date_col="发布日期", metric="播放量")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# content_type_analysis
# ---------------------------------------------------------------------------

class TestContentTypeAnalysis:
    """内容类型占比分析。"""

    def test_returns_str(self, df):
        result = content_type_analysis(df)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, df):
        result = content_type_analysis(df)
        assert result.startswith("[内容类型占比分析")

    def test_contains_type_names(self, df):
        result = content_type_analysis(df)
        assert "短视频" in result or "图文" in result


# ---------------------------------------------------------------------------
# top_content_analysis
# ---------------------------------------------------------------------------

class TestTopContentAnalysis:
    """Top N 内容分析。"""

    def test_returns_str(self, df):
        result = top_content_analysis(df)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, df):
        result = top_content_analysis(df)
        assert result.startswith("[Top")

    def test_custom_n(self, df):
        """自定义 N 值。"""
        result = top_content_analysis(df, n=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_metric(self, df):
        """自定义排序指标。"""
        result = top_content_analysis(df, metric="点赞数", n=2)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# distribution_analysis
# ---------------------------------------------------------------------------

class TestDistributionAnalysis:
    """分布分析。"""

    def test_returns_str(self, df):
        result = distribution_analysis(df)
        assert isinstance(result, str)

    def test_starts_with_prefix(self, df):
        result = distribution_analysis(df)
        assert result.startswith("[分布分析")

    def test_custom_metric(self, df):
        result = distribution_analysis(df, metric="点赞数")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# compute_basic_stats（底层工具函数）
# ---------------------------------------------------------------------------

class TestComputeBasicStats:
    """基础统计工具函数。"""

    def test_returns_dict(self, df):
        result = compute_basic_stats(df, "播放量")
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, df):
        result = compute_basic_stats(df, "播放量")
        # 应包含常见统计量
        expected_keys = {"mean", "max", "min"}
        assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# run_analysis 统一调度入口
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """统一调度入口测试。"""

    @pytest.mark.parametrize("analysis_type", [
        "describe", "correlation", "trend",
        "content_type", "top", "distribution",
    ])
    def test_all_types_return_str(self, df, analysis_type):
        """所有分析类型都应返回非空字符串。"""
        result = run_analysis(df, analysis_type, params={})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_type_raises(self, df):
        """非法 analysis_type 应抛出异常。"""
        with pytest.raises((ValueError, KeyError)):
            run_analysis(df, "invalid_type", params={})

    def test_with_params(self, df):
        """带参数调用。"""
        result = run_analysis(df, "top", params={"n": 3, "metric": "播放量"})
        assert isinstance(result, str)
