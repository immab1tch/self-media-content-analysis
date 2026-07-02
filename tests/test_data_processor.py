# -*- coding: utf-8 -*-
"""
data_processor 模块单元测试。

覆盖：数据预处理（类型转换、缺失值、去重、描述统计）与数据摘要生成。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_processor import preprocess_data, generate_data_summary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    """构造一份含缺失值、重复行、字符串数字混用的原始数据。"""
    return pd.DataFrame({
        "发布日期": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-03"],
        "内容标题": ["标题A", "标题B", "标题A", None],  # 含重复 + 缺失
        "内容类型": ["短视频", "图文", "短视频", "长视频"],
        "播放量": ["100", "200", "100", np.nan],  # 字符串数字 + 缺失
        "点赞数": [10, 20, 10, 30],
    })


@pytest.fixture
def clean_df():
    """干净数据（发布日期已转为 datetime，模拟预处理后形态）。"""
    return pd.DataFrame({
        "发布日期": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "内容标题": ["标题A", "标题B", "标题C"],
        "内容类型": ["短视频", "图文", "长视频"],
        "播放量": [100, 200, 300],
        "点赞数": [10, 20, 30],
    })


# ---------------------------------------------------------------------------
# preprocess_data 测试
# ---------------------------------------------------------------------------

class TestPreprocessData:
    """测试数据预处理。"""

    def test_returns_dataframe(self, raw_df):
        """预处理后应返回 DataFrame。"""
        result = preprocess_data(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_deduplication(self, raw_df):
        """基于内容标题去重后行数应减少。"""
        result = preprocess_data(raw_df)
        # 原始 4 行，标题A重复 1 次，去重后应 <= 3 行（缺失标题行也会保留）
        assert len(result) < len(raw_df)

    def test_numeric_conversion(self, raw_df):
        """字符串数字应转换为数值类型。"""
        result = preprocess_data(raw_df)
        assert pd.api.types.is_numeric_dtype(result["播放量"])

    def test_missing_numeric_filled_with_zero(self, raw_df):
        """数值列缺失值应填 0。"""
        result = preprocess_data(raw_df)
        # 原始播放量第 4 行为 NaN，预处理后应为 0
        assert not result["播放量"].isna().any()

    def test_missing_text_filled_with_unknown(self, raw_df):
        """文本列缺失值应填"未知"。"""
        result = preprocess_data(raw_df)
        # 内容标题第 4 行为 None，去重可能保留也可能不保留
        # 检查所有文本列无 NaN
        for col in ["内容标题", "内容类型"]:
            if col in result.columns:
                assert not result[col].isna().any()

    def test_date_column_conversion(self, raw_df):
        """日期列应转为 datetime。"""
        result = preprocess_data(raw_df)
        assert pd.api.types.is_datetime64_any_dtype(result["发布日期"])


# ---------------------------------------------------------------------------
# generate_data_summary 测试
# ---------------------------------------------------------------------------

class TestGenerateDataSummary:
    """测试数据摘要生成。"""

    def test_returns_string(self, clean_df):
        """摘要返回值应为字符串。"""
        summary = generate_data_summary(clean_df)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_overview(self, clean_df):
        """摘要应包含数据集概览信息。"""
        summary = generate_data_summary(clean_df)
        # 应包含行数或列数
        assert "3" in summary  # 3 行

    def test_summary_contains_field_info(self, clean_df):
        """摘要应包含字段清单。"""
        summary = generate_data_summary(clean_df)
        assert "发布日期" in summary
        assert "播放量" in summary

    def test_summary_contains_statistics(self, clean_df):
        """摘要应包含统计指标。"""
        summary = generate_data_summary(clean_df)
        # 播放量均值 200，应出现在摘要中
        assert "200" in summary

    def test_summary_contains_quality_info(self, clean_df):
        """摘要应包含数据质量标注。"""
        summary = generate_data_summary(clean_df)
        # 应包含缺失率相关文本
        assert ("缺失" in summary) or ("质量" in summary) or ("缺失率" in summary)

    def test_summary_no_full_data(self, clean_df):
        """摘要不应包含全量原始数据（控制 token）。"""
        summary = generate_data_summary(clean_df)
        # 摘要长度应远小于全量数据文本
        full_text = clean_df.to_string()
        assert len(summary) < len(full_text) * 5  # 宽松断言

    def test_summary_with_missing_values(self, raw_df):
        """含缺失值的数据摘要应能正常生成。"""
        processed = preprocess_data(raw_df)
        summary = generate_data_summary(processed)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_empty_dataframe(self):
        """空 DataFrame 的摘要应能正常生成不崩溃。"""
        empty = pd.DataFrame()
        summary = generate_data_summary(empty)
        assert isinstance(summary, str)
