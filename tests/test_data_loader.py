# -*- coding: utf-8 -*-
"""
data_loader 模块单元测试。

覆盖：CSV 加载、Excel 加载、异常文件处理（不存在/不支持格式/空文件）。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# 将 src 目录加入 sys.path 以便导入模块
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import load_data, SUPPORTED_EXTENSIONS  # noqa: E402

# 示例数据目录
_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_path():
    """示例 CSV 文件路径。"""
    return str(_SAMPLE_DIR / "sample_content.csv")


@pytest.fixture
def sample_xlsx_path():
    """示例 Excel 文件路径。"""
    return str(_SAMPLE_DIR / "sample_content.xlsx")


@pytest.fixture
def tmp_csv(tmp_path):
    """在临时目录创建一个简单 CSV 文件。"""
    p = tmp_path / "test.csv"
    df = pd.DataFrame({
        "发布日期": ["2025-01-01", "2025-01-02"],
        "内容标题": ["标题A", "标题B"],
        "播放量": [100, 200],
    })
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


@pytest.fixture
def tmp_xlsx(tmp_path):
    """在临时目录创建一个简单 Excel 文件。"""
    p = tmp_path / "test.xlsx"
    df = pd.DataFrame({
        "发布日期": ["2025-01-01", "2025-01-02"],
        "内容标题": ["标题A", "标题B"],
        "播放量": [100, 200],
    })
    df.to_excel(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# 正常加载测试
# ---------------------------------------------------------------------------

class TestLoadDataSuccess:
    """测试正常数据加载。"""

    def test_load_csv_returns_dataframe(self, sample_csv_path):
        """CSV 加载应返回 DataFrame。"""
        df = load_data(sample_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_load_csv_has_data(self, sample_csv_path):
        """CSV 加载后应有数据行。"""
        df = load_data(sample_csv_path)
        assert len(df) > 0
        assert len(df.columns) == 9

    def test_load_csv_columns(self, sample_csv_path):
        """CSV 列名应与预期一致。"""
        df = load_data(sample_csv_path)
        expected_cols = {
            "发布日期", "内容标题", "内容类型", "播放量",
            "点赞数", "评论数", "转发数", "收藏数", "粉丝增量",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_load_xlsx_returns_dataframe(self, sample_xlsx_path):
        """Excel 加载应返回 DataFrame。"""
        df = load_data(sample_xlsx_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_tmp_csv(self, tmp_csv):
        """临时 CSV 文件加载。"""
        df = load_data(tmp_csv)
        assert len(df) == 2
        assert list(df.columns) == ["发布日期", "内容标题", "播放量"]

    def test_load_tmp_xlsx(self, tmp_xlsx):
        """临时 Excel 文件加载。"""
        df = load_data(tmp_xlsx)
        assert len(df) == 2
        assert "播放量" in df.columns


# ---------------------------------------------------------------------------
# 异常处理测试
# ---------------------------------------------------------------------------

class TestLoadDataExceptions:
    """测试异常文件处理。"""

    def test_file_not_found(self, tmp_path):
        """文件不存在应抛出 FileNotFoundError。"""
        nonexistent = str(tmp_path / "not_exist.csv")
        with pytest.raises(FileNotFoundError):
            load_data(nonexistent)

    def test_unsupported_format(self, tmp_path):
        """不支持的格式应抛出 ValueError。"""
        p = tmp_path / "data.txt"
        p.write_text("some content", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_data(str(p))

    def test_empty_csv(self, tmp_path):
        """空 CSV 文件加载应返回空 DataFrame（不抛异常）。"""
        p = tmp_path / "empty.csv"
        p.write_text("发布日期,内容标题\n", encoding="utf-8-sig")
        df = load_data(str(p))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_supported_extensions_constant(self):
        """SUPPORTED_EXTENSIONS 应包含 csv 和 xlsx。"""
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# 编码测试
# ---------------------------------------------------------------------------

class TestEncodingHandling:
    """测试 CSV 编码处理。"""

    def test_gbk_csv(self, tmp_path):
        """GBK 编码的 CSV 应能正常加载。"""
        p = tmp_path / "gbk.csv"
        df = pd.DataFrame({"标题": ["中文标题A", "中文标题B"], "数值": [1, 2]})
        df.to_csv(p, index=False, encoding="gbk")
        loaded = load_data(str(p))
        assert len(loaded) == 2
        assert "标题" in loaded.columns

    def test_utf8_bom_csv(self, tmp_path):
        """UTF-8 BOM 编码的 CSV 应能正常加载。"""
        p = tmp_path / "bom.csv"
        df = pd.DataFrame({"标题": ["测试A", "测试B"]})
        df.to_csv(p, index=False, encoding="utf-8-sig")
        loaded = load_data(str(p))
        assert len(loaded) == 2
        # BOM 不应出现在列名中
        assert "标题" in loaded.columns
        assert not any("\ufeff" in col for col in loaded.columns)
