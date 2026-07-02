# -*- coding: utf-8 -*-
"""
可视化图表生成模块。

基于 matplotlib 与 plotly 生成自媒体数据分析图表，
图表标题与轴标签均使用中文。
"""

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "柱状图",
) -> Any:
    """
    生成柱状图。

    参数:
        df: 数据 DataFrame。
        x_column: X 轴列名。
        y_column: Y 轴列名。
        title: 图表标题。

    返回:
        图表对象（matplotlib Figure 或 plotly Figure）。
    """
    raise NotImplementedError


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "折线图",
) -> Any:
    """
    生成折线图。

    参数:
        df: 数据 DataFrame。
        x_column: X 轴列名。
        y_column: Y 轴列名。
        title: 图表标题。

    返回:
        图表对象（matplotlib Figure 或 plotly Figure）。
    """
    raise NotImplementedError


def create_chart(
    chart_type: str,
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> Any:
    """
    根据图表类型生成对应可视化。

    参数:
        chart_type: 图表类型标识，如 bar、line、pie。
        df: 数据 DataFrame。
        params: 图表参数字典。

    返回:
        图表对象。
    """
    raise NotImplementedError
