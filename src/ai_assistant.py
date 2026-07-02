# -*- coding: utf-8 -*-
"""
AI 自然语言数据分析助手核心模块。

接收用户自然语言提问，结合数据摘要与统计函数，
调用大模型 API 生成分析结论与可视化建议。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AIAssistant:
    """AI 驱动的自然语言数据分析助手。"""

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        初始化 AI 助手。

        参数:
            df: 当前会话关联的数据 DataFrame。
        """
        self._df = df

    def ask(self, question: str) -> Dict[str, Any]:
        """
        处理用户自然语言提问并返回分析结果。

        参数:
            question: 用户的自然语言问题。

        返回:
            包含 AI 分析结论、统计依据及图表建议的字典。
        """
        raise NotImplementedError

    def set_data(self, df: pd.DataFrame) -> None:
        """
        更新当前会话关联的数据。

        参数:
            df: 新的数据 DataFrame。
        """
        self._df = df
