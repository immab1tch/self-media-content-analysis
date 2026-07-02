# -*- coding: utf-8 -*-
"""
Prompt 模板引擎模块。

管理与大模型交互的 Prompt 模板，确保只向模型发送
数据摘要而非全量原始数据。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_system_prompt() -> str:
    """
    构建系统级 Prompt。

    返回:
        系统 Prompt 字符串。
    """
    raise NotImplementedError


def build_analysis_prompt(
    question: str,
    data_summary: Dict[str, Any],
    stats_context: str = "",
) -> str:
    """
    构建数据分析问答 Prompt。

    参数:
        question: 用户提问。
        data_summary: 数据摘要字典。
        stats_context: 底层统计依据文本。

    返回:
        完整的用户 Prompt 字符串。
    """
    raise NotImplementedError


def build_followup_prompt(
    question: str,
    history: List[Dict[str, str]],
    data_summary: Dict[str, Any],
) -> str:
    """
    构建多轮追问 Prompt。

    参数:
        question: 当前用户提问。
        history: 对话历史列表。
        data_summary: 数据摘要字典。

    返回:
        包含上下文的 Prompt 字符串。
    """
    raise NotImplementedError
