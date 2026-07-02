# -*- coding: utf-8 -*-
"""
对话上下文管理模块。

维护用户与 AI 助手之间的多轮对话历史，
支持上下文窗口管理与历史裁剪。
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextManager:
    """对话上下文管理器。"""

    def __init__(self, max_turns: int = 10) -> None:
        """
        初始化上下文管理器。

        参数:
            max_turns: 保留的最大对话轮数。
        """
        self._max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条对话消息。

        参数:
            role: 消息角色，如 user 或 assistant。
            content: 消息内容。
        """
        raise NotImplementedError

    def get_history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史。

        返回:
            对话历史消息列表。
        """
        return list(self._history)

    def clear(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def trim_history(self) -> None:
        """
        裁剪超出窗口的对话历史，保留最近 max_turns 轮。
        """
        raise NotImplementedError
