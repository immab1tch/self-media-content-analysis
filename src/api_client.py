# -*- coding: utf-8 -*-
"""
大模型 API 调用层模块。

封装对 Q Cloud、Workbody、DeepSeek 等大模型 API 的
HTTP 请求，统一处理超时、鉴权失败与返回格式异常。
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class APIClient:
    """大模型 API 客户端。"""

    def __init__(self, provider: Optional[str] = None) -> None:
        """
        初始化 API 客户端。

        参数:
            provider: 模型提供商标识，默认从环境变量读取。
        """
        self._provider = provider or os.environ.get("DEFAULT_LLM_PROVIDER", "deepseek")
        self._api_key = self._load_api_key()
        self._base_url = self._load_base_url()
        self._timeout = int(os.environ.get("API_TIMEOUT", "30"))

    def _load_api_key(self) -> str:
        """
        从环境变量加载 API 密钥。

        返回:
            API 密钥字符串。
        """
        raise NotImplementedError

    def _load_base_url(self) -> str:
        """
        从环境变量加载 API 基础 URL。

        返回:
            API 基础 URL 字符串。
        """
        raise NotImplementedError

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """
        调用大模型聊天补全接口。

        参数:
            messages: 消息列表，符合 OpenAI 格式。
            model: 模型名称，默认从环境变量读取。

        返回:
            模型生成的文本回复。

        异常:
            requests.Timeout: 请求超时。
            requests.HTTPError: 鉴权失败或 HTTP 错误。
            ValueError: 返回格式异常。
        """
        raise NotImplementedError
