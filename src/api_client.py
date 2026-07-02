# -*- coding: utf-8 -*-
"""
大模型 API 调用层模块。

封装对 OpenAI 兼容大模型 API 的 HTTP 请求，
统一处理超时、鉴权失败、返回格式异常与自动重试。

本模块仅负责"发请求、收响应、处理异常"，
Prompt 模板与业务逻辑由其他模块负责。
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class LLMClient:
    """大模型 API 客户端（兼容 OpenAI Chat Completions 协议）。"""

    def __init__(self) -> None:
        """
        初始化大模型客户端，从环境变量读取配置。

        环境变量：
            LLM_API_KEY: API 密钥（必填）。
            LLM_API_URL: 完整的 Chat Completions 端点 URL。
            LLM_MODEL: 模型名称，默认 deepseek-chat。
            LLM_TIMEOUT: 请求超时秒数，默认 30。

        异常:
            ValueError: API_KEY 未配置时抛出。
        """
        self._api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError(
                "未配置大模型 API 密钥。请检查 .env 文件中是否设置了 LLM_API_KEY。"
            )

        self._api_url = os.environ.get(
            "LLM_API_URL",
            "https://api.deepseek.com/v1/chat/completions",
        ).strip()

        self._model = os.environ.get("LLM_MODEL", "deepseek-chat").strip()

        timeout_str = os.environ.get("LLM_TIMEOUT", "30").strip()
        try:
            self._timeout = float(timeout_str)
        except (ValueError, TypeError):
            logger.warning("LLM_TIMEOUT 配置无效 '%s'，使用默认值 30 秒。", timeout_str)
            self._timeout = 30.0

        logger.info(
            "LLMClient 初始化完成：model=%s，url=%s，timeout=%.1fs，key=%s",
            self._model,
            self._api_url,
            self._timeout,
            self._masked_key(self._api_key),
        )

    @staticmethod
    def _masked_key(api_key: str) -> str:
        """
        对 API 密钥做脱敏处理，仅保留首尾各 3 个字符用于日志排查。

        参数:
            api_key: 原始 API 密钥。

        返回:
            脱敏后的密钥字符串。
        """
        if not api_key:
            return "(空)"
        if len(api_key) <= 8:
            return "***"
        return f"{api_key[:3]}...{api_key[-3:]}"

    def _build_headers(self) -> Dict[str, str]:
        """
        构造请求头。

        返回:
            包含 Authorization 与 Content-Type 的请求头字典。
        """
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        """
        从 API 响应 JSON 中提取模型回复文本。

        参数:
            response_json: API 返回的 JSON 对象。

        返回:
            模型回复的文本内容。

        异常:
            ValueError: 返回格式不符合预期时抛出。
        """
        try:
            choices = response_json["choices"]
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError("choices 字段为空或不是列表")
            message = choices[0]["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError(f"content 不是字符串类型：{type(content).__name__}")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("API 返回格式异常：%s", exc)
            raise ValueError(
                f"大模型 API 返回格式异常，无法提取回复内容。原因：{exc}"
            ) from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        发送单次聊天补全请求到大模型 API。

        参数:
            messages: 消息列表，格式为：
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
            temperature: 采样温度，默认 0.7。
            max_tokens: 最大生成 token 数，默认 2000。

        返回:
            模型生成的回复文本字符串。

        异常:
            ValueError: 参数非法或返回格式异常。
            requests.exceptions.Timeout: 请求超时。
            requests.exceptions.HTTPError: HTTP 状态码异常（含鉴权失败）。
            requests.exceptions.RequestException: 其他网络错误。
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages 参数必须为非空列表。")

        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    "messages 列表中每条消息必须包含 'role' 和 'content' 字段。"
                )

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(
            "发送大模型请求：url=%s，model=%s，messages=%d 条，temperature=%.2f，max_tokens=%d",
            self._api_url, self._model, len(messages), temperature, max_tokens,
        )

        try:
            response = requests.post(
                self._api_url,
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            logger.error("大模型 API 请求超时（%.1f 秒）：%s", self._timeout, self._api_url)
            raise
        except requests.exceptions.ConnectionError as exc:
            logger.error("大模型 API 连接失败：%s", exc)
            raise

        # 处理 HTTP 错误
        if response.status_code == 401:
            logger.error("鉴权失败（401）：请检查 LLM_API_KEY 是否正确。")
            raise requests.exceptions.HTTPError(
                "鉴权失败（HTTP 401）：API 密钥无效或已过期。请检查 .env 文件中的 LLM_API_KEY。",
                response=response,
            )
        if response.status_code == 429:
            logger.warning("请求被限流（429）：请稍后重试。")
            raise requests.exceptions.HTTPError(
                "请求被限流（HTTP 429）：调用频率超限，请稍后重试。",
                response=response,
            )
        if response.status_code >= 500:
            logger.error("服务器错误（%d）：%s", response.status_code, response.text[:200])
            raise requests.exceptions.HTTPError(
                f"大模型服务端错误（HTTP {response.status_code}），请稍后重试。",
                response=response,
            )
        if response.status_code >= 400:
            logger.error(
                "请求失败（%d）：%s",
                response.status_code, response.text[:200],
            )
            raise requests.exceptions.HTTPError(
                f"请求失败（HTTP {response.status_code}）：{response.text[:100]}",
                response=response,
            )

        # 解析 JSON 响应
        try:
            response_json = response.json()
        except ValueError as exc:
            logger.error("API 响应不是合法 JSON：%s", response.text[:200])
            raise ValueError(
                f"大模型 API 返回内容不是合法 JSON。状态码：{response.status_code}"
            ) from exc

        return self._extract_content(response_json)

    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        带重试的聊天补全请求。

        以下情况自动重试（间隔递增）：
        - 网络超时（Timeout）
        - 429 限流
        - 5xx 服务器错误
        - 网络连接异常（ConnectionError）

        其他异常（如 401 鉴权失败、400 参数错误）立即抛出，不重试。

        参数:
            messages: 消息列表，格式同 chat()。
            max_retries: 最大重试次数，默认 3 次（加上首次共 max_retries+1 次请求）。
            initial_delay: 首次重试等待秒数，默认 1 秒。
            backoff_factor: 退避倍数，默认 2（每次等待时间翻倍）。
            temperature: 采样温度，默认 0.7。
            max_tokens: 最大生成 token 数，默认 2000。

        返回:
            模型生成的回复文本字符串。

        异常:
            最终一次重试仍失败时，抛出最后一次的异常。
        """
        if max_retries < 0:
            max_retries = 0

        last_exception: Optional[Exception] = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_exception = exc
                if attempt < max_retries:
                    logger.warning(
                        "第 %d/%d 次请求失败（网络问题），%.1f 秒后重试：%s",
                        attempt + 1, max_retries + 1, delay, exc,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(
                        "已达最大重试次数（%d 次），网络请求仍失败：%s",
                        max_retries + 1, exc,
                    )
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                # 仅对 429 和 5xx 重试
                if status_code == 429 or status_code >= 500:
                    last_exception = exc
                    if attempt < max_retries:
                        logger.warning(
                            "第 %d/%d 次请求失败（HTTP %d），%.1f 秒后重试",
                            attempt + 1, max_retries + 1, status_code, delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "已达最大重试次数（%d 次），HTTP %d 仍失败",
                            max_retries + 1, status_code,
                        )
                else:
                    # 401、400 等不重试
                    raise

        # 所有重试用完，抛出最后一次异常
        if last_exception is not None:
            raise last_exception

        # 理论上不会到达这里
        raise RuntimeError("chat_with_retry 异常终止：无可用异常信息。")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("LLMClient 单元测试")
    print("=" * 60)

    # 1. 测试未配置密钥的异常
    print("\n1. 测试无 API_KEY 时抛出 ValueError ...", end=" ")
    original_key = os.environ.pop("LLM_API_KEY", None)
    try:
        _ = LLMClient()
        print("FAIL（应抛出异常）")
    except ValueError as exc:
        if "LLM_API_KEY" in str(exc):
            print("OK")
        else:
            print(f"FAIL（异常信息不包含 LLM_API_KEY：{exc}）")
    finally:
        if original_key is not None:
            os.environ["LLM_API_KEY"] = original_key

    # 2. 测试 messages 参数校验
    print("2. 测试 messages 非法参数校验 ...", end=" ")
    os.environ["LLM_API_KEY"] = "sk-test-fake-key-for-unit-test"
    client = LLMClient()
    try:
        client.chat([])
        print("FAIL（应抛出 ValueError）")
    except ValueError:
        print("OK")

    try:
        client.chat([{"role": "user"}])  # 缺少 content
        print("FAIL（应抛出 ValueError）")
    except ValueError:
        print("   格式校验 OK")

    # 3. 测试密钥脱敏
    print("3. 测试密钥脱敏函数 ...", end=" ")
    masked = LLMClient._masked_key("sk-1234567890abcdef")
    if masked == "sk-...def":
        print("OK")
    else:
        print(f"FAIL（{masked}）")

    # 4. 测试请求头构造
    print("4. 测试请求头包含 Authorization ...", end=" ")
    headers = client._build_headers()
    if "Bearer sk-test-fake-key-for-unit-test" in headers.get("Authorization", ""):
        print("OK")
    else:
        print("FAIL")

    # 5. 测试响应解析（正常）
    print("5. 测试正常响应解析 ...", end=" ")
    mock_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "  你好，这是测试回复。  "}}
        ]
    }
    result = client._extract_content(mock_response)
    if result == "你好，这是测试回复。":
        print("OK")
    else:
        print(f"FAIL（{result}）")

    # 6. 测试响应解析（异常格式）
    print("6. 测试异常响应格式抛出 ValueError ...", end=" ")
    try:
        client._extract_content({"error": "something"})
        print("FAIL（应抛出 ValueError）")
    except ValueError:
        print("OK")

    # 7. 测试超时重试逻辑（用 mock 方式验证退避次数）
    print("7. 测试重试次数与退避间隔 ...", end=" ")
    import unittest.mock as mock

    call_counter = [0]

    def fake_chat(*args, **kwargs):
        call_counter[0] += 1
        raise requests.exceptions.Timeout("模拟超时")

    with mock.patch.object(client, "chat", side_effect=fake_chat):
        with mock.patch("time.sleep") as mock_sleep:
            try:
                client.chat_with_retry(
                    [{"role": "user", "content": "hi"}],
                    max_retries=2,
                    initial_delay=0.5,
                    backoff_factor=2,
                )
            except requests.exceptions.Timeout:
                pass

    if call_counter[0] == 3 and mock_sleep.call_count == 2:
        print("OK（3 次请求 + 2 次等待）")
    else:
        print(f"FAIL（call_count={call_counter[0]}, sleep_count={mock_sleep.call_count}）")

    # 清理
    if original_key is None:
        os.environ.pop("LLM_API_KEY", None)
    else:
        os.environ["LLM_API_KEY"] = original_key

    print("\n测试完成。")
