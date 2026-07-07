# -*- coding: utf-8 -*-
"""
prompt_engine 模块单元测试。

覆盖：system/user/followup prompt 构建、响应解析。
涉及大模型 API 的部分使用 unittest.mock 模拟，不真实调用 API。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prompt_engine import (  # noqa: E402
    ANALYSIS_CAPABILITIES,
    VALID_ANALYSIS_TYPES,
    build_followup_prompt,
    build_system_prompt,
    build_user_prompt,
    parse_response,
)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    """系统提示词构建。"""

    def test_returns_str(self):
        result = build_system_prompt("测试摘要")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_role_setting(self):
        """应包含角色设定。"""
        result = build_system_prompt("摘要")
        assert "数析" in result
        assert "自媒体" in result

    def test_contains_data_summary(self):
        """应包含传入的数据摘要。"""
        summary = "这是一段测试数据摘要内容"
        result = build_system_prompt(summary)
        assert summary in result

    def test_contains_capabilities(self):
        """应列出分析能力清单。"""
        result = build_system_prompt("摘要")
        assert "describe" in result
        assert "trend" in result
        assert "correlation" in result

    def test_contains_output_rules(self):
        """应包含 JSON 输出规则。"""
        result = build_system_prompt("摘要")
        assert "conclusion" in result
        assert "analysis_type" in result

    def test_contains_prohibition_rules(self):
        """应包含禁止规则。"""
        result = build_system_prompt("摘要")
        assert "编造" in result
        assert "禁止" in result

    def test_no_full_data_keyword(self):
        """system prompt 不应包含"全量数据"传输的暗示。"""
        summary = "摘要内容"
        result = build_system_prompt(summary)
        # 禁止规则会提到"不得编造数据"，但不应有"发送全量数据"的指令
        assert "发送全量原始数据" not in result


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    """用户问题提示词构建。"""

    def test_returns_str(self):
        result = build_user_prompt("播放量趋势如何？")
        assert isinstance(result, str)

    def test_contains_question(self):
        question = "哪种内容类型表现最好？"
        result = build_user_prompt(question)
        assert question in result

    def test_empty_question_raises(self):
        """空问题应抛出 ValueError。"""
        with pytest.raises(ValueError):
            build_user_prompt("")

    def test_whitespace_only_raises(self):
        """纯空白问题应抛出 ValueError。"""
        with pytest.raises(ValueError):
            build_user_prompt("   ")


# ---------------------------------------------------------------------------
# build_followup_prompt
# ---------------------------------------------------------------------------

class TestBuildFollowupPrompt:
    """追问提示词构建。"""

    def test_returns_str(self):
        history = [{"role": "user", "content": "问题1"}]
        result = build_followup_prompt("追问", history)
        assert isinstance(result, str)

    def test_contains_question(self):
        history = [{"role": "user", "content": "之前的问题"}]
        result = build_followup_prompt("新的追问", history)
        assert "新的追问" in result

    def test_contains_history(self):
        history = [
            {"role": "user", "content": "历史问题A"},
            {"role": "assistant", "content": "历史回答A"},
        ]
        result = build_followup_prompt("追问", history)
        assert "历史问题A" in result
        assert "历史回答A" in result

    def test_empty_history(self):
        """空历史列表应能正常构建。"""
        result = build_followup_prompt("问题", [])
        assert isinstance(result, str)
        assert "问题" in result

    def test_empty_question_raises(self):
        """空追问问题应抛出 ValueError。"""
        with pytest.raises(ValueError):
            build_followup_prompt("", [{"role": "user", "content": "x"}])

    def test_invalid_history_raises(self):
        """history 非列表应抛出 ValueError。"""
        with pytest.raises(ValueError):
            build_followup_prompt("问题", "not a list")  # type: ignore


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    """响应解析测试。"""

    def test_pure_json(self):
        """纯 JSON 字符串。"""
        resp = '{"conclusion": "测试结论", "analysis_type": "trend"}'
        result = parse_response(resp)
        assert result["conclusion"] == "测试结论"
        assert result["analysis_type"] == "trend"

    def test_markdown_code_block(self):
        """markdown 代码块包裹的 JSON。"""
        resp = '```json\n{"conclusion": "结论", "analysis_type": "correlation"}\n```'
        result = parse_response(resp)
        assert result["conclusion"] == "结论"
        assert result["analysis_type"] == "correlation"

    def test_plain_code_block(self):
        """无 json 标记的代码块。"""
        resp = '```\n{"conclusion": "结论B", "analysis_type": "top"}\n```'
        result = parse_response(resp)
        assert result["conclusion"] == "结论B"
        assert result["analysis_type"] == "top"

    def test_null_analysis_type(self):
        """analysis_type 为 null。"""
        resp = '{"conclusion": "你好", "analysis_type": null}'
        result = parse_response(resp)
        assert result["conclusion"] == "你好"
        assert result["analysis_type"] is None

    def test_string_null_analysis_type(self):
        """analysis_type 为字符串 "null"。"""
        resp = '{"conclusion": "你好", "analysis_type": "null"}'
        result = parse_response(resp)
        assert result["analysis_type"] is None

    def test_invalid_analysis_type(self):
        """非法 analysis_type 应置为 None。"""
        resp = '{"conclusion": "测试", "analysis_type": "unknown_type"}'
        result = parse_response(resp)
        assert result["conclusion"] == "测试"
        assert result["analysis_type"] is None

    def test_embedded_json(self):
        """文本中嵌入 JSON。"""
        resp = '以下是结果：{"conclusion": "Top1", "analysis_type": "top"} 希望有帮助'
        result = parse_response(resp)
        assert result["conclusion"] == "Top1"
        assert result["analysis_type"] == "top"

    def test_plain_text_fallback(self):
        """无法解析的纯文本应降级返回原文。"""
        resp = "这是一段非 JSON 文本。"
        result = parse_response(resp)
        assert result["conclusion"] == resp
        assert result["analysis_type"] is None

    def test_empty_string(self):
        """空字符串。"""
        result = parse_response("")
        assert result["conclusion"] == ""
        assert result["analysis_type"] is None

    def test_whitespace_only(self):
        """纯空白。"""
        result = parse_response("   \n  ")
        assert result["conclusion"] == ""
        assert result["analysis_type"] is None

    def test_missing_conclusion_field(self):
        """缺少 conclusion 字段时应降级返回原文。"""
        resp = '{"analysis_type": "trend"}'
        result = parse_response(resp)
        assert result["conclusion"] == resp
        assert result["analysis_type"] == "trend"

    def test_all_valid_types(self):
        """所有合法 analysis_type 都能正确解析。"""
        for atype in VALID_ANALYSIS_TYPES:
            resp = f'{{"conclusion": "x", "analysis_type": "{atype}"}}'
            result = parse_response(resp)
            assert result["analysis_type"] == atype, f"类型 {atype} 解析失败"


# ---------------------------------------------------------------------------
# 常量校验
# ---------------------------------------------------------------------------

class TestConstants:
    """能力清单常量校验。"""

    def test_capabilities_not_empty(self):
        assert len(ANALYSIS_CAPABILITIES) > 0

    def test_valid_types_not_empty(self):
        assert len(VALID_ANALYSIS_TYPES) > 0

    def test_capabilities_have_required_fields(self):
        """每个能力项应包含 analysis_type / function / description。"""
        for cap in ANALYSIS_CAPABILITIES:
            assert "analysis_type" in cap
            assert "function" in cap
            assert "description" in cap

    def test_capabilities_match_valid_types(self):
        """能力清单的 analysis_type 应与 VALID_ANALYSIS_TYPES 一致。"""
        cap_types = {cap["analysis_type"] for cap in ANALYSIS_CAPABILITIES}
        assert cap_types == VALID_ANALYSIS_TYPES


# ---------------------------------------------------------------------------
# 端到端 mock 测试：模拟大模型 API 调用流程
# ---------------------------------------------------------------------------

class TestEndToEndWithMock:
    """
    使用 mock 模拟大模型 API 调用，验证 prompt 构建 → mock 返回 → 解析的完整流程。

    不真实调用任何 API。
    """

    def test_full_pipeline_with_mock_response(self):
        """模拟大模型返回标准 JSON，验证完整流程。"""
        # 1. 构建 prompt
        data_summary = "数据集概览：行数=30，列数=9"
        system_prompt = build_system_prompt(data_summary)
        user_prompt = build_user_prompt("播放量趋势如何？")

        assert "数据集概览" in system_prompt
        assert "播放量趋势如何" in user_prompt

        # 2. 模拟大模型返回（不真实调用 API）
        mock_response = '{"conclusion": "播放量整体呈上升趋势，从1万增长到3万", "analysis_type": "trend"}'

        # 3. 解析
        parsed = parse_response(mock_response)
        assert parsed["conclusion"] == "播放量整体呈上升趋势，从1万增长到3万"
        assert parsed["analysis_type"] == "trend"

    def test_mock_llm_client_chat_with_retry(self):
        """
        使用 unittest.mock.patch 模拟 LLMClient.chat_with_retry，
        验证 prompt_engine 产出的 messages 能被正确消费并解析。
        """
        # 模拟 LLMClient 的 chat_with_retry 方法返回值
        mock_llm_response = (
            '{"conclusion": "短视频类型平均播放量最高，为15678", '
            '"analysis_type": "content_type"}'
        )

        # patch LLMClient 类（在 api_client 模块中定义）
        with patch("api_client.LLMClient") as MockLLMClient:
            # 配置 mock 实例的 chat_with_retry 返回值
            mock_instance = MockLLMClient.return_value
            mock_instance.chat_with_retry.return_value = mock_llm_response
            mock_instance.is_degraded = False

            # 模拟使用流程
            from api_client import LLMClient

            client = LLMClient()  # 此时返回 mock 实例
            data_summary = "测试摘要"
            messages = [
                {"role": "system", "content": build_system_prompt(data_summary)},
                {"role": "user", "content": build_user_prompt("哪种类型最好？")},
            ]

            # 调用（mock 不会真实请求网络）
            response = client.chat_with_retry(messages=messages, max_retries=2)

            # 验证 mock 被正确调用
            mock_instance.chat_with_retry.assert_called_once()
            call_kwargs = mock_instance.chat_with_retry.call_args
            assert call_kwargs.kwargs["max_retries"] == 2

            # 解析响应
            parsed = parse_response(response)
            assert parsed["conclusion"] == "短视频类型平均播放量最高，为15678"
            assert parsed["analysis_type"] == "content_type"

    def test_mock_api_failure_fallback(self):
        """模拟 API 调用失败，验证降级处理流程。"""
        with patch("api_client.LLMClient") as MockLLMClient:
            mock_instance = MockLLMClient.return_value
            # 模拟 API 调用抛出异常
            mock_instance.chat_with_retry.side_effect = ConnectionError("网络不可用")

            from api_client import LLMClient

            client = LLMClient()
            messages = [
                {"role": "system", "content": build_system_prompt("摘要")},
                {"role": "user", "content": build_user_prompt("问题")},
            ]

            # API 调用应抛出异常（由上层降级处理）
            with pytest.raises(ConnectionError):
                client.chat_with_retry(messages=messages)

    def test_mock_malformed_response(self):
        """模拟大模型返回格式异常，验证降级解析。"""
        # 大模型返回非 JSON 文本
        mock_response = "抱歉，我无法理解您的问题。"

        parsed = parse_response(mock_response)
        # 降级：conclusion 为原文，analysis_type 为 None
        assert parsed["conclusion"] == mock_response
        assert parsed["analysis_type"] is None
