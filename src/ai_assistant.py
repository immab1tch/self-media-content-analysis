# -*- coding: utf-8 -*-
"""
AI 自然语言数据分析助手核心编排模块。

整合 api_client、prompt_engine、context_manager、analyzer、visualizer，
将用户自然语言提问转化为「结论文字 + 可视化图表」的联动输出。

设计原则：
1. 此模块不直接处理用户界面，只返回结构化数据。
2. 图表对象返回后由调用方（app.py）决定如何展示。
3. API 调用失败时降级为本地统计分析，保证可用性。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

# 同目录模块导入
from analyzer import run_analysis
from api_client import LLMClient
from context_manager import ConversationContext
from data_processor import generate_data_summary
from prompt_engine import (
    build_followup_prompt,
    build_system_prompt,
    build_user_prompt,
    parse_response,
)
from visualizer import create_chart

logger = logging.getLogger(__name__)


# analysis_type → 是否有对应图表
_CHART_AVAILABLE = {
    "describe": False,  # 描述性统计无对应图表
    "correlation": True,
    "trend": True,
    "content_type": True,
    "top": True,
    "distribution": True,
}

# 降级时使用的 analysis_type
_FALLBACK_ANALYSIS_TYPE = "describe"


class AIAssistant:
    """AI 驱动的自然语言数据分析助手。"""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        初始化 AI 助手。

        参数:
            df: 待分析的数据 DataFrame。
        """
        self._df = df
        self._context = ConversationContext(max_history=3)

        # 生成数据摘要（供 system prompt 使用，不发全量数据）
        self._data_summary = generate_data_summary(df)
        self._system_prompt = build_system_prompt(self._data_summary)
        logger.info("AI 助手初始化完成，数据摘要长度=%d 字符", len(self._data_summary))

        # 初始化 LLM 客户端（API_KEY 未配置时进入降级模式）
        self._llm_client: Optional[LLMClient] = None
        self._degraded_mode = False
        try:
            self._llm_client = LLMClient()
        except ValueError as exc:
            logger.warning("LLM 客户端初始化失败，进入降级模式：%s", exc)
            self._degraded_mode = True

    @property
    def is_degraded(self) -> bool:
        """返回是否处于降级模式（API 不可用）。"""
        return self._degraded_mode

    @property
    def data_summary(self) -> str:
        """返回当前数据摘要文本。"""
        return self._data_summary

    def ask(self, question: str) -> Dict[str, Any]:
        """
        处理用户自然语言提问并返回分析结果（单轮问答）。

        流程：
        a. 构建 system prompt + user prompt
        b. 调用 LLMClient.chat_with_retry 获取回复
        c. 通过 prompt_engine.parse_response 解析
        d. 若解析出 analysis_type，调用 analyzer 获取统计结果
        e. 若解析出 analysis_type，调用 visualizer 生成图表
        f. 返回 {conclusion, chart, analysis_type}
        g. 将本轮对话存入 ConversationContext

        参数:
            question: 用户的自然语言问题。

        返回:
            包含以下字段的字典：
            - conclusion: 分析结论文字
            - chart: matplotlib/plotly Figure 对象，或 None
            - analysis_type: 分析类型标识字符串，或 None
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空。")

        user_prompt = build_user_prompt(question)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self._run_pipeline(question, messages)

    def ask_followup(self, question: str) -> Dict[str, Any]:
        """
        处理用户追问，使用历史上下文。

        流程同 ask，区别在于使用 build_followup_prompt 拼接历史对话。

        参数:
            question: 当前追问问题。

        返回:
            同 ask 方法的返回结构。
        """
        if not question or not question.strip():
            raise ValueError("追问问题不能为空。")

        history = self._context.get_history()
        followup_prompt = build_followup_prompt(question, history)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": followup_prompt},
        ]

        return self._run_pipeline(question, messages)

    def get_history(self) -> List[Dict[str, str]]:
        """
        返回对话历史。

        返回:
            对话历史列表，每项格式 {"role":..., "content":...}。
        """
        return self._context.get_history()

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._context.clear()

    def _run_pipeline(
        self,
        question: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        执行问答核心流水线：调用模型 → 解析 → 触发分析 → 生成图表 → 存历史。

        参数:
            question: 原始用户问题（用于历史记录）。
            messages: 发送给大模型的完整消息列表。

        返回:
            包含 conclusion / chart / analysis_type 的结果字典。
        """
        # 降级模式：直接走本地统计分析
        if self._degraded_mode or self._llm_client is None:
            logger.info("降级模式：返回本地统计分析结果。")
            return self._fallback_response(question, reason="AI 服务未配置")

        # 调用大模型
        try:
            raw_response = self._llm_client.chat_with_retry(
                messages=messages,
                max_retries=2,
            )
        except Exception as exc:
            logger.error("大模型 API 调用失败，降级处理：%s", exc)
            return self._fallback_response(question, reason=str(exc))

        # 解析响应
        parsed = parse_response(raw_response)
        conclusion = parsed.get("conclusion", raw_response)
        analysis_type = parsed.get("analysis_type")

        # 解析失败：直接展示原始回复，不触发分析
        if not conclusion or analysis_type is None and not raw_response.strip():
            logger.warning("大模型返回为空，降级处理。")
            return self._fallback_response(question, reason="大模型返回为空")

        # 解析失败但有大模型文本：展示原文，不触发图表
        if analysis_type is None:
            logger.info("未解析出 analysis_type，直接展示大模型回复。")
            self._context.add_message("user", question)
            self._context.add_message("assistant", conclusion)
            return {
                "conclusion": conclusion,
                "chart": None,
                "analysis_type": None,
            }

        # 解析成功：触发统计分析 + 图表生成
        chart = None
        stats_text = ""
        try:
            stats_text = run_analysis(self._df, analysis_type, params={})
            # 将统计依据追加到结论后（标注为底层统计）
            if stats_text:
                conclusion = (
                    f"{conclusion}\n\n"
                    f"—— 底层统计依据 ——\n{stats_text}"
                )
        except Exception as exc:
            logger.error("调用 analyzer 失败（type=%s）：%s", analysis_type, exc)

        try:
            if _CHART_AVAILABLE.get(analysis_type, False):
                chart = create_chart(analysis_type, self._df, params={})
        except Exception as exc:
            logger.error("调用 visualizer 失败（type=%s）：%s", analysis_type, exc)

        # 存入对话历史
        self._context.add_message("user", question)
        self._context.add_message("assistant", conclusion)

        return {
            "conclusion": conclusion,
            "chart": chart,
            "analysis_type": analysis_type,
        }

    def _fallback_response(
        self,
        question: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        降级响应：API 不可用时返回本地统计分析结果。

        参数:
            question: 用户问题（用于历史记录）。
            reason: 降级原因（仅记日志）。

        返回:
            降级结果字典，analysis_type 固定为 describe。
        """
        logger.warning("降级响应触发，原因：%s", reason)

        # 尝试根据问题关键词选择更贴近的分析类型
        analysis_type = self._guess_analysis_type(question)
        chart = None
        stats_text = ""

        try:
            stats_text = run_analysis(self._df, analysis_type, params={})
        except Exception as exc:
            logger.error("降级统计分析失败（type=%s）：%s", analysis_type, exc)
            # 再次降级到 describe
            if analysis_type != _FALLBACK_ANALYSIS_TYPE:
                analysis_type = _FALLBACK_ANALYSIS_TYPE
                try:
                    stats_text = run_analysis(self._df, analysis_type, params={})
                except Exception as exc2:
                    logger.error("降级 describe 也失败：%s", exc2)
                    stats_text = ""

        try:
            if _CHART_AVAILABLE.get(analysis_type, False):
                chart = create_chart(analysis_type, self._df, params={})
        except Exception as exc:
            logger.error("降级图表生成失败（type=%s）：%s", analysis_type, exc)

        conclusion = (
            "【AI 服务暂时不可用，以下是基于本地统计的参考结果】\n\n"
            f"{stats_text}"
        ) if stats_text else "AI 服务暂时不可用，且本地统计分析失败。"

        # 存入历史
        self._context.add_message("user", question)
        self._context.add_message("assistant", conclusion)

        return {
            "conclusion": conclusion,
            "chart": chart,
            "analysis_type": analysis_type if stats_text else None,
        }

    @staticmethod
    def _guess_analysis_type(question: str) -> str:
        """
        根据问题关键词猜测分析类型（降级时使用）。

        参数:
            question: 用户问题文本。

        返回:
            猜测的 analysis_type，默认 describe。
        """
        q = question.lower()
        keyword_map = [
            ("趋势", "trend"),
            ("变化", "trend"),
            ("走势", "trend"),
            ("相关", "correlation"),
            ("关联", "correlation"),
            ("类型", "content_type"),
            ("占比", "content_type"),
            ("分布", "distribution"),
            ("异常", "distribution"),
            ("top", "top"),
            ("排行", "top"),
            ("排名", "top"),
            ("前几", "top"),
            ("最高", "top"),
        ]
        for keyword, atype in keyword_map:
            if keyword in q:
                return atype
        return _FALLBACK_ANALYSIS_TYPE


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("AIAssistant 单元测试")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_data
    from data_processor import preprocess_data

    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_content.csv"
    raw_df = load_data(str(sample_file))
    df = preprocess_data(raw_df)

    assistant = AIAssistant(df)
    print(f"\n降级模式：{assistant.is_degraded}")
    print(f"数据摘要长度：{len(assistant.data_summary)} 字符")

    # 1. 测试降级模式下的 ask
    print("\n1. 测试降级 ask（趋势问题）...")
    result1 = assistant.ask("播放量趋势如何？")
    print(f"   conclusion 长度：{len(result1['conclusion'])} 字符")
    print(f"   analysis_type：{result1['analysis_type']}")
    print(f"   chart 类型：{type(result1['chart']).__name__}")
    if (
        result1["analysis_type"] == "trend"
        and result1["chart"] is not None
        and "不可用" in result1["conclusion"]
    ):
        print("   结果：OK")
    else:
        print("   结果：FAIL")

    # 2. 测试降级 ask（相关性问题）
    print("\n2. 测试降级 ask（相关性问题）...")
    result2 = assistant.ask("播放量和点赞数相关性如何？")
    print(f"   analysis_type：{result2['analysis_type']}")
    if (
        result2["analysis_type"] == "correlation"
        and result2["chart"] is not None
    ):
        print("   结果：OK")
    else:
        print("   结果：FAIL")

    # 3. 测试降级 ask（Top N 问题）
    print("\n3. 测试降级 ask（Top N 问题）...")
    result3 = assistant.ask("播放量最高的 5 条内容是哪些？")
    print(f"   analysis_type：{result3['analysis_type']}")
    if result3["analysis_type"] == "top" and result3["chart"] is not None:
        print("   结果：OK")
    else:
        print("   结果：FAIL")

    # 4. 测试降级 ask（默认 describe）
    print("\n4. 测试降级 ask（无明显关键词）...")
    result4 = assistant.ask("数据整体情况怎么样？")
    print(f"   analysis_type：{result4['analysis_type']}")
    if result4["analysis_type"] == "describe" and result4["chart"] is None:
        print("   结果：OK")
    else:
        print("   结果：FAIL")

    # 5. 测试空问题抛异常
    print("\n5. 测试空问题抛 ValueError ...", end=" ")
    try:
        assistant.ask("")
        print("FAIL（应抛异常）")
    except ValueError:
        print("OK")

    # 6. 测试 ask_followup
    print("\n6. 测试 ask_followup（追问）...")
    result6 = assistant.ask_followup("那它的点赞情况呢？")
    print(f"   conclusion 长度：{len(result6['conclusion'])} 字符")
    print(f"   analysis_type：{result6['analysis_type']}")
    if result6["conclusion"] and "不可用" in result6["conclusion"]:
        print("   结果：OK")
    else:
        print("   结果：FAIL")

    # 7. 测试 get_history
    print("\n7. 测试 get_history ...", end=" ")
    history = assistant.get_history()
    # 之前 4 次 ask + 1 次 followup = 5 轮 = 10 条
    # max_history=3，应保留最近 3 轮 = 6 条
    print(f"历史条数：{len(history)}（max_history=3）")
    if len(history) <= 6 and len(history) > 0:
        print("   结果：OK")
    else:
        print(f"   结果：FAIL（len={len(history)}）")

    # 8. 测试 clear_history
    print("\n8. 测试 clear_history ...", end=" ")
    assistant.clear_history()
    if len(assistant.get_history()) == 0:
        print("OK")
    else:
        print("FAIL")

    # 9. 测试 _guess_analysis_type
    print("\n9. 测试 _guess_analysis_type ...", end=" ")
    cases = [
        ("播放量趋势如何", "trend"),
        ("点赞和评论的相关性", "correlation"),
        ("哪种内容类型最多", "content_type"),
        ("播放量分布情况", "distribution"),
        ("播放量 top 5", "top"),
        ("整体情况", "describe"),
    ]
    all_ok = True
    for q, expected in cases:
        got = AIAssistant._guess_analysis_type(q)
        if got != expected:
            print(f"FAIL（'{q}' → {got}，预期 {expected}）")
            all_ok = False
            break
    if all_ok:
        print("OK")

    print("\n测试完成。")
    print("注：如需测试真实 API 调用，请在 .env 中配置 LLM_API_KEY 后重新运行。")
