# -*- coding: utf-8 -*-
"""
AI 自然语言数据分析助手核心编排模块。

整合 api_client、prompt_engine、context_manager、analyzer、visualizer，
将用户自然语言提问转化为「结论文字 + 可视化图表」的联动输出。

设计原则：
1. AI 结论优先——让大模型真正发挥作用，本地统计仅作为补充和图表支撑
2. 降级模式兜底——API 不可用时使用本地统计分析
3. 推荐类问题由 AI 直接生成，不需要本地统计函数
4. 完善诊断日志，便于排查问题
"""

import logging
import requests
from typing import Any, Dict, List, Optional

import pandas as pd

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


_CHART_AVAILABLE = {
    "describe": False,
    "correlation": True,
    "trend": True,
    "content_type": True,
    "top": True,
    "distribution": True,
    "recommend": False,
}

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

        self._data_summary = generate_data_summary(df)
        self._system_prompt = build_system_prompt(self._data_summary)
        logger.info("AI 助手初始化完成，数据摘要长度=%d 字符", len(self._data_summary))

        self._llm_client: Optional[LLMClient] = None
        self._degraded_mode = False
        try:
            self._llm_client = LLMClient()
        except ValueError as exc:
            logger.warning("LLM 客户端初始化失败，进入降级模式：%s", exc)
            self._degraded_mode = True

    @property
    def is_degraded(self) -> bool:
        return self._degraded_mode

    @property
    def data_summary(self) -> str:
        return self._data_summary

    def ask(self, question: str) -> Dict[str, Any]:
        """
        处理用户自然语言提问并返回分析结果（单轮问答）。
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
        return self._context.get_history()

    def clear_history(self) -> None:
        self._context.clear()

    def _run_pipeline(
        self,
        question: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        执行问答核心流水线：调用模型 → 解析 → 触发分析 → 生成图表 → 存历史。
        """
        if self._degraded_mode or self._llm_client is None:
            logger.info("降级模式：返回本地统计分析结果。")
            return self._fallback_response(question, reason="AI 服务未配置")

        search_context = self._try_web_search(question)
        if search_context:
            enhanced_system = (
                self._system_prompt
                + f"\n\n# 联网搜索补充信息（来自实时网络搜索，可作为推荐参考）\n{search_context}"
            )
            messages = [
                {"role": "system", "content": enhanced_system},
                messages[1],
            ]
            logger.info("已注入联网搜索结果（长度=%d）", len(search_context))

        try:
            raw_response = self._llm_client.chat_with_retry(
                messages=messages,
                max_retries=2,
                temperature=0.5,
                max_tokens=3000,
            )
            logger.info("LLM API 调用成功，原始回复长度=%d", len(raw_response))
        except Exception as exc:
            logger.error("大模型 API 调用失败，降级处理：%s", exc)
            return self._fallback_response(question, reason=str(exc))

        parsed = parse_response(raw_response)
        conclusion = parsed.get("conclusion")
        analysis_type = parsed.get("analysis_type")

        logger.info(
            "解析结果：conclusion长度=%d, analysis_type=%s",
            len(conclusion) if conclusion else 0,
            analysis_type,
        )

        if not conclusion or not conclusion.strip():
            if raw_response and raw_response.strip():
                conclusion = raw_response
                logger.warning("解析结论为空，使用原始 LLM 回复")
            else:
                logger.warning("大模型返回完全为空，降级处理")
                return self._fallback_response(question, reason="大模型返回为空")

        if analysis_type is None:
            analysis_type = self._guess_analysis_type(question)
            logger.info("基于关键词猜测 analysis_type=%s", analysis_type)

        chart = None
        if analysis_type != "recommend":
            try:
                if _CHART_AVAILABLE.get(analysis_type, False):
                    chart = create_chart(analysis_type, self._df, params={})
                    logger.info("图表生成成功（type=%s）", analysis_type)
            except Exception as exc:
                logger.error("调用 visualizer 失败（type=%s）：%s", analysis_type, exc)

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
        """
        logger.warning("降级响应触发，原因：%s", reason)

        analysis_type = self._guess_analysis_type(question)
        chart = None
        stats_text = ""

        if analysis_type == "recommend":
            logger.info("recommend 类型降级：返回基于数据的本地推荐分析。")
            try:
                type_analysis = run_analysis(self._df, "content_type", params={})
                top_analysis = run_analysis(self._df, "top", params={"n": 3})
                stats_text = (
                    "【本地统计模式 · 内容推荐分析】\n\n"
                    f"📊 内容类型分布：\n{type_analysis}\n\n"
                    f"🏆 播放量 Top 3 内容：\n{top_analysis}\n\n"
                    "💡 建议方向：\n"
                    "1. 延续播放量较高的内容分类进行选题\n"
                    "2. 关注互动率（点赞/播放）高的内容特征\n"
                    "3. 结合热门标题关键词进行创作"
                )
            except Exception as exc:
                logger.error("recommend 降级统计失败：%s", exc)
                stats_text = (
                    "【本地统计模式 · 内容推荐分析】\n\n"
                    "当前无法连接 AI 服务，但基于已有数据给出参考建议：\n"
                    "1. 查看上方「视频类型占比」和「Top N 排行」图表\n"
                    "2. 延续播放量较高的内容分类进行选题\n"
                    "3. 关注互动率（点赞/播放）高的内容特征"
                )
        else:
            try:
                stats_text = run_analysis(self._df, analysis_type, params={})
            except Exception as exc:
                logger.error("降级统计分析失败（type=%s）：%s", analysis_type, exc)
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

        self._context.add_message("user", question)
        self._context.add_message("assistant", conclusion)

        return {
            "conclusion": conclusion,
            "chart": chart,
            "analysis_type": analysis_type if stats_text else None,
        }

    @staticmethod
    def _try_web_search(question: str) -> str:
        recommend_keywords = [
            "推荐", "建议", "下一步", "拍什么", "做什么",
            "内容方向", "选题", "趋势", "热门", "火",
        ]
        q_lower = question.lower()
        if not any(kw in q_lower for kw in recommend_keywords):
            return ""

        search_query = f"自媒体内容创作趋势 2026 热门选题方向"

        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": search_query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                },
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                snippets = []
                abstract = data.get("AbstractText", "")
                if abstract:
                    snippets.append(abstract)
                for topic in data.get("RelatedTopics", [])[:5]:
                    text = topic.get("Text", "")
                    if text:
                        snippets.append(text)
                if snippets:
                    result = "以下是最新的自媒体创作趋势信息：\n" + "\n".join(
                        f"- {s}" for s in snippets[:8]
                    )
                    logger.debug("联网搜索成功，获取 %d 条结果", len(snippets))
                    return result
        except Exception:
            pass

        return ""

    @staticmethod
    def _guess_analysis_type(question: str) -> str:
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
            ("推荐", "recommend"),
            ("建议", "recommend"),
            ("下一步", "recommend"),
            ("拍什么", "recommend"),
            ("做什么", "recommend"),
            ("内容方向", "recommend"),
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

    if not assistant.is_degraded:
        print("\nAI 模式测试：")
        print("=" * 60)
        
        test_questions = [
            "播放量趋势如何？",
            "哪种内容类型表现最好？",
            "自媒体账号下一步创作什么内容？",
        ]
        
        for q in test_questions:
            print(f"\n问题：{q}")
            try:
                result = assistant.ask(q)
                print(f"分析类型：{result['analysis_type']}")
                print(f"结论长度：{len(result['conclusion'])} 字符")
                print(f"图表：{'有' if result['chart'] else '无'}")
                print(f"结论预览：{result['conclusion'][:150]}...")
            except Exception as e:
                print(f"错误：{e}")
    else:
        print("\n降级模式测试：")
        print("=" * 60)
        
        result = assistant.ask("播放量趋势如何？")
        print(f"分析类型：{result['analysis_type']}")
        print(f"结论长度：{len(result['conclusion'])} 字符")
        
        result2 = assistant.ask("自媒体账号下一步创作什么内容？")
        print(f"\n推荐问题分析类型：{result2['analysis_type']}")
        print(f"结论长度：{len(result2['conclusion'])} 字符")

    print("\n测试完成。")