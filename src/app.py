# -*- coding: utf-8 -*-
"""
主程序入口模块。

基于 Streamlit 搭建 Web 界面，整合数据导入、AI 问答与可视化展示。
"""

import io
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logging.getLogger("streamlit").setLevel(logging.WARNING)

# 将 src 目录加入 sys.path，以便同目录模块可互相导入
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from analyzer import run_analysis
from data_loader import load_data
from data_processor import preprocess_data
from data_fetcher import fetch_data, get_platform_info, SUPPORTED_PLATFORMS
from visualizer import create_chart

logger = logging.getLogger(__name__)
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger.info("=== 自媒体账号内容数据分析系统启动 ===")
logger.info(f"Python 版本: {sys.version}")
logger.info(f"工作目录: {os.getcwd()}")

# 页面配置（必须是首个 Streamlit 命令）
st.set_page_config(
    page_title="自媒体账号内容数据分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    """初始化 Streamlit 会话状态。"""
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None  # 原始数据
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None  # 预处理后数据
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None  # 数据摘要
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = None  # AI 助手实例
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 对话历史（每条含 role/content/chart/analysis_type）
    if "degraded_warning_shown" not in st.session_state:
        st.session_state.degraded_warning_shown = False


def _load_ai_assistant():
    """
    懒加载 AI 助手实例（首次使用时导入，避免 Streamlit 启动慢）。
    """
    if st.session_state.ai_assistant is not None:
        return

    logger.info("========== 初始化 AI 助手 ==========")
    logger.info(f"LLM_API_KEY 配置: {'已配置 (' + str(len(os.environ.get('LLM_API_KEY', ''))) + ' 字符)' if os.environ.get('LLM_API_KEY') else '未配置'}")
    logger.info(f"LLM_API_URL 配置: {os.environ.get('LLM_API_URL', '未配置')}")
    logger.info(f"LLM_MODEL 配置: {os.environ.get('LLM_MODEL', '未配置')}")

    from ai_assistant import AIAssistant

    try:
        logger.info("正在创建 AIAssistant 实例...")
        st.session_state.ai_assistant = AIAssistant(st.session_state.df_processed)
        logger.info(f"AI 助手初始化成功！降级模式: {st.session_state.ai_assistant.is_degraded}")
        if st.session_state.ai_assistant.is_degraded:
            logger.warning("⚠️ 降级模式：AI 服务不可用，将使用本地统计模式")
        else:
            logger.info("✅ AI 模式：DeepSeek API 已连接")
    except Exception as exc:
        logger.error("❌ AI 助手初始化失败：%s", exc, exc_info=True)
        st.error(f"AI 助手初始化失败：{exc}")
        st.session_state.ai_assistant = None


def _handle_file_upload(uploaded_file) -> None:
    """
    处理文件上传：加载数据 → 预处理 → 生成摘要。

    参数:
        uploaded_file: Streamlit 上传的文件对象。
    """
    if uploaded_file is None:
        return

    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    # 写入临时文件后调用 load_data（load_data 只接受文件路径）
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        df_raw = load_data(tmp_path)
        df_processed = preprocess_data(df_raw)
    except Exception as exc:
        logger.error("数据加载失败：%s", exc)
        st.error(f"数据加载失败：{exc}")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    st.session_state.df_raw = df_raw
    st.session_state.df_processed = df_processed

    # 生成数据摘要（供 AI 使用）
    from data_processor import generate_data_summary
    st.session_state.data_summary = generate_data_summary(df_processed)

    # 重置 AI 助手与对话历史
    st.session_state.ai_assistant = None
    st.session_state.chat_history = []

    st.success(f"数据加载成功！共 {len(df_processed)} 条记录。")


def _render_sidebar() -> None:
    """渲染侧边栏：文件上传 + 自动获取 + 数据预览。"""
    with st.sidebar:
        st.title("⚙️ 数据配置")
        st.markdown("---")

        tab1, tab2 = st.tabs(["📤 上传文件", "🔗 自动获取"])

        with tab1:
            uploaded_file = st.file_uploader(
                "上传数据文件",
                type=["csv", "xlsx", "xls"],
                help="支持 CSV、Excel（.xlsx/.xls）格式",
            )
            if uploaded_file is not None:
                _handle_file_upload(uploaded_file)

        with tab2:
            platform = st.selectbox(
                "选择平台",
                list(SUPPORTED_PLATFORMS.keys()),
                index=0,
            )

            info = get_platform_info(platform)
            st.caption(f"示例UID: {info.get('account_id_example', '')}")
            st.caption(info.get("note", ""))

            account_id = st.text_input(
                "输入账号ID",
                placeholder=info.get("account_id_format", "请输入账号ID"),
            )

            max_videos = st.slider(
                "获取视频数量",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )

            if st.button("🚀 获取数据", use_container_width=True):
                if not account_id or not account_id.strip():
                    st.warning("请输入账号ID")
                else:
                    with st.spinner("正在获取数据..."):
                        try:
                            df_raw = fetch_data(platform, account_id.strip(), max_videos)
                            df_processed = preprocess_data(df_raw)

                            st.session_state.df_raw = df_raw
                            st.session_state.df_processed = df_processed

                            from data_processor import generate_data_summary
                            st.session_state.data_summary = generate_data_summary(df_processed)

                            st.session_state.ai_assistant = None
                            st.session_state.chat_history = []

                            st.success(f"成功获取 {len(df_processed)} 条视频数据！")
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"获取数据失败：{exc}")

        st.markdown("---")

        if st.session_state.df_processed is not None:
            st.subheader("📋 数据预览")
            df = st.session_state.df_processed
            st.write(f"共 **{len(df)}** 条记录，**{len(df.columns)}** 个字段")

            preview_count = min(10, len(df))
            st.dataframe(
                df.head(preview_count),
                width='stretch',
                height=300,
            )

            with st.expander("📊 数据摘要"):
                if st.session_state.data_summary:
                    st.text(st.session_state.data_summary)

            st.markdown("---")
            if st.button("🗑️ 清空聊天历史", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.ai_assistant is not None:
                    st.session_state.ai_assistant.clear_history()
                st.success("聊天历史已清空。")
                st.rerun()
        else:
            st.info("请先上传数据文件或自动获取数据")

            sample_path = _src_dir.parent / "data" / "sample" / "sample_content.csv"
            if sample_path.exists():
                st.markdown("---")
                st.caption("💡 没有数据？试试示例数据")
                if st.button("📎 加载示例数据", use_container_width=True):
                    try:
                        df_raw = load_data(str(sample_path))
                        df_processed = preprocess_data(df_raw)
                        st.session_state.df_raw = df_raw
                        st.session_state.df_processed = df_processed
                        from data_processor import generate_data_summary
                        st.session_state.data_summary = generate_data_summary(df_processed)
                        st.session_state.ai_assistant = None
                        st.session_state.chat_history = []
                        st.success("示例数据加载成功！")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"示例数据加载失败：{exc}")

        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if api_key:
            model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
            st.caption(f"🤖 AI 智能模式（{model}）")
        else:
            st.caption("📊 本地统计模式（配置 API Key 可开启 AI 对话）")


def _render_degraded_warning() -> None:
    """渲染降级模式提示（API 未配置时）。"""
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if api_key:
        return

    if not st.session_state.degraded_warning_shown:
        st.info(
            "💡 **提示**：未配置 AI 密钥，当前使用本地统计模式。"
            "在 `.env` 文件中配置 `LLM_API_KEY`（DeepSeek API Key）即可开启 AI 智能对话。"
            "\n\n本地模式也能做数据分析和图表，只是没有 AI 对话那么智能。",
        )
        st.session_state.degraded_warning_shown = True


def _render_chat_input() -> Optional[str]:
    """
    渲染 AI 问答输入框。

    返回:
        用户输入的问题字符串，或 None。
    """
    st.subheader("💬 AI 数据分析助手")

    disabled = st.session_state.df_processed is None
    if disabled:
        st.info("请先在侧边栏上传数据文件，然后开始提问。")
        return None

    quick_questions = [
        "📈 分析播放量趋势",
        "🔗 各指标相关性分析",
        "🏆 播放量 Top 5 内容",
        "📊 内容类型占比分析",
        "💡 推荐下一步内容方向",
        "📋 数据整体描述统计",
    ]

    selected_question = None
    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, use_container_width=True):
                selected_question = q

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            question = st.text_input(
                "请输入你的问题",
                placeholder="例如：哪种内容类型的播放量最高？",
                label_visibility="collapsed",
                disabled=disabled,
            )
        with col2:
            submitted = st.form_submit_button("🚀 发送", use_container_width=True)

    if submitted and question and question.strip():
        return question.strip()
    if selected_question:
        return selected_question
    return None


def _render_conclusion(result: Dict[str, Any]) -> None:
    """
    渲染 AI 分析结论文字，模仿 ChatGPT 风格。

    参数:
        result: AI 助手返回的结果字典。
    """
    conclusion = result.get("content", "") or result.get("conclusion", "")
    analysis_type = result.get("analysis_type")

    if not conclusion:
        return

    st.success(f"🤖 **AI 智能分析**\n\n{conclusion}")

    if analysis_type:
        type_names = {
            "describe": "📋 数据概览",
            "correlation": "🔗 相关性分析",
            "trend": "📈 趋势分析",
            "content_type": "📊 类型对比",
            "top": "🏆 排行榜",
            "distribution": "📉 分布分析",
            "recommend": "💡 内容推荐",
            "content_recommend": "💡 内容推荐",
        }
        type_name = type_names.get(analysis_type, analysis_type)
        st.caption(type_name)


def _render_chart(result: Dict[str, Any]) -> None:
    """
    渲染可视化图表。

    参数:
        result: AI 助手返回的结果字典。
    """
    chart = result.get("chart")
    if chart is None:
        return

    # 判断是 matplotlib 还是 plotly
    if hasattr(chart, "update_layout") and hasattr(chart, "write_image"):
        # plotly Figure
        st.plotly_chart(chart, width='stretch')
    elif hasattr(chart, "savefig"):
        # matplotlib Figure
        st.pyplot(chart, width='stretch')
    else:
        logger.warning("未知图表类型：%s", type(chart).__name__)


def _render_history() -> None:
    """渲染底部对话历史记录。"""
    if not st.session_state.chat_history:
        return

    st.markdown("---")
    st.subheader("� 对话历史")

    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_conclusion(msg)
                if msg.get("chart") is not None:
                    st.subheader("📊 可视化图表")
                    _render_chart(msg)


def _process_question(question: str) -> None:
    """
    处理用户问题：调用 AI 助手 → 存储历史 → 展示结果。

    参数:
        question: 用户问题文本。
    """
    logger.info("\n========== 处理用户问题 ==========")
    logger.info(f"问题: {question}")

    # 先添加用户消息到历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # 判断是首问还是追问
    has_history = len(
        [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    ) > 0
    logger.info(f"是否追问: {has_history}")

    # 懒加载 AI 助手
    _load_ai_assistant()

    result: Dict[str, Any] = {
        "conclusion": "",
        "chart": None,
        "analysis_type": None,
    }

    if st.session_state.ai_assistant is not None and not st.session_state.ai_assistant.is_degraded:
        logger.info("模式: AI 模式")
        try:
            if has_history:
                logger.info("调用: ask_followup")
                result = st.session_state.ai_assistant.ask_followup(question)
            else:
                logger.info("调用: ask")
                result = st.session_state.ai_assistant.ask(question)
            logger.info(f"AI 返回结果: conclusion长度={len(result.get('conclusion',''))}, analysis_type={result.get('analysis_type')}, chart={'有' if result.get('chart') else '无'}")
        except Exception as exc:
            logger.error("❌ AI 问答失败：%s", exc, exc_info=True)
            result["conclusion"] = f"AI 服务调用失败：{exc}"
            result["analysis_type"] = None
    else:
        logger.info("模式: 降级模式（本地统计）")
        result = _local_statistics_answer(question)
        logger.info(f"本地统计结果: conclusion长度={len(result.get('conclusion',''))}, analysis_type={result.get('analysis_type')}")

    # 存储助手回复
    assistant_msg = {
        "role": "assistant",
        "content": result.get("conclusion", ""),
        "chart": result.get("chart"),
        "analysis_type": result.get("analysis_type"),
    }
    st.session_state.chat_history.append(assistant_msg)


def _local_statistics_answer(question: str) -> Dict[str, Any]:
    """
    降级模式：基于关键词匹配的本地统计分析。

    参数:
        question: 用户问题。

    返回:
        同 AIAssistant.ask 的返回结构。
    """
    df = st.session_state.df_processed
    if df is None:
        return {"conclusion": "暂无数据。", "chart": None, "analysis_type": None}

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
        ("最高", "top"),
        ("最大", "top"),
        ("推荐", "recommend"),
        ("建议", "recommend"),
        ("下一步", "recommend"),
        ("拍什么", "recommend"),
        ("做什么", "recommend"),
        ("内容方向", "recommend"),
    ]
    q = question.lower()
    analysis_type = "describe"
    matched = False
    for keyword, atype in keyword_map:
        if keyword in q:
            analysis_type = atype
            matched = True
            break

    chart = None
    stats_text = ""
    try:
        if analysis_type == "recommend":
            try:
                stats_text = run_analysis(df, "recommend", params={})
                if not stats_text or len(stats_text.strip()) < 20:
                    # 兜底
                    type_analysis = run_analysis(df, "content_type", params={})
                    top_analysis = run_analysis(df, "top", params={"n": 3})
                    stats_text = (
                        f"【本地推荐分析】\n\n{type_analysis}\n\n{top_analysis}\n\n"
                        "建议：延续高表现内容分类的选题方向，结合热门关键词进行创作。"
                    )
            except Exception as exc:
                logger.error("本地 recommend 失败：%s", exc)
                stats_text = (
                    "【本地推荐分析】\n基于已有数据的参考建议：\n"
                    "1. 查看视频类型/内容分类占比分布\n"
                    "2. 关注播放量 Top 内容的共同特征\n"
                    "3. 延续高表现的内容分类进行选题"
                )
        else:
            stats_text = run_analysis(df, analysis_type, params={})
            if not matched and stats_text:
                stats_text = f"未识别到特定的分析意图，以下为数据整体描述统计：\n\n{stats_text}"
    except Exception as exc:
        logger.error("本地统计分析失败：%s", exc)
        if analysis_type != "describe":
            analysis_type = "describe"
            try:
                stats_text = run_analysis(df, analysis_type, params={})
                if stats_text:
                    stats_text = f"未识别到特定的分析意图，以下为数据整体描述统计：\n\n{stats_text}"
            except Exception as exc2:
                logger.error("本地 describe 也失败：%s", exc2)
                stats_text = ""

    try:
        chart_available = {
            "describe": False,
            "correlation": True,
            "trend": True,
            "content_type": True,
            "top": True,
            "distribution": True,
            "recommend": False,
            "content_recommend": False,
        }
        if chart_available.get(analysis_type, False):
            chart = create_chart(analysis_type, df, params={})
    except Exception as exc:
        logger.error("本地图表生成失败：%s", exc)

    conclusion = stats_text if stats_text else "本地统计分析失败。"

    return {
        "conclusion": conclusion,
        "chart": chart,
        "analysis_type": analysis_type if stats_text else None,
    }


def _render_main_area() -> None:
    """渲染主区域：问答输入 + 对话历史。"""
    st.title("📊 自媒体账号内容数据分析系统")
    st.caption("AI 驱动的自然语言数据分析助手 · 导入数据，用大白话提问，即刻获得结论与图表")
    st.divider()

    _render_degraded_warning()

    question = _render_chat_input()
    if question:
        with st.spinner("🤔 正在分析你的数据..."):
            _process_question(question)

    if st.session_state.chat_history:
        st.markdown("---")
        # 倒序展示，最新对话在最上面（更像聊天软件）
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    _render_conclusion(msg)
                    if msg.get("chart") is not None:
                        with st.expander("📊 查看图表", expanded=True):
                            _render_chart(msg)
    else:
        if st.session_state.df_processed is not None:
            st.info(
                "👋 数据已加载！试着问我一些问题，例如：\n\n"
                "- 📈 播放量趋势如何？\n"
                "- 🔗 播放量和点赞数的相关性怎么样？\n"
                "- 🏆 播放量最高的 5 条内容是哪些？\n"
                "- 📊 哪种内容分类表现最好？\n"
                "- 💡 推荐下一步做什么内容？"
            )


def main() -> None:
    """Streamlit 应用主入口。"""
    _init_session_state()
    _render_sidebar()
    _render_main_area()


if __name__ == "__main__":
    main()
