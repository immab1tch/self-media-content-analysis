# -*- coding: utf-8 -*-
"""
可视化图表生成模块。

基于 matplotlib 与 plotly 生成自媒体数据分析图表，
图表标题与轴标签均使用中文。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 中文字体配置（必须在绘图前设置）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
# 使用非交互后端，避免 GUI 依赖
matplotlib.use("Agg")

# 相关性分析默认指标列（粉丝增量为可选字段，不在默认分析中）
_DEFAULT_CORR_COLS = ("播放量", "点赞数", "评论数", "转发数", "收藏数")
# 视频格式类型列名（原"内容类型"，现语义更准确）
_VIDEO_TYPE_COL = "视频类型"
# 内容分类列名（新增，存主题领域如美食/旅游/科技等）
_CONTENT_CATEGORY_COL = "内容分类"
# 兼容旧字段名：同时支持"视频类型"和旧的"内容类型"
_TYPE_COLS = ("视频类型", "内容类型")
# 内容标题列名
_TITLE_COL = "内容标题"
# 图表默认尺寸
_DEFAULT_FIGSIZE = (10, 6)


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    确保日期列为 datetime 类型，返回转换后的 Series。

    参数:
        df: 数据 DataFrame。
        date_col: 日期列名。

    返回:
        datetime 类型的 Series。
    """
    return pd.to_datetime(df[date_col], errors="coerce")


def _auto_date_grain(df: pd.DataFrame, date_col: str) -> str:
    """
    根据数据时间跨度自动选择日期聚合粒度。

    规则：>90 天按月，>30 天按周，否则按天。

    参数:
        df: 数据 DataFrame。
        date_col: 日期列名。

    返回:
        粒度标识：'D' 天 / 'W' 周 / 'M' 月。
    """
    series = _ensure_datetime(df, date_col).dropna()
    if series.empty:
        return "D"
    span_days = (series.max() - series.min()).days
    if span_days > 90:
        return "M"
    if span_days > 30:
        return "W"
    return "D"


def plot_trend(
    df: pd.DataFrame,
    date_col: str = "发布日期",
    metric: str = "播放量",
    use_plotly: bool = False,
) -> Any:
    """
    折线图：按日期展示指定指标趋势。

    日期粒度自动选择（天/周/月）。默认返回 matplotlib Figure，
    设 use_plotly=True 返回 plotly Figure 交互式图表。

    参数:
        df: 数据 DataFrame。
        date_col: 日期列名，默认 "发布日期"。
        metric: 指标列名，默认 "播放量"。
        use_plotly: 是否使用 plotly 交互式图表，默认 False。

    返回:
        matplotlib Figure 或 plotly Figure 对象。
    """
    if date_col not in df.columns:
        raise ValueError(f"趋势图：未找到日期列 '{date_col}'")
    if metric not in df.columns:
        raise ValueError(f"趋势图：未找到指标列 '{metric}'")

    temp_df = df.copy()
    temp_df[date_col] = _ensure_datetime(temp_df, date_col)
    temp_df = temp_df.dropna(subset=[date_col])

    grain = _auto_date_grain(temp_df, date_col)
    grain_label = {"D": "日", "W": "周", "M": "月"}[grain]

    grouped = (
        temp_df.set_index(date_col)[metric]
        .resample(grain)
        .sum()
        .dropna()
    )

    if use_plotly:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grouped.index,
                y=grouped.values,
                mode="lines+markers",
                name=metric,
                hovertemplate="%{x|%Y-%m-%d}<br>" + f"{metric}: " + "%{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"{metric}{grain_label}度趋势",
            xaxis_title="日期",
            yaxis_title=metric,
            template="plotly_white",
        )
        return fig

    # matplotlib
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.plot(grouped.index, grouped.values, marker="o", linewidth=2, color="#4C78A8")

    for x, y in zip(grouped.index, grouped.values):
        ax.text(
            x, y + max(grouped.values) * 0.01,
            f"{y:.0f}", ha="center", va="bottom", fontsize=8,
        )

    if len(grouped) > 0:
        peak_idx = grouped.idxmax()
        peak_val = grouped.max()
        ax.annotate(
            f"峰值 {peak_val:.0f}",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx, peak_val + max(grouped.values) * 0.08),
            arrowprops=dict(arrowstyle="->", color="#E45756", lw=1.5),
            ha="center", va="bottom", fontsize=8, color="#E45756",
        )

        valley_idx = grouped.idxmin()
        valley_val = grouped.min()
        ax.annotate(
            f"谷值 {valley_val:.0f}",
            xy=(valley_idx, valley_val),
            xytext=(valley_idx, valley_val - max(grouped.values) * 0.08),
            arrowprops=dict(arrowstyle="->", color="#54A24B", lw=1.5),
            ha="center", va="top", fontsize=8, color="#54A24B",
        )

        start_val = grouped.iloc[0]
        end_val = grouped.iloc[-1]
        change_rate = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
        date_range = f"{grouped.index[0].strftime('%Y-%m')} 至 {grouped.index[-1].strftime('%Y-%m')}"
        subtitle = f"{date_range}，总{'增长' if change_rate >= 0 else '下降'} {change_rate:+.1f}%"
        ax.set_title(f"{metric}{grain_label}度趋势\n{subtitle}", fontsize=14, pad=12)
    else:
        ax.set_title(f"{metric}{grain_label}度趋势", fontsize=14, pad=12)

    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    use_plotly: bool = False,
) -> Any:
    """
    热力图：数值字段相关系数矩阵。

    默认返回 matplotlib Figure，设 use_plotly=True 返回 plotly 交互式图表。

    参数:
        df: 数据 DataFrame。
        use_plotly: 是否使用 plotly 交互式图表，默认 False。

    返回:
        matplotlib Figure 或 plotly Figure 对象。
    """
    avail_cols = [c for c in _DEFAULT_CORR_COLS if c in df.columns and df[c].dtype.kind in "biufc"]
    if len(avail_cols) < 2:
        raise ValueError("热力图：可分析的数值列不足 2 个")

    corr_matrix = df[avail_cols].corr()

    if use_plotly:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
                showscale=True,
                hovertemplate="%{x} vs %{y}<br>相关系数: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="指标相关性热力图",
            xaxis_title="",
            yaxis_title="",
            template="plotly_white",
        )
        return fig

    # matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        corr_matrix.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(avail_cols)))
    ax.set_xticklabels(avail_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(avail_cols)))
    ax.set_yticklabels(avail_cols)
    ax.set_title("指标相关性热力图\n数值为皮尔逊相关系数，|r|>0.7 为强相关", fontsize=14, pad=12)
    ax.tick_params(axis="both", labelsize=9)

    # 在每个格子里标注数值
    for i in range(len(avail_cols)):
        for j in range(len(avail_cols)):
            value = corr_matrix.values[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=9, color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def plot_content_type_bar(df: pd.DataFrame) -> Any:
    """
    柱状图：不同视频类型（或内容分类）的数量与平均播放量对比（双 Y 轴）。

    左轴为内容数量，右轴为平均播放量。
    优先使用"视频类型"列，若不存在则尝试"内容类型"列。

    参数:
        df: 数据 DataFrame。

    返回:
        matplotlib Figure 对象。
    """
    # 兼容新旧字段名
    type_col = None
    for col in _TYPE_COLS:
        if col in df.columns:
            type_col = col
            break

    if type_col is None:
        raise ValueError(f"内容类型柱状图：未找到视频类型列，可选：{list(df.columns)}")

    metric = "播放量" if "播放量" in df.columns else None
    type_counts = df[type_col].value_counts().sort_values(ascending=False)
    types = type_counts.index.tolist()
    counts = type_counts.values.tolist()

    fig, ax1 = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    x_pos = np.arange(len(types))
    width = 0.4

    bars = ax1.bar(x_pos - width / 2, counts, width, label="内容数量", color="#4C78A8", alpha=0.85)
    ax1.set_xlabel("视频类型", fontsize=11)
    ax1.set_ylabel("内容数量（条）", fontsize=11, color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8", labelsize=9)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(types)

    subtitle = f"共 {len(types)} 类内容"
    if metric and df[metric].dtype.kind in "biufc":
        avg_plays = [
            float(df[df[type_col] == t][metric].mean()) for t in types
        ]
        subtitle += f"，最高均播：{max(avg_plays):,.0f}"
    ax1.set_title(f"内容类型分布与平均播放量对比\n{subtitle}", fontsize=14, pad=12)

    for bar, val in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(val),
            ha="center", va="bottom", fontsize=9,
        )

    if metric and df[metric].dtype.kind in "biufc":
        avg_plays = [
            float(df[df[type_col] == t][metric].mean()) for t in types
        ]
        ax2 = ax1.twinx()
        line = ax2.plot(
            x_pos + width / 2, avg_plays,
            color="#E45756", marker="s", linewidth=2, label="平均播放量",
        )
        ax2.set_ylabel(f"平均{metric}", fontsize=11, color="#E45756")
        ax2.tick_params(axis="y", labelcolor="#E45756", labelsize=9)
        for xi, val in zip(x_pos, avg_plays):
            ax2.text(
                xi + width / 2, val + max(avg_plays) * 0.02,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=9, color="#E45756",
            )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    if metric and df[metric].dtype.kind in "biufc":
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    else:
        ax1.legend(lines1, labels1, loc="upper left", fontsize=9)

    ax1.grid(True, alpha=0.2, linestyle="--", axis="y")
    fig.tight_layout()
    return fig


def plot_distribution_box(
    df: pd.DataFrame,
    metric: str = "播放量",
) -> Any:
    """
    箱形图：指定指标的分布与异常值。

    使用 IQR 法判定异常值，异常值以红色圆点标注。

    参数:
        df: 数据 DataFrame。
        metric: 指标列名，默认 "播放量"。

    返回:
        matplotlib Figure 对象。
    """
    if metric not in df.columns:
        raise ValueError(f"箱形图：未找到指标列 '{metric}'")
    if df[metric].dtype.kind not in "biufc":
        raise ValueError(f"箱形图：指标列 '{metric}' 不是数值类型")

    series = df[metric].dropna()
    if series.empty:
        raise ValueError(f"箱形图：指标列 '{metric}' 无有效数据")

    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(
        [series.values],
        vert=True,
        patch_artist=True,
        labels=[metric],
        showfliers=True,
        flierprops=dict(marker="o", markerfacecolor="#E45756", markersize=6, alpha=0.7),
        medianprops=dict(color="#E45756", linewidth=2),
        boxprops=dict(facecolor="#4C78A8", alpha=0.6, edgecolor="#333"),
        whiskerprops=dict(linewidth=1.2, color="#333"),
        capprops=dict(linewidth=1.2, color="#333"),
    )

    # 标注四分位数数值
    q1 = float(series.quantile(0.25))
    med = float(series.median())
    q3 = float(series.quantile(0.75))

    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    outlier_count = int(len(outliers))

    ax.text(1.15, q1, f"Q1={q1:.0f}", va="center", fontsize=9, color="#333")
    ax.text(1.15, med, f"中位数={med:.0f}", va="center", fontsize=9, color="#E45756")
    ax.text(1.15, q3, f"Q3={q3:.0f}", va="center", fontsize=9, color="#333")

    for outlier_val in outliers:
        ax.text(
            1.18, outlier_val, f"{outlier_val:.0f}",
            va="center", fontsize=8, color="#E45756",
        )

    subtitle = f"IQR: {iqr:.0f}，异常值: {outlier_count} 个"
    ax.set_title(f"{metric} 分布箱形图\n{subtitle}", fontsize=14, pad=12)

    fig.tight_layout()
    return fig


def plot_top_content(
    df: pd.DataFrame,
    metric: str = "播放量",
    n: int = 5,
) -> Any:
    """
    横向柱状图：Top N 内容对比。

    按指定指标排序，从上到下展示前 N 条内容标题。

    参数:
        df: 数据 DataFrame。
        metric: 排序指标列名，默认 "播放量"。
        n: 返回前 N 条，默认 5。

    返回:
        matplotlib Figure 对象。
    """
    if metric not in df.columns:
        raise ValueError(f"Top N 柱状图：未找到指标列 '{metric}'")
    if df[metric].dtype.kind not in "biufc":
        raise ValueError(f"Top N 柱状图：指标列 '{metric}' 不是数值类型")

    title_col = _TITLE_COL if _TITLE_COL in df.columns else df.columns[0]
    top_df = df.nlargest(n, metric).iloc[::-1]  # 反转，使最大的在顶部

    fig_height = max(4, 0.8 * len(top_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(top_df))
    values = top_df[metric].values
    titles = top_df[title_col].astype(str).tolist()

    # 标题过长截断
    display_titles = [t if len(t) <= 18 else t[:16] + "…" for t in titles]

    bars = ax.barh(y_pos, values, height=0.6, color="#4C78A8", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_titles, fontsize=10)
    ax.set_xlabel(metric, fontsize=11)

    subtitle = f"Top {len(top_df)}，最高：{max(values):,.0f}" if len(values) > 0 else f"Top {len(top_df)}"
    ax.set_title(f"{metric} Top {len(top_df)} 内容排行\n{subtitle}", fontsize=14, pad=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.invert_yaxis()

    # 数值标注
    max_val = max(values) if len(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlim(right=max_val * 1.15)
    ax.grid(True, alpha=0.2, linestyle="--", axis="x")
    fig.tight_layout()
    return fig


def create_chart(
    chart_type: str,
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> Any:
    """
    根据图表类型生成对应可视化（AI 助手统一调度入口）。

    参数:
        chart_type: 图表类型标识，可选：
            trend / correlation / content_type / distribution / top
        df: 数据 DataFrame。
        params: 图表参数字典，对应各函数的 keyword 参数。

    返回:
        matplotlib Figure 或 plotly Figure 对象。

    异常:
        ValueError: 图表类型不支持时抛出。
    """
    params = params or {}

    dispatch = {
        "trend": lambda: plot_trend(
            df,
            date_col=params.get("date_col", "发布日期"),
            metric=params.get("metric", "播放量"),
            use_plotly=params.get("use_plotly", False),
        ),
        "correlation": lambda: plot_correlation_heatmap(
            df,
            use_plotly=params.get("use_plotly", False),
        ),
        "content_type": lambda: plot_content_type_bar(df),
        "distribution": lambda: plot_distribution_box(
            df,
            metric=params.get("metric", "播放量"),
        ),
        "top": lambda: plot_top_content(
            df,
            metric=params.get("metric", "播放量"),
            n=int(params.get("n", 5)),
        ),
        "content_recommend": lambda: None,
    }

    if chart_type not in dispatch:
        raise ValueError(
            f"不支持的图表类型 '{chart_type}'，可选：{', '.join(dispatch.keys())}"
        )

    return dispatch[chart_type]()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_data
    from data_processor import preprocess_data

    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_content.csv"
    raw_df = load_data(str(sample_file))
    df = preprocess_data(raw_df)

    output_dir = Path(__file__).resolve().parent.parent / "data" / "sample" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        ("趋势图", lambda: plot_trend(df)),
        ("相关性热力图", lambda: plot_correlation_heatmap(df)),
        ("内容类型柱状图", lambda: plot_content_type_bar(df)),
        ("分布箱形图", lambda: plot_distribution_box(df)),
        ("Top 5 横向柱状图", lambda: plot_top_content(df)),
        ("统一入口 - top n=3", lambda: create_chart("top", df, {"n": 3, "metric": "点赞数"})),
    ]

    all_ok = True
    for name, func in tests:
        print(f"\n测试：{name} ...", end=" ")
        try:
            fig = func()
            safe_name = name.replace(" ", "_").replace("-", "_")
            # matplotlib Figure 保存为 PNG
            if hasattr(fig, "savefig"):
                fig.savefig(output_dir / f"{safe_name}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print("OK (matplotlib)")
            elif hasattr(fig, "write_image"):
                fig.write_image(str(output_dir / f"{safe_name}.png"))
                print("OK (plotly)")
            else:
                print(f"OK (type={type(fig).__name__})")
        except Exception as exc:
            all_ok = False
            print(f"FAIL: {exc}")
            logger.error("测试 '%s' 失败: %s", name, exc)

    print(f"\n全部测试完成：{'通过' if all_ok else '存在失败'}")
    print(f"图表输出目录：{output_dir}")
