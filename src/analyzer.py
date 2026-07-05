# -*- coding: utf-8 -*-
"""
统计分析模块。

封装自媒体内容数据的常用统计函数，供 AI 助手调用
或可视化模块直接使用。所有分析函数仅返回文字结果，
绘图由 visualizer.py 负责。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 相关性分析默认指标列
_DEFAULT_CORR_COLS = ("播放量", "点赞数", "评论数", "转发数", "收藏数", "粉丝增量")
# 内容类型列名
_CONTENT_TYPE_COL = "内容类型"
# 内容标题列名
_TITLE_COL = "内容标题"


def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """
    获取数值列名列表（排除日期列与内部标记列）。

    参数:
        df: 数据 DataFrame。

    返回:
        数值列名列表。
    """
    return [
        col for col in df.columns
        if df[col].dtype.kind in "biufc"
        and not col.endswith("_缺失标记")
    ]


def _validate_column(df: pd.DataFrame, col: str, context: str) -> None:
    """
    校验列是否存在，不存在则抛出 ValueError。

    参数:
        df: 数据 DataFrame。
        col: 待校验列名。
        context: 错误上下文描述。
    """
    if col not in df.columns:
        raise ValueError(f"{context}：未找到列 '{col}'，可选列：{list(df.columns)}")


def compute_basic_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    计算指定列的基础统计指标。

    参数:
        df: 数据 DataFrame。
        column: 待统计的列名。

    返回:
        包含均值、中位数、最值、分位数、标准差的字典。
    """
    _validate_column(df, column, "基础统计")
    if df[column].dtype.kind not in "biufc":
        raise ValueError(f"基础统计：列 '{column}' 不是数值类型（{df[column].dtype}）")

    series = df[column].dropna()
    if series.empty:
        return {}

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()) if series.count() > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }


def aggregate_by_dimension(
    df: pd.DataFrame,
    group_by: str,
    metric: str,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    按维度聚合统计指标。

    参数:
        df: 数据 DataFrame。
        group_by: 分组维度列名。
        metric: 聚合指标列名。
        agg_func: 聚合函数名称，如 sum、mean、count。

    返回:
        聚合后的 DataFrame，索引为分组键，列为聚合指标。
    """
    _validate_column(df, group_by, "按维度聚合")
    if agg_func != "count":
        _validate_column(df, metric, "按维度聚合")

    agg_map = {metric: agg_func} if agg_func != "count" else {group_by: "count"}
    result = df.groupby(group_by).agg(agg_map)
    if agg_func == "count":
        result.columns = ["数量"]
    return result.sort_index()


def describe_statistics(df: pd.DataFrame) -> str:
    """
    描述性统计分析。

    输出各数值字段的均值、中位数、标准差、分位数。

    参数:
        df: 数据 DataFrame。

    返回:
        描述性统计结果文本。
    """
    numeric_cols = _get_numeric_cols(df)
    if not numeric_cols:
        return "[描述性统计] 无数值列可分析。"

    lines: List[str] = ["[描述性统计] 分析结果："]
    for col in numeric_cols:
        stats = compute_basic_stats(df, col)
        if not stats:
            continue
        lines.append(
            f"- {col}：样本数={stats['count']}，均值={stats['mean']:.2f}，"
            f"中位数={stats['median']:.2f}，标准差={stats['std']:.2f}，"
            f"最小值={stats['min']:.2f}，最大值={stats['max']:.2f}，"
            f"Q25={stats['q25']:.2f}，Q75={stats['q75']:.2f}"
        )
    return "\n".join(lines)


def correlation_analysis(df: pd.DataFrame) -> str:
    """
    相关性分析。

    计算播放量、点赞、评论、转发、收藏、粉丝增量之间的相关系数矩阵。

    参数:
        df: 数据 DataFrame。

    返回:
        相关系数分析结果文本。
    """
    avail_cols = [c for c in _DEFAULT_CORR_COLS if c in df.columns and df[c].dtype.kind in "biufc"]
    if len(avail_cols) < 2:
        return "[相关性分析] 可分析的数值列不足 2 个，无法计算相关性。"

    corr_matrix = df[avail_cols].corr()
    lines: List[str] = [f"[相关性分析] 相关系数矩阵（{', '.join(avail_cols)}）："]

    # 逐行输出上三角矩阵，避免冗余
    for i, row_col in enumerate(avail_cols):
        for j, col_col in enumerate(avail_cols):
            if j <= i:
                continue
            value = float(corr_matrix.loc[row_col, col_col])
            if abs(value) >= 0.8:
                level = "强相关"
            elif abs(value) >= 0.5:
                level = "中等相关"
            elif abs(value) >= 0.3:
                level = "弱相关"
            else:
                level = "几乎无关"
            lines.append(f"- {row_col} 与 {col_col}：{value:+.3f}（{level}）")
    return "\n".join(lines)


def _auto_date_grain(
    df: pd.DataFrame,
    date_col: str,
) -> str:
    """
    根据数据时间跨度自动选择日期聚合粒度。

    规则：>90 天按月，>30 天按周，否则按天。

    参数:
        df: 数据 DataFrame。
        date_col: 日期列名。

    返回:
        粒度标识：'D' 天 / 'W' 周 / 'M' 月。
    """
    series = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if series.empty:
        return "D"
    span_days = (series.max() - series.min()).days
    if span_days > 90:
        return "M"
    if span_days > 30:
        return "W"
    return "D"


def trend_analysis(
    df: pd.DataFrame,
    date_col: str = "发布日期",
    metric: str = "播放量",
) -> str:
    """
    趋势分析：按日期聚合，输出指定指标的变化趋势。

    日期粒度按数据跨度自动选择（天/周/月）。

    参数:
        df: 数据 DataFrame。
        date_col: 日期列名，默认 "发布日期"。
        metric: 待分析指标列名，默认 "播放量"。

    返回:
        趋势分析结果文本。
    """
    _validate_column(df, date_col, "趋势分析")
    _validate_column(df, metric, "趋势分析")
    if df[metric].dtype.kind not in "biufc":
        raise ValueError(f"趋势分析：指标列 '{metric}' 不是数值类型")

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col])
    if temp_df.empty:
        return f"[趋势分析] 日期列 '{date_col}' 无有效数据。"

    grain = _auto_date_grain(temp_df, date_col)
    grain_label = {"D": "天", "W": "周", "M": "月"}[grain]

    grouped = (
        temp_df.set_index(date_col)[metric]
        .resample(grain)
        .sum()
        .dropna()
    )
    if grouped.empty:
        return f"[趋势分析] 按{grain_label}聚合后无数据。"

    total = float(grouped.sum())
    peak_period = grouped.idxmax()
    peak_value = float(grouped.max())
    valley_period = grouped.idxmin()
    valley_value = float(grouped.min())
    first_value = float(grouped.iloc[0])
    last_value = float(grouped.iloc[-1])
    change_pct = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0.0

    # 生成各周期数据摘要（最多展示 10 个周期）
    period_lines = []
    if len(grouped) <= 10:
        for period, val in grouped.items():
            if grain == "M":
                label = period.strftime("%Y-%m")
            elif grain == "W":
                label = period.strftime("%Y-%m-%d")
            else:
                label = period.strftime("%Y-%m-%d")
            period_lines.append(f"{label}：{val:.0f}")
    else:
        for period, val in list(grouped.items())[:5]:
            if grain == "M":
                label = period.strftime("%Y-%m")
            elif grain == "W":
                label = period.strftime("%Y-%m-%d")
            else:
                label = period.strftime("%Y-%m-%d")
            period_lines.append(f"{label}：{val:.0f}")
        period_lines.append("...")
        for period, val in list(grouped.items())[-5:]:
            if grain == "M":
                label = period.strftime("%Y-%m")
            elif grain == "W":
                label = period.strftime("%Y-%m-%d")
            else:
                label = period.strftime("%Y-%m-%d")
            period_lines.append(f"{label}：{val:.0f}")

    peak_label = peak_period.strftime("%Y-%m-%d") if grain != "M" else peak_period.strftime("%Y-%m")
    valley_label = valley_period.strftime("%Y-%m-%d") if grain != "M" else valley_period.strftime("%Y-%m")

    lines = [
        f"[趋势分析] {metric}（按{grain_label}聚合，共 {len(grouped)} 个周期）：",
        f"- 周期累计：{total:.0f}",
        f"- 峰值：{peak_value:.0f}（{peak_label}）",
        f"- 谷值：{valley_value:.0f}（{valley_label}）",
        f"- 首尾变化：{first_value:.0f} → {last_value:.0f}（{change_pct:+.1f}%）",
        "- 各周期数据：" + "；".join(period_lines),
    ]
    return "\n".join(lines)


def content_type_analysis(df: pd.DataFrame) -> str:
    """
    内容类型占比分析。

    输出不同内容类型的数量和平均播放量。

    参数:
        df: 数据 DataFrame。

    返回:
        内容类型分析结果文本。
    """
    if _CONTENT_TYPE_COL not in df.columns:
        return f"[内容类型占比分析] 未找到内容类型列 '{_CONTENT_TYPE_COL}'。"

    metric = "播放量" if "播放量" in df.columns else None
    type_counts = df[_CONTENT_TYPE_COL].value_counts()
    total = int(type_counts.sum())

    lines = [f"[内容类型占比分析] 共 {total} 条内容，类型分布："]
    for ctype, count in type_counts.items():
        pct = count / total * 100 if total > 0 else 0
        avg_play = ""
        if metric and df[metric].dtype.kind in "biufc":
            avg_val = float(df[df[_CONTENT_TYPE_COL] == ctype][metric].mean())
            avg_play = f"，平均{metric}={avg_val:.0f}"
        lines.append(f"- {ctype}：{count} 条（{pct:.1f}%）{avg_play}")
    return "\n".join(lines)


def top_content_analysis(
    df: pd.DataFrame,
    metric: str = "播放量",
    n: int = 5,
) -> str:
    """
    Top N 内容分析：按指定指标排序，输出前 N 条内容。

    参数:
        df: 数据 DataFrame。
        metric: 排序指标列名，默认 "播放量"。
        n: 返回前 N 条，默认 5。

    返回:
        Top N 内容分析结果文本。
    """
    _validate_column(df, metric, "Top N 分析")
    if df[metric].dtype.kind not in "biufc":
        raise ValueError(f"Top N 分析：指标列 '{metric}' 不是数值类型")

    title_col = _TITLE_COL if _TITLE_COL in df.columns else df.columns[0]
    top_df = df.nlargest(n, metric)

    lines = [f"[Top N 分析] 按 {metric} 排序的前 {len(top_df)} 条内容："]
    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        title = str(row[title_col])
        value = float(row[metric])
        lines.append(f"  {i}. {title} — {metric}：{value:.0f}")
    return "\n".join(lines)


def distribution_analysis(
    df: pd.DataFrame,
    metric: str = "播放量",
) -> str:
    """
    分布分析：箱形图统计信息（四分位数、异常值）。

    使用 IQR 法检测异常值。

    参数:
        df: 数据 DataFrame。
        metric: 待分析指标列名，默认 "播放量"。

    返回:
        分布分析结果文本。
    """
    _validate_column(df, metric, "分布分析")
    if df[metric].dtype.kind not in "biufc":
        raise ValueError(f"分布分析：指标列 '{metric}' 不是数值类型")

    series = df[metric].dropna()
    if series.empty:
        return f"[分布分析] 指标列 '{metric}' 无有效数据。"

    stats = compute_basic_stats(df, metric)
    q1 = stats["q25"]
    q3 = stats["q75"]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers_low = int((series < lower).sum())
    outliers_high = int((series > upper).sum())
    outliers_total = outliers_low + outliers_high

    lines = [
        f"[分布分析] {metric} 的箱形图统计：",
        f"- 最小值：{stats['min']:.2f}",
        f"- Q1（下四分位数）：{q1:.2f}",
        f"- 中位数：{stats['median']:.2f}",
        f"- Q3（上四分位数）：{q3:.2f}",
        f"- 最大值：{stats['max']:.2f}",
        f"- IQR（四分位距）：{iqr:.2f}",
        f"- 异常值下界：{lower:.2f}，上界：{upper:.2f}",
        f"- 异常值总数：{outliers_total} 条（偏低 {outliers_low}，偏高 {outliers_high}）",
    ]
    return "\n".join(lines)


def recommend_content(df: pd.DataFrame) -> str:
    """
    内容推荐分析：基于历史数据模式，生成下一批视频内容的选题建议。

    分析维度：
    1. 高表现内容类型（播放量/点赞率最高的类型）
    2. 热门标题关键词
    3. 发布时间规律
    4. 互动表现好的内容特征

    参数:
        df: 数据 DataFrame。

    返回:
        内容推荐分析结果文本，包含推荐理由和选题建议。
    """
    lines = ["[内容推荐分析] 基于历史数据的选题建议："]

    if _CONTENT_TYPE_COL in df.columns:
        type_group = df.groupby(_CONTENT_TYPE_COL).agg(
            视频数=(_CONTENT_TYPE_COL, "count"),
            平均播放量=("播放量", "mean") if "播放量" in df.columns else None,
            平均点赞率=("点赞数", lambda x: (x / df.loc[x.index, "播放量"]).mean() * 100) if ("点赞数" in df.columns and "播放量" in df.columns) else None,
        ).dropna()

        if "平均播放量" in type_group.columns:
            top_type_play = type_group["平均播放量"].idxmax()
            top_play_value = float(type_group.loc[top_type_play, "平均播放量"])
            lines.append(f"1. 表现最佳类型：{top_type_play}（平均播放量 {top_play_value:.0f}）")

        if "平均点赞率" in type_group.columns:
            top_type_like = type_group["平均点赞率"].idxmax()
            top_like_value = float(type_group.loc[top_type_like, "平均点赞率"])
            lines.append(f"2. 互动最佳类型：{top_type_like}（平均点赞率 {top_like_value:.2f}%）")

    if "播放量" in df.columns and _TITLE_COL in df.columns:
        top_contents = df.nlargest(5, "播放量")
        keywords = []
        for title in top_contents[_TITLE_COL]:
            title = str(title)
            for kw in ["测评", "教程", "分享", "体验", "干货", "技巧", "攻略", "推荐"]:
                if kw in title and kw not in keywords:
                    keywords.append(kw)
        if keywords:
            lines.append(f"3. 热门标题关键词：{', '.join(keywords)}")

    if "发布日期" in df.columns:
        try:
            df_copy = df.copy()
            df_copy["发布日期"] = pd.to_datetime(df_copy["发布日期"], errors="coerce")
            df_copy = df_copy.dropna(subset=["发布日期"])
            if not df_copy.empty:
                df_copy["星期"] = df_copy["发布日期"].dt.dayofweek
                weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
                weekday_counts = df_copy["星期"].value_counts().sort_index()
                best_weekday = weekday_counts.idxmax()
                best_count = int(weekday_counts[best_weekday])
                lines.append(f"4. 发布时间建议：{weekday_map.get(best_weekday, best_weekday)}（发布 {best_count} 条，频率最高）")
        except Exception:
            pass

    if "播放量" in df.columns:
        high_performance = df[df["播放量"] > df["播放量"].quantile(0.75)]
        if len(high_performance) >= 3:
            avg_play = float(high_performance["播放量"].mean())
            avg_like = float(high_performance["点赞数"].mean()) if "点赞数" in df.columns else 0
            like_rate = (avg_like / avg_play * 100) if avg_play > 0 else 0
            lines.append(f"5. 高表现内容基准：播放量 > {int(df['播放量'].quantile(0.75))}，点赞率 {like_rate:.2f}%")

    lines.append("\n推荐选题方向：")
    lines.append("- 延续高表现类型的内容，结合热门关键词")
    lines.append("- 尝试在高频率发布日发布新内容")
    lines.append("- 关注互动率高的内容特征进行选题")

    return "\n".join(lines)


def run_analysis(
    df: pd.DataFrame,
    analysis_type: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    根据分析类型执行对应的统计函数（AI 助手统一调度入口）。

    参数:
        df: 数据 DataFrame。
        analysis_type: 分析类型标识，可选：
            describe / correlation / trend / content_type / top / distribution
        params: 分析参数字典，对应各函数的 keyword 参数。

    返回:
        分析结果文本字符串。

    异常:
        ValueError: 分析类型不支持时抛出。
    """
    params = params or {}

    dispatch = {
        "describe": lambda p: describe_statistics(df),
        "correlation": lambda p: correlation_analysis(df),
        "trend": lambda p: trend_analysis(
            df,
            date_col=p.get("date_col", "发布日期"),
            metric=p.get("metric", "播放量"),
        ),
        "content_type": lambda p: content_type_analysis(df),
        "top": lambda p: top_content_analysis(
            df,
            metric=p.get("metric", "播放量"),
            n=int(p.get("n", 5)),
        ),
        "distribution": lambda p: distribution_analysis(
            df,
            metric=p.get("metric", "播放量"),
        ),
        "content_recommend": lambda p: recommend_content(df),
    }

    if analysis_type not in dispatch:
        raise ValueError(
            f"不支持的分析类型 '{analysis_type}'，可选：{', '.join(dispatch.keys())}"
        )

    return dispatch[analysis_type](params)


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

    tests = [
        ("描述性统计", lambda: describe_statistics(df)),
        ("相关性分析", lambda: correlation_analysis(df)),
        ("趋势分析", lambda: trend_analysis(df)),
        ("内容类型占比", lambda: content_type_analysis(df)),
        ("Top 5 内容", lambda: top_content_analysis(df)),
        ("分布分析", lambda: distribution_analysis(df)),
        ("统一入口 - describe", lambda: run_analysis(df, "describe")),
        ("统一入口 - top n=3", lambda: run_analysis(df, "top", {"n": 3, "metric": "点赞数"})),
    ]

    for name, func in tests:
        print(f"\n{'=' * 60}")
        print(f"测试：{name}")
        print("=" * 60)
        try:
            result = func()
            print(result)
        except Exception as exc:
            logger.error("测试 '%s' 失败: %s", name, exc)
