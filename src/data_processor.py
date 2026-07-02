# -*- coding: utf-8 -*-
"""
数据预处理与上下文摘要生成模块。

负责对原始数据进行清洗、标准化，并生成供 AI 助手使用的
数据摘要，避免向大模型发送全量原始数据。
"""

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# 数值列缺失填充值
_MISSING_NUMERIC_FILL = 0
# 文本列缺失填充值
_MISSING_TEXT_FILL = "未知"
# 缺失标记列后缀（用于还原原始缺失情况，供摘要报告数据质量）
_MISSING_FLAG_SUFFIX = "_缺失标记"
# 日期列名关键词
_DATE_COLUMN_KEYWORDS = ("日期", "date", "time", "时间")
# 去重依据列
_DEDUP_COLUMN = "内容标题"


def _is_date_column(column_name: str) -> bool:
    """
    判断列名是否为日期类型列。

    参数:
        column_name: 列名字符串。

    返回:
        若列名包含日期相关关键词则返回 True。
    """
    name_lower = str(column_name).lower()
    return any(keyword in name_lower for keyword in _DATE_COLUMN_KEYWORDS)


def _convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将日期列转换为 datetime 类型。

    参数:
        df: 待转换的 DataFrame。

    返回:
        转换后的 DataFrame。
    """
    for col in df.columns:
        if _is_date_column(col):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info("日期列转换成功: '%s'", col)
            except Exception as exc:
                logger.warning("日期列 '%s' 转换失败: %s", col, exc)
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将可解析为数值的列转换为 int 或 float。

    仅当原非空值全部可成功转为数值时才执行转换，避免误转文本列。

    参数:
        df: 待转换的 DataFrame。

    返回:
        转换后的 DataFrame。
    """
    for col in df.columns:
        if _is_date_column(col):
            continue
        # 已是数值类型则跳过
        if df[col].dtype.kind in "biufc":
            continue
        # 仅处理 object 类型列
        if df[col].dtype.kind != "O":
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        original_non_null_mask = df[col].notna()
        if original_non_null_mask.sum() == 0:
            continue
        # 原非空值需全部成功转为数值
        if not converted[original_non_null_mask].notna().all():
            continue

        non_null = converted[original_non_null_mask]
        if (non_null % 1 == 0).all():
            # 整数列：含缺失时保留 float，否则转 int
            if converted.isna().any():
                df[col] = converted
            else:
                df[col] = converted.astype(int)
            logger.info("数值列转换成功: '%s' -> 整数", col)
        else:
            df[col] = converted.astype(float)
            logger.info("数值列转换成功: '%s' -> float", col)
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测并处理缺失值。

    数值列缺失填 0 并新增缺失标记列；文本列缺失填 "未知"。

    参数:
        df: 待处理的 DataFrame。

    返回:
        处理后的 DataFrame（可能新增缺失标记列）。
    """
    for col in df.columns:
        if col.endswith(_MISSING_FLAG_SUFFIX):
            continue
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        if df[col].dtype.kind in "biufc":
            # 数值列：填 0 并新增标记列，便于摘要还原真实缺失情况
            flag_col = f"{col}{_MISSING_FLAG_SUFFIX}"
            df[flag_col] = df[col].isna().astype(int)
            df[col] = df[col].fillna(_MISSING_NUMERIC_FILL)
            logger.info(
                "数值列 '%s' 缺失 %d 条，已填 0 并新增标记列 '%s'",
                col, missing_count, flag_col,
            )
        else:
            # 文本列：填 "未知"
            df[col] = df[col].fillna(_MISSING_TEXT_FILL)
            logger.info(
                "文本列 '%s' 缺失 %d 条，已填 '%s'",
                col, missing_count, _MISSING_TEXT_FILL,
            )
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于内容标题列去重，保留首条记录。

    参数:
        df: 待去重的 DataFrame。

    返回:
        去重后的 DataFrame。
    """
    if _DEDUP_COLUMN not in df.columns:
        logger.warning("未找到去重依据列 '%s'，跳过去重步骤。", _DEDUP_COLUMN)
        return df

    before = len(df)
    df = df.drop_duplicates(subset=[_DEDUP_COLUMN], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.info("基于 '%s' 去重，删除重复记录 %d 条。", _DEDUP_COLUMN, removed)
    else:
        logger.info("基于 '%s' 检查去重，无重复记录。", _DEDUP_COLUMN)
    return df


def _log_descriptive_stats(df: pd.DataFrame) -> None:
    """
    输出数值列的描述性统计（均值、中位数、最大值、最小值、标准差）。

    参数:
        df: 已预处理的 DataFrame。
    """
    numeric_cols = [
        col for col in df.columns
        if df[col].dtype.kind in "biufc"
        and not col.endswith(_MISSING_FLAG_SUFFIX)
        and not _is_date_column(col)
    ]
    if not numeric_cols:
        logger.info("无数值列可生成描述性统计。")
        return

    logger.info("描述性统计（数值列）：")
    for col in numeric_cols:
        series = df[col]
        logger.info(
            "  %s: 均值=%.2f, 中位数=%.2f, 最大值=%.2f, 最小值=%.2f, 标准差=%.2f",
            col,
            float(series.mean()),
            float(series.median()),
            float(series.max()),
            float(series.min()),
            float(series.std()),
        )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始数据进行预处理。

    处理流程：数据类型转换 → 缺失值填充与标记 → 去重 → 描述性统计输出。

    参数:
        df: 原始数据 DataFrame。

    返回:
        预处理后的 DataFrame。
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，跳过预处理。")
        return df if df is not None else pd.DataFrame()

    logger.info("开始数据预处理，原始行数: %d", len(df))

    # 1. 数据类型转换（先转换，数值列缺失变为 NaN 便于后续识别）
    df = _convert_date_columns(df)
    df = _convert_numeric_columns(df)

    # 2. 缺失值处理
    df = _handle_missing_values(df)

    # 3. 去重
    df = _deduplicate(df)

    # 4. 描述性统计输出
    _log_descriptive_stats(df)

    logger.info("数据预处理完成，处理后行数: %d", len(df))
    return df


def generate_data_summary(df: pd.DataFrame) -> str:
    """
    生成数据上下文摘要文本，供 AI 助手拼入 Prompt。

    摘要包含：数据集概览、字段清单、核心统计指标、数据质量标注。
    仅包含统计摘要与少量样本（每字段前 3 个示例），不包含全量原始数据，
    以控制大模型 token 消耗。

    参数:
        df: 预处理后的 DataFrame。

    返回:
        纯文本格式的数据摘要字符串。
    """
    if df is None or df.empty:
        return "【数据摘要】当前数据集为空。"

    # 业务字段（排除内部缺失标记列）
    data_cols = [c for c in df.columns if not c.endswith(_MISSING_FLAG_SUFFIX)]
    numeric_cols = [
        c for c in data_cols
        if df[c].dtype.kind in "biufc" and not _is_date_column(c)
    ]
    date_cols = [c for c in data_cols if _is_date_column(c)]

    lines: List[str] = []

    # 一、数据集概览
    lines.append("【数据集概览】")
    lines.append(f"- 行数：{len(df)}")
    lines.append(f"- 列数：{len(data_cols)}")
    for col in date_cols:
        valid = df[col].dropna()
        if not valid.empty:
            lines.append(
                f"- 时间范围（{col}）：{valid.min().strftime('%Y-%m-%d')} "
                f"至 {valid.max().strftime('%Y-%m-%d')}"
            )
    lines.append("")

    # 二、字段清单
    lines.append("【字段清单】")
    for idx, col in enumerate(data_cols, start=1):
        dtype = str(df[col].dtype)
        if _is_date_column(col) and df[col].dtype.kind == "M":
            samples = df[col].dropna().dt.strftime("%Y-%m-%d").unique()[:3]
        else:
            samples = df[col].dropna().astype(str).unique()[:3]
        sample_str = "、".join(samples) if len(samples) > 0 else "无"
        lines.append(f"{idx}. {col}（{dtype}）示例：{sample_str}")
    lines.append("")

    # 三、核心统计指标
    lines.append("【核心统计指标】")
    if numeric_cols:
        for col in numeric_cols:
            series = df[col]
            lines.append(
                f"- {col}：均值={series.mean():.2f}，"
                f"最大值={series.max():.2f}，最小值={series.min():.2f}"
            )
    else:
        lines.append("- 无数值列")
    lines.append("")

    # 四、数据质量标注
    lines.append("【数据质量标注】")
    total_cells = len(df) * len(data_cols)
    missing_total = 0
    quality_lines = []
    for col in data_cols:
        flag_col = f"{col}{_MISSING_FLAG_SUFFIX}"
        if flag_col in df.columns:
            miss = int(df[flag_col].sum())
        else:
            miss = int(df[col].isna().sum())
        if miss > 0:
            rate = miss / len(df) * 100 if len(df) > 0 else 0
            quality_lines.append(f"{col} 缺失 {miss} 条（{rate:.1f}%）")
            missing_total += miss
    overall_rate = (missing_total / total_cells * 100) if total_cells > 0 else 0
    lines.append(f"- 整体缺失率：{overall_rate:.1f}%")
    if quality_lines:
        lines.append("- 各列缺失情况：" + "；".join(quality_lines))
    else:
        lines.append("- 各列缺失情况：无缺失")

    # 异常值检测（IQR 法）
    outlier_lines = []
    for col in numeric_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        if outliers > 0:
            outlier_lines.append(f"{col} 异常 {outliers} 条")
    if outlier_lines:
        lines.append("- 异常值数量（IQR法）：" + "；".join(outlier_lines))
    else:
        lines.append("- 异常值数量（IQR法）：无")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # 确保 src 目录在导入路径中，兼容多种运行方式
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_data

    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    test_files = [
        sample_dir / "sample_content.csv",
        sample_dir / "sample_content.xlsx",
    ]

    for test_file in test_files:
        print(f"\n{'=' * 60}")
        print(f"测试文件: {test_file.name}")
        print("=" * 60)
        try:
            raw_df = load_data(str(test_file))
            processed_df = preprocess_data(raw_df.copy())
            print("\n--- 预处理后前 3 行 ---")
            print(processed_df.head(3).to_string(index=False))
            print("\n--- 数据上下文摘要 ---")
            summary = generate_data_summary(processed_df)
            print(summary)
        except Exception as exc:
            logger.error("处理失败: %s", exc)
