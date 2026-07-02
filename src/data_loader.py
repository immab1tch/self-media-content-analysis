# -*- coding: utf-8 -*-
"""
数据导入与解析模块。

负责从 CSV、Excel 等文件格式加载自媒体账号内容数据，
并提供统一的数据读取接口供后续模块使用。
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def _validate_file_exists(file_path: str) -> Path:
    """
    校验文件是否存在。

    参数:
        file_path: 数据文件路径。

    返回:
        解析后的 Path 对象。

    异常:
        FileNotFoundError: 文件不存在时抛出。
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在，请检查路径是否正确: {file_path}"
        )
    return path


def _detect_format(file_path: Path) -> str:
    """
    根据文件扩展名识别数据格式。

    参数:
        file_path: 文件 Path 对象。

    返回:
        小写扩展名字符串，如 .csv、.xlsx。

    异常:
        ValueError: 文件格式不支持时抛出。
    """
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式 '{ext}'，请使用 CSV (.csv) 或 Excel (.xlsx) 文件。"
        )
    return ext


def _load_csv(file_path: Path) -> pd.DataFrame:
    """
    加载 CSV 格式数据文件。

    参数:
        file_path: CSV 文件 Path 对象。

    返回:
        加载后的 DataFrame。
    """
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        f"无法解析 CSV 文件编码，请确认文件为 UTF-8 或 GBK 编码: {file_path}"
    ) from last_error


def _load_excel(file_path: Path) -> pd.DataFrame:
    """
    加载 Excel (.xlsx) 格式数据文件，多 sheet 时默认读取第一个。

    参数:
        file_path: Excel 文件 Path 对象。

    返回:
        加载后的 DataFrame。
    """
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    if not sheet_names:
        logger.warning(f"警告: Excel 文件 '{file_path}' 不包含任何工作表。")
        return pd.DataFrame()

    if len(sheet_names) > 1:
        logger.warning(
            "Excel 文件包含多个工作表 %s，默认读取第一个工作表: '%s'",
            sheet_names,
            sheet_names[0],
        )

    return pd.read_excel(file_path, sheet_name=sheet_names[0])


def _warn_if_empty(df: pd.DataFrame, file_path: str) -> None:
    """
    对空数据或空表头给出明确警告。

    参数:
        df: 已加载的 DataFrame。
        file_path: 源文件路径，用于日志提示。
    """
    if df.empty:
        logger.warning("警告: 文件 '%s' 未包含任何数据行。", file_path)
        return

    columns = [str(col).strip() for col in df.columns]
    if not columns or all(col == "" for col in columns):
        logger.warning("警告: 文件 '%s' 表头为空，请检查数据格式。", file_path)
        return

    if all(col.startswith("Unnamed:") for col in columns):
        logger.warning(
            "警告: 文件 '%s' 未检测到有效表头，请确认首行是否为字段名。", file_path
        )


def _print_data_info(df: pd.DataFrame, file_path: str) -> None:
    """
    打印数据基本信息。

    参数:
        df: 已加载的 DataFrame。
        file_path: 源文件路径。
    """
    logger.info("成功加载文件: %s", file_path)
    logger.info("  行数: %d", len(df))
    logger.info("  列数: %d", len(df.columns))
    logger.info("  字段名: %s", list(df.columns))


def load_data(file_path: str) -> pd.DataFrame:
    """
    从指定文件路径加载数据。

    参数:
        file_path: 数据文件路径，支持 CSV 与 Excel (.xlsx) 格式。

    返回:
        加载后的 pandas DataFrame。

    异常:
        FileNotFoundError: 文件不存在。
        ValueError: 文件格式不支持或 CSV 编码无法识别。
    """
    path = _validate_file_exists(file_path)
    ext = _detect_format(path)

    if ext == ".csv":
        df = _load_csv(path)
    else:
        df = _load_excel(path)

    _warn_if_empty(df, file_path)
    _print_data_info(df, file_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    test_files = [
        sample_dir / "sample_content.csv",
        sample_dir / "sample_content.xlsx",
    ]

    for test_file in test_files:
        print(f"\n{'=' * 50}")
        print(f"测试文件: {test_file.name}")
        print("=" * 50)
        try:
            result = load_data(str(test_file))
            print(result.head(3).to_string(index=False))
        except (FileNotFoundError, ValueError) as exc:
            logger.error("加载失败: %s", exc)
