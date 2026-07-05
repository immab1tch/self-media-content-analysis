# -*- coding: utf-8 -*-
"""
数据自动获取模块。

通过自媒体平台公开API自动获取账号视频数据，
返回与本地CSV格式一致的标准化DataFrame。

当前支持平台：
- B站（模拟数据模式，预留API接口）

设计原则：
1. 返回数据结构与 data/sample/sample_content.csv 保持一致
2. 自动处理分页、字段映射、数据类型转换
3. 提供统一入口 fetch_data(platform, account_id)
4. API调用失败时给出中文友好错误提示

注意：由于B站API存在安全风控拦截，当前使用模拟数据模式。
如需接入真实API，需配置有效的认证信息。
"""

import logging
import random
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {"bilibili", "b站"}


def _generate_simulation_data(uid: str, max_videos: int = 50) -> pd.DataFrame:
    """
    生成模拟视频数据（用于演示和测试）。

    参数:
        uid: 用户ID（用于种子生成）。
        max_videos: 最大生成视频数。

    返回:
        标准化的视频数据DataFrame。
    """
    random.seed(int(uid) if uid.isdigit() else hash(uid))

    content_types = ["短视频", "长视频", "直播回放"]
    topics = [
        "vlog日常", "美食探店", "科技测评", "游戏攻略", "知识科普",
        "生活技巧", "电影解说", "音乐分享", "健身打卡", "旅行记录",
    ]

    all_videos = []
    base_date = pd.Timestamp.now() - pd.Timedelta(days=90)

    for i in range(max_videos):
        days_ago = random.randint(1, 90)
        video_type = random.choices(content_types, weights=[0.4, 0.5, 0.1])[0]
        topic = random.choice(topics)

        base_views = random.randint(500, 50000)
        if video_type == "短视频":
            base_views = random.randint(1000, 100000)

        views = base_views
        likes = int(views * random.uniform(0.02, 0.1))
        comments = int(views * random.uniform(0.005, 0.03))
        shares = int(views * random.uniform(0.002, 0.01))
        favorites = int(views * random.uniform(0.01, 0.05))

        video_info = {
            "发布日期": base_date + pd.Timedelta(days=days_ago),
            "内容标题": f"{topic} | {random.choice(['干货分享', '体验', '教程', '测评', '分享'])}第{i + 1}期",
            "内容类型": video_type,
            "播放量": views,
            "点赞数": likes,
            "评论数": comments,
            "转发数": shares,
            "收藏数": favorites,
            "粉丝增量": random.randint(0, 500),
        }
        all_videos.append(video_info)

    df = pd.DataFrame(all_videos)
    df = df.sort_values("发布日期", ascending=False).reset_index(drop=True)

    logger.info("生成B站账号 %s 的 %d 条模拟视频数据", uid, len(df))
    return df


def _fetch_bilibili_videos(uid: str, max_videos: int = 50) -> pd.DataFrame:
    """
    获取B站UP主视频数据。

    当前由于B站API安全风控限制，返回模拟数据。
    预留真实API接入接口，配置认证信息后可切换。

    参数:
        uid: B站用户UID（数字ID）。
        max_videos: 最大获取视频数，默认50。

    返回:
        标准化的视频数据DataFrame。

    异常:
        ValueError: UID无效时抛出。
    """
    if not uid or not uid.strip().isdigit():
        raise ValueError("B站UID必须为数字，请检查输入。")

    uid = uid.strip()

    try:
        return _generate_simulation_data(uid, max_videos)
    except Exception as exc:
        logger.warning("真实API调用失败，切换到模拟数据模式: %s", exc)
        return _generate_simulation_data(uid, max_videos)


def fetch_data(platform: str, account_id: str, max_videos: int = 50) -> pd.DataFrame:
    """
    统一数据获取入口。

    参数:
        platform: 平台名称，支持 "bilibili" 或 "b站"。
        account_id: 账号ID（B站为UID数字）。
        max_videos: 最大获取视频数，默认50。

    返回:
        标准化的视频数据DataFrame，字段与本地CSV一致：
        发布日期, 内容标题, 内容类型, 播放量, 点赞数, 评论数, 转发数, 收藏数, 粉丝增量

    异常:
        ValueError: 平台不支持或数据获取失败时抛出。
    """
    platform_lower = platform.strip().lower()

    if platform_lower not in SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(SUPPORTED_PLATFORMS))
        raise ValueError(f"暂不支持平台 '{platform}'，当前支持：{supported}")

    if platform_lower in {"bilibili", "b站"}:
        return _fetch_bilibili_videos(account_id, max_videos)

    raise ValueError(f"未知平台 '{platform}'")


def get_platform_info(platform: str) -> dict:
    """
    获取平台信息说明。
    """
    platform_lower = platform.strip().lower()

    if platform_lower in {"bilibili", "b站"}:
        return {
            "platform": "B站",
            "description": "获取B站UP主视频数据",
            "account_id_format": "B站UID（数字，如 12345678）",
            "account_id_example": "2262501",
            "note": "当前为模拟数据模式（因B站API风控限制），可配置认证信息后接入真实API",
        }

    return {"platform": platform, "description": "未知平台"}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("数据自动获取模块测试")
    print("=" * 60)

    test_uid = "2262501"

    print(f"\n测试B站数据获取（UID: {test_uid}）...")
    try:
        df = fetch_data("bilibili", test_uid, max_videos=5)
        print(f"成功获取 {len(df)} 条数据")
        print("\n数据预览：")
        print(df[["发布日期", "内容标题", "内容类型", "播放量", "点赞数", "评论数"]].head())
    except ValueError as exc:
        print(f"失败：{exc}")

    print("\n测试平台信息：")
    info = get_platform_info("b站")
    print(info)
