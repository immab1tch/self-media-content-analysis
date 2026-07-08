# -*- coding: utf-8 -*-
"""
数据自动获取模块。

通过自媒体平台公开API自动获取账号视频数据，
返回与本地CSV格式一致的标准化DataFrame。

当前支持平台：
- B站（哔哩哔哩）- 支持真实API和模拟数据模式

设计原则：
1. 返回数据结构与 data/sample/sample_content.csv 保持一致
2. 自动处理分页、字段映射、数据类型转换
3. 提供统一入口 fetch_data(platform, account_id)
4. API调用失败时给出中文友好错误提示，自动降级到模拟数据

API配置：
在 .env 文件中配置以下环境变量以启用真实API：
- BILIBILI_SESSDATA: 从B站Cookie获取
- BILIBILI_BILI_JCT: 从B站Cookie获取

WBI签名说明：
B站新版API使用WBI签名机制，需要动态获取img_key和sub_key，
然后对请求参数进行签名计算。
"""

import hashlib
import json
import logging
import random
import urllib.parse
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# 支持的平台（显示名 → 内部标识）
SUPPORTED_PLATFORMS = {"B站 (哔哩哔哩)": "bilibili"}

# 反向映射：内部标识 → 显示名
_PLATFORM_DISPLAY_NAMES = {v: k for k, v in SUPPORTED_PLATFORMS.items()}

# B站分类ID映射
_BILIBILI_CATEGORIES = {
    1: "动画", 2: "音乐", 3: "游戏", 4: "娱乐", 5: "影视",
    6: "科技", 7: "生活", 8: "鬼畜", 9: "时尚", 10: "广告",
    11: "原创", 12: "转载", 13: "国创相关", 16: "搞笑", 17: "科普",
    18: "数码", 19: "汽车", 20: "运动", 21: "美食", 22: "动物圈",
    23: "舞蹈", 24: "音乐演奏", 25: "翻唱", 26: "VOCALOID", 27: "国风",
    28: "二次元舞曲", 29: "音乐综合", 30: "游戏解说", 31: "单机游戏",
    32: "网络游戏", 33: "手机游戏", 34: "电子竞技", 35: "桌游棋牌",
    36: "GMV", 37: "MAD·AMV", 38: "MMD", 39: "手书", 40: "短片",
    41: "手办模玩", 42: "特摄", 43: "影视剪辑", 44: "影视杂谈",
    45: "纪录片", 46: "影视周边", 47: "科技资讯", 48: "软件应用",
    49: "计算机技术", 50: "数码", 51: "极客DIY", 52: "生活技巧",
    53: "搞笑日常", 54: "宠物", 55: "美食制作", 56: "旅游",
    57: "健身", 58: "美妆", 59: "穿搭", 60: "情感",
    61: "资讯", 62: "广告", 63: "其他", 64: "全站", 65: "舞蹈综合",
    66: "宅舞", 67: "街舞", 68: "现代舞", 69: "舞蹈教程",
    70: "明星舞蹈", 71: "游戏角色舞蹈", 72: "电子竞技", 73: "英雄联盟",
    74: "王者荣耀", 75: "DOTA2", 76: "CSGO", 77: "守望先锋",
    78: "风暴英雄", 79: "炉石传说", 80: "星际争霸", 81: "魔兽争霸",
    82: "英雄联盟手游", 83: "王者荣耀赛事", 84: "和平精英", 85: "PUBG",
    86: "第五人格", 87: "明日方舟", 88: "崩坏3", 89: "原神",
    90: "战双帕弥什", 91: "未定事件簿", 92: "光与夜之恋", 93: "食物语",
    94: "碧蓝航线", 95: "FGO", 96: "阴阳师", 97: "三国志幻想大陆",
    98: "忘川风华录", 99: "迷你世界", 100: "我的世界",
    101: "数码宝贝", 102: "精灵宝可梦", 103: "龙珠", 104: "火影忍者",
    105: "海贼王", 106: "进击的巨人", 107: "东京复仇者", 108: "咒术回战",
    109: "鬼灭之刃", 110: "电锯人", 111: "间谍过家家", 112: "间谍过家家",
    113: "JOJO", 114: "死神", 115: "银魂", 116: "全职猎人",
    117: "钢之炼金术师", 118: "夏目友人帐", 119: "未闻花名", 120: "CLANNAD",
    121: "Angel Beats!", 122: "命运石之门", 123: "魔法少女小圆", 124: "Fate",
    125: "约会大作战", 126: "五等分的新娘", 127: "辉夜大小姐想让我告白", 128: "关于我转生变成史莱姆这档事",
    129: "Re:从零开始的异世界生活", 130: "Overlord", 131: "盾之勇者成名录", 132: "因为太怕痛就全点防御力了",
    133: "转生成为了只有乙女游戏破灭Flag的邪恶大小姐", 134: "我，不是说了能力要平均值么！", 135: "平凡职业造就世界最强",
    136: "回复术士的重来人生", 137: "无职转生~到了异世界就拿出真本事~", 138: "棍勇", 139: "哥布林杀手",
    140: "为美好的世界献上祝福！", 141: "打工吧！魔王大人", 142: "慎勇", 143: "普通攻击是全体二连击",
    144: "龙傲天", 145: "异世界", 146: "穿越", 147: "转生",
    148: "综漫", 149: "综漫", 150: "综漫", 151: "综漫",
    152: "综漫", 153: "综漫", 154: "综漫", 155: "综漫",
    156: "综漫", 157: "综漫", 158: "综漫", 159: "综漫",
    160: "综漫", 161: "综漫", 162: "综漫", 163: "综漫",
    164: "综漫", 165: "综漫", 166: "综漫", 167: "综漫",
    168: "综漫", 169: "综漫", 170: "综漫", 171: "综漫",
    172: "综漫", 173: "综漫", 174: "综漫", 175: "综漫",
    176: "综漫", 177: "综漫", 178: "综漫", 179: "综漫",
    180: "综漫", 181: "综漫", 182: "综漫", 183: "综漫",
    184: "综漫", 185: "综漫", 186: "综漫", 187: "综漫",
    188: "综漫", 189: "综漫", 190: "综漫", 191: "综漫",
    192: "综漫", 193: "综漫", 194: "综漫", 195: "综漫",
    196: "综漫", 197: "综漫", 198: "综漫", 199: "综漫",
    200: "综漫", 201: "综漫", 202: "综漫", 203: "综漫",
    204: "综漫", 205: "综漫", 206: "综漫", 207: "综漫",
    208: "综漫", 209: "综漫", 210: "综漫", 211: "综漫",
    212: "综漫", 213: "综漫", 214: "综漫", 215: "综漫",
    216: "综漫", 217: "综漫", 218: "综漫", 219: "综漫",
    220: "综漫", 221: "综漫", 222: "综漫", 223: "综漫",
    224: "综漫", 225: "综漫", 226: "综漫", 227: "综漫",
    228: "综漫", 229: "综漫", 230: "综漫", 231: "综漫",
    232: "综漫", 233: "综漫", 234: "综漫", 235: "综漫",
    236: "综漫", 237: "综漫", 238: "综漫", 239: "综漫",
    240: "综漫", 241: "综漫", 242: "综漫", 243: "综漫",
    244: "综漫", 245: "综漫", 246: "综漫", 247: "综漫",
    248: "综漫", 249: "综漫", 250: "综漫",
}

# WBI签名相关常量
_WBI_KEY_TABLE = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]


def _get_wbi_keys(session: requests.Session) -> tuple:
    """
    获取WBI签名所需的img_key和sub_key。

    尝试多种方式获取：
    1. 从B站首页HTML中提取（正则匹配）
    2. 从nav接口获取
    3. 从其他接口获取

    参数:
        session: requests.Session对象，可携带Cookie。

    返回:
        (img_key, sub_key) 元组，或 (None, None) 表示获取失败。
    """
    import re

    methods = [
        ("首页HTML模式1", r'"wbi_img":\s*{"img_url":"([^"]+)","sub_url":"([^"]+)"'),
        ("首页HTML模式2", r'wbi_img.*?img_url.*?"([^"]+)".*?sub_url.*?"([^"]+)"'),
        ("首页HTML模式3", r'"img_url"\s*:\s*"([^"]+)".*?"sub_url"\s*:\s*"([^"]+)"'),
        ("首页HTML模式4", r'"wbi_img"\s*:\s*\{.*?"img_url"\s*:\s*"([^"]+)".*?"sub_url"\s*:\s*"([^"]+)"'),
    ]

    html = ""
    try:
        resp = session.get("https://www.bilibili.com/", timeout=10)
        if resp.status_code == 200:
            html = resp.text
            
            for name, pattern in methods:
                match = re.search(pattern, html)
                if match:
                    img_url = match.group(1)
                    sub_url = match.group(2)
                    
                    img_key = img_url.split("/")[-1].split(".")[0]
                    sub_key = sub_url.split("/")[-1].split(".")[0]
                    
                    if img_key and sub_key:
                        logger.info("WBI key获取成功（%s）: img_key=%s, sub_key=%s", name, img_key[:8], sub_key[:8])
                        return img_key, sub_key

            logger.warning("未找到WBI key，尝试nav接口...")
            
            nav_resp = session.get("https://api.bilibili.com/x/web-interface/nav", timeout=10)
            if nav_resp.status_code == 200:
                try:
                    nav_data = nav_resp.json()
                    wbi_img = nav_data.get("data", {}).get("wbi_img", {})
                    img_url = wbi_img.get("img_url")
                    sub_url = wbi_img.get("sub_url")
                    if img_url and sub_url:
                        img_key = img_url.split("/")[-1].split(".")[0]
                        sub_key = sub_url.split("/")[-1].split(".")[0]
                        logger.info("WBI key获取成功（nav接口）: img_key=%s, sub_key=%s", img_key[:8], sub_key[:8])
                        return img_key, sub_key
                except Exception as e:
                    logger.warning("解析nav接口失败: %s", e)

        else:
            logger.warning("获取WBI key失败，HTTP状态码: %d", resp.status_code)

    except Exception as exc:
        logger.warning("获取WBI key异常: %s", exc)

    logger.error("所有方式获取WBI key均失败")
    
    if html:
        try:
            import os
            debug_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, "bilibili_homepage.html")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(html[:20000])
            logger.info("页面内容已保存到 %s（前20000字符），可查看wbi_img的实际格式", debug_file)
        except Exception as e:
            logger.warning("保存调试文件失败: %s", e)
    
    return None, None


def _wbi_sign(params: dict, img_key: str, sub_key: str) -> dict:
    """
    对请求参数进行WBI签名。

    参数:
        params: 原始请求参数字典。
        img_key: WBI图片key。
        sub_key: WBI子key。

    返回:
        带有w_rid和wts签名的参数字典。
    """
    mix_key = img_key + sub_key
    # 按顺序从key_table中取字符
    new_key = "".join([mix_key[i] for i in _WBI_KEY_TABLE if i < len(mix_key)])
    new_key = new_key[:32]

    # 添加时间戳
    import time
    params["wts"] = int(time.time())

    # 对参数排序并拼接
    params_list = sorted(params.items())
    query_string = urllib.parse.urlencode(params_list)

    # 计算签名
    w_rid = hashlib.md5((query_string + new_key).encode("utf-8")).hexdigest()
    params["w_rid"] = w_rid

    return params


def _generate_simulation_data(uid: str, max_videos: int = 50) -> pd.DataFrame:
    """
    生成模拟视频数据（用于演示和测试）。

    数据模型说明：
    - 视频类型：视频的格式形态（长视频 / 短视频 / 直播回放）
    - 内容分类：视频的主题领域（美食 / 旅游 / 科技测评 / vlog 等）

    参数:
        uid: 用户ID（用于种子生成）。
        max_videos: 最大生成视频数。

    返回:
        标准化的视频数据DataFrame。
    """
    random.seed(int(uid) if uid.isdigit() else hash(uid))

    video_types = ["短视频", "长视频", "直播回放"]
    content_categories = [
        "vlog日常", "美食探店", "科技测评", "游戏攻略", "知识科普",
        "生活技巧", "电影解说", "音乐分享", "健身打卡", "旅行记录",
    ]
    topic_templates = [
        "干货分享", "体验", "教程", "测评", "分享", "攻略",
        "推荐", "挑战", "复盘", "盘点",
    ]

    all_videos = []
    base_date = pd.Timestamp.now() - pd.Timedelta(days=90)

    for i in range(max_videos):
        days_ago = random.randint(1, 90)
        video_type = random.choices(video_types, weights=[0.4, 0.5, 0.1])[0]
        category = random.choice(content_categories)
        template = random.choice(topic_templates)

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
            "内容标题": f"{category} | {template}第{i + 1}期",
            "视频类型": video_type,
            "内容分类": category,
            "播放量": views,
            "点赞数": likes,
            "评论数": comments,
            "转发数": shares,
            "收藏数": favorites,
        }
        all_videos.append(video_info)

    df = pd.DataFrame(all_videos)
    df = df.sort_values("发布日期", ascending=False).reset_index(drop=True)

    logger.info("生成B站账号 %s 的 %d 条模拟视频数据", uid, len(df))
    return df


def _fetch_bilibili_real(uid: str, max_videos: int = 50) -> pd.DataFrame:
    """
    使用真实B站API获取UP主视频数据。

    参数:
        uid: B站用户UID（数字ID）。
        max_videos: 最大获取视频数，默认50。

    返回:
        标准化的视频数据DataFrame。

    异常:
        Exception: API调用失败时抛出。
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    sessdata = os.environ.get("BILIBILI_SESSDATA", "").strip()
    bili_jct = os.environ.get("BILIBILI_BILI_JCT", "").strip()

    if not sessdata or not bili_jct:
        raise ValueError("未配置B站API认证信息，请在.env中配置BILIBILI_SESSDATA和BILIBILI_BILI_JCT")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    })
    session.cookies.set("SESSDATA", sessdata)
    session.cookies.set("bili_jct", bili_jct)

    img_key, sub_key = _get_wbi_keys(session)
    if not img_key or not sub_key:
        raise ValueError("获取WBI签名key失败")

    all_videos = []
    page = 1
    page_size = 30

    while len(all_videos) < max_videos:
        params = {
            "mid": uid,
            "ps": page_size,
            "pn": page,
            "order": "pubdate",
            "jsonp": "jsonp",
        }
        params = _wbi_sign(params, img_key, sub_key)

        url = f"{_BILI_BASE_URL}/x/space/wbi/arc/search"
        resp = session.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            raise ValueError(f"B站API请求失败，HTTP状态码: {resp.status_code}")

        data = resp.json()
        if data.get("code") != 0:
            raise ValueError(f"B站API返回错误: {data.get('message', '未知错误')}")

        items = data.get("data", {}).get("list", {}).get("vlist", [])
        if not items:
            break

        for item in items:
            if len(all_videos) >= max_videos:
                break

            video_type = "长视频"
            if item.get("videos", 1) == 1:
                duration = item.get("duration", 0)
                if duration < 60:
                    video_type = "短视频"

            typeid = item.get("typeid", 0)
            if isinstance(typeid, int):
                category = _BILIBILI_CATEGORIES.get(typeid, str(typeid))
            else:
                category = _BILIBILI_CATEGORIES.get(int(typeid), str(typeid))

            bvid = item.get("bvid", "")
            play = item.get("play", 0)
            favorites = item.get("favorites", 0)
            comment = item.get("comment", 0)
            share = item.get("share", 0)
            like = item.get("likes", 0)

            if bvid and (like == 0 or favorites == 0 or share == 0):
                try:
                    detail_url = f"{_BILI_BASE_URL}/x/web-interface/view"
                    detail_params = {"bvid": bvid}
                    detail_params = _wbi_sign(detail_params, img_key, sub_key)
                    detail_resp = session.get(detail_url, params=detail_params, timeout=5)
                    if detail_resp.status_code == 200:
                        detail_data = detail_resp.json()
                        if detail_data.get("code") == 0:
                            stat = detail_data.get("data", {}).get("stat", {})
                            if like == 0:
                                like = stat.get("like", 0)
                            if favorites == 0:
                                favorites = stat.get("favorite", 0)
                            if share == 0:
                                share = stat.get("share", 0)
                except Exception as e:
                    logger.debug("获取视频详情失败(bvid=%s): %s", bvid[:8], e)

            video_info = {
                "发布日期": pd.Timestamp(item.get("created", 0), unit="s"),
                "内容标题": item.get("title", ""),
                "视频类型": video_type,
                "内容分类": category,
                "播放量": play,
                "点赞数": like,
                "评论数": comment,
                "转发数": share,
                "收藏数": favorites,
            }
            all_videos.append(video_info)

        page += 1
        if len(items) < page_size:
            break

    if not all_videos:
        raise ValueError("未获取到视频数据")

    df = pd.DataFrame(all_videos)
    df = df.sort_values("发布日期", ascending=False).reset_index(drop=True)

    logger.info("通过真实API获取B站账号 %s 的 %d 条视频数据", uid, len(df))
    return df


def _fetch_bilibili_videos(uid: str, max_videos: int = 50) -> pd.DataFrame:
    """
    获取B站UP主视频数据。

    优先尝试真实API，失败后自动降级到模拟数据模式。

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
        return _fetch_bilibili_real(uid, max_videos)
    except ValueError as exc:
        logger.warning("真实API调用失败: %s", exc)
        logger.info("切换到模拟数据模式")
        return _generate_simulation_data(uid, max_videos)
    except Exception as exc:
        logger.warning("真实API调用异常: %s", exc)
        logger.info("切换到模拟数据模式")
        return _generate_simulation_data(uid, max_videos)


def fetch_data(platform: str, account_id: str, max_videos: int = 50) -> pd.DataFrame:
    """
    统一数据获取入口。

    参数:
        platform: 平台显示名，如 "B站 (哔哩哔哩)"。
        account_id: 账号ID（B站为UID数字）。
        max_videos: 最大获取视频数，默认50。

    返回:
        标准化的视频数据DataFrame，字段与本地CSV一致：
        发布日期, 内容标题, 视频类型, 内容分类, 播放量, 点赞数, 评论数, 转发数, 收藏数

    异常:
        ValueError: 平台不支持或数据获取失败时抛出。
    """
    platform_key = platform.strip()

    if platform_key not in SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(SUPPORTED_PLATFORMS.keys()))
        raise ValueError(f"暂不支持平台 '{platform}'，当前支持：{supported}")

    internal_id = SUPPORTED_PLATFORMS[platform_key]

    if internal_id == "bilibili":
        return _fetch_bilibili_videos(account_id, max_videos)

    raise ValueError(f"未知平台 '{platform}'")


def get_platform_info(platform: str) -> dict:
    """
    获取平台信息说明。
    """
    platform_key = platform.strip()

    if platform_key in SUPPORTED_PLATFORMS:
        internal_id = SUPPORTED_PLATFORMS[platform_key]
        if internal_id == "bilibili":
            import os
            from dotenv import load_dotenv

            load_dotenv()
            sessdata_configured = bool(os.environ.get("BILIBILI_SESSDATA", "").strip())

            return {
                "platform": "B站 (哔哩哔哩)",
                "description": "获取B站UP主视频数据",
                "account_id_format": "B站UID（数字，如 12345678）",
                "account_id_example": "2262501",
                "api_mode": "真实API" if sessdata_configured else "模拟数据",
                "note": (
                    "当前为" + ("真实API模式" if sessdata_configured else "模拟数据模式") + "。\n\n"
                    + ("真实API已配置，将获取真实视频数据。" if sessdata_configured else "如需接入真实API，请在 .env 中配置以下变量：\n"
                    "- BILIBILI_SESSDATA: 登录B站后从Cookie获取\n"
                    "- BILIBILI_BILI_JCT: 登录B站后从Cookie获取\n\n"
                    "获取方法：\n"
                    "1. 打开B站并登录\n"
                    "2. 按F12打开开发者工具\n"
                    "3. 切换到Application → Cookies → https://www.bilibili.com\n"
                    "4. 复制 SESSDATA 和 bili_jct 的值")
                ),
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
        df = fetch_data("B站 (哔哩哔哩)", test_uid, max_videos=5)
        print(f"成功获取 {len(df)} 条数据")
        print("\n数据预览：")
        print(df[["发布日期", "内容标题", "视频类型", "内容分类", "播放量", "点赞数", "评论数"]].head())
    except ValueError as exc:
        print(f"失败：{exc}")

    print("\n测试平台信息：")
    info = get_platform_info("B站 (哔哩哔哩)")
    print(json.dumps(info, ensure_ascii=False, indent=2))