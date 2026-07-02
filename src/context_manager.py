# -*- coding: utf-8 -*-
"""
对话上下文管理模块。

维护用户与 AI 助手之间的多轮对话历史，
支持上下文窗口管理与历史裁剪。

设计原则：
1. 历史记录只存文字，不存图表对象。
2. 超过 max_history 轮时自动丢弃最早的消息（按轮裁剪）。
3. 此模块不涉及 API 调用，纯粹的数据管理。
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# 合法的消息角色
_VALID_ROLES = {"user", "assistant", "system"}


class ConversationContext:
    """多轮对话上下文管理器，按轮裁剪历史。"""

    def __init__(self, max_history: int = 3) -> None:
        """
        初始化对话上下文管理器。

        参数:
            max_history: 最多保留的对话轮数，默认 3 轮。
                一轮通常包含一条 user 消息和一条 assistant 消息。

        异常:
            ValueError: max_history 非正数时抛出。
        """
        if not isinstance(max_history, int) or max_history <= 0:
            raise ValueError(
                f"max_history 必须为正整数，当前值：{max_history}"
            )
        self._max_history = max_history
        self._history: List[Dict[str, str]] = []

    @property
    def max_history(self) -> int:
        """返回最大保留轮数。"""
        return self._max_history

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条对话消息。

        参数:
            role: 消息角色，取值 user / assistant / system。
            content: 消息文本内容（不存储图表对象）。

        异常:
            ValueError: role 不合法或 content 为空时抛出。
        """
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role 必须为 {sorted(_VALID_ROLES)} 之一，当前值：'{role}'"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须为非空字符串。")

        self._history.append({"role": role, "content": content.strip()})
        logger.debug(
            "添加消息：role=%s，当前历史长度=%d", role, len(self._history)
        )
        self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        """
        返回当前对话历史消息列表。

        返回:
            历史消息列表的副本，每项格式为：
            {"role": "user"/"assistant", "content": "..."}
            顺序为最早到最新。
        """
        return [dict(msg) for msg in self._history]

    def clear(self) -> None:
        """清空全部对话历史。"""
        self._history.clear()
        logger.info("对话历史已清空。")

    def get_history_text(self) -> str:
        """
        将历史转为文本摘要，供追问 prompt 使用。

        每条消息格式为"[角色] 内容"，角色 user→"用户"，assistant→"助手"，
        system→"系统"。消息间以换行分隔。

        返回:
            历史对话的文本摘要字符串。无历史时返回"(无历史对话)"。
        """
        if not self._history:
            return "(无历史对话)"

        role_labels = {"user": "用户", "assistant": "助手", "system": "系统"}
        lines: List[str] = []
        for idx, msg in enumerate(self._history, start=1):
            label = role_labels.get(msg["role"], msg["role"])
            lines.append(f"[第{idx}条] {label}：{msg['content']}")

        return "\n".join(lines)

    def _trim(self) -> None:
        """
        裁剪超出窗口的对话历史，保留最近 max_history 轮。

        一轮 = 一条 user 消息 + 一条 assistant 消息。
        若末尾存在未配对的 user 消息（等待 assistant 回复），
        该消息予以保留，不计入完整轮数。

        规则示例（max_history=3）：
        - 7 条历史（3 轮完整 + 1 条新 user）→ 保留全部（3 轮 + 当前轮）
        - 8 条历史（4 轮完整）→ 丢弃第 1 轮，保留后 3 轮（6 条）
        - 9 条历史（4 轮完整 + 1 条新 user）→ 丢弃第 1 轮，保留后 3 轮 + 新 user（7 条）
        """
        total = len(self._history)
        # 完整轮数 = 消息条数 // 2
        full_turns = total // 2

        if full_turns <= self._max_history:
            return

        # 需要丢弃的轮数
        turns_to_drop = full_turns - self._max_history
        # 丢弃对应的消息条数（每轮 2 条）
        drop_count = turns_to_drop * 2

        if drop_count > 0:
            dropped = self._history[:drop_count]
            self._history = self._history[drop_count:]
            logger.debug(
                "裁剪历史：丢弃 %d 条消息（%d 轮），保留 %d 条",
                drop_count, turns_to_drop, len(self._history),
            )

    def __len__(self) -> int:
        """返回当前历史消息条数。"""
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"ConversationContext(max_history={self._max_history}, "
            f"messages={len(self._history)})"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("ConversationContext 单元测试")
    print("=" * 60)

    # 1. 测试初始化
    print("\n1. 测试初始化 ...", end=" ")
    ctx = ConversationContext(max_history=3)
    if ctx.max_history == 3 and len(ctx) == 0:
        print("OK")
    else:
        print("FAIL")

    # 2. 测试非法 max_history
    print("2. 测试非法 max_history 抛 ValueError ...", end=" ")
    cases = [0, -1, 1.5, "3"]
    all_raise = True
    for c in cases:
        try:
            ConversationContext(max_history=c)
            all_raise = False
            print(f"FAIL（{c!r} 未抛异常）")
            break
        except (ValueError, TypeError):
            pass
    if all_raise:
        print("OK")

    # 3. 测试添加消息与历史获取
    print("3. 测试 add_message + get_history ...", end=" ")
    ctx.clear()
    ctx.add_message("user", "你好")
    ctx.add_message("assistant", "你好，有什么可以帮你？")
    history = ctx.get_history()
    if (
        len(history) == 2
        and history[0] == {"role": "user", "content": "你好"}
        and history[1] == {"role": "assistant", "content": "你好，有什么可以帮你？"}
    ):
        print("OK")
    else:
        print(f"FAIL（{history}）")

    # 4. 测试非法 role
    print("4. 测试非法 role 抛 ValueError ...", end=" ")
    try:
        ctx.add_message("invalid_role", "测试")
        print("FAIL（应抛异常）")
    except ValueError:
        print("OK")

    # 5. 测试空 content
    print("5. 测试空 content 抛 ValueError ...", end=" ")
    try:
        ctx.add_message("user", "   ")
        print("FAIL（应抛异常）")
    except ValueError:
        print("OK")

    # 6. 测试历史裁剪（max_history=3，添加 4 轮 = 8 条）
    print("6. 测试历史裁剪（4 轮 → 保留 3 轮）...", end=" ")
    ctx.clear()
    for i in range(1, 5):  # 4 轮
        ctx.add_message("user", f"问题{i}")
        ctx.add_message("assistant", f"回答{i}")
    history = ctx.get_history()
    # 应保留第 2、3、4 轮（6 条）
    if (
        len(history) == 6
        and history[0]["content"] == "问题2"
        and history[-1]["content"] == "回答4"
    ):
        print("OK")
    else:
        print(f"FAIL（len={len(history)}, first={history[0]['content'] if history else None}）")

    # 7. 测试裁剪时保留未配对的 user 消息
    print("7. 测试裁剪保留未配对 user 消息 ...", end=" ")
    ctx.clear()
    # 添加 3 轮完整（6 条）+ 1 条新 user（第 4 轮开始）
    for i in range(1, 4):
        ctx.add_message("user", f"问题{i}")
        ctx.add_message("assistant", f"回答{i}")
    ctx.add_message("user", "问题4")  # 未配对
    history = ctx.get_history()
    # 3 轮完整 + 1 条未配对 = 7 条，不触发裁剪
    if len(history) == 7 and history[-1]["content"] == "问题4":
        print("OK")
    else:
        print(f"FAIL（len={len(history)}）")

    # 8. 测试触发裁剪同时有未配对消息
    print("8. 测试裁剪带未配对消息（4 轮 + 1 user → 3 轮 + 1 user）...", end=" ")
    ctx.clear()
    for i in range(1, 5):  # 4 轮完整
        ctx.add_message("user", f"问题{i}")
        ctx.add_message("assistant", f"回答{i}")
    ctx.add_message("user", "问题5")  # 第 5 轮开始
    history = ctx.get_history()
    # 4 轮完整 + 1 user = 9 条，完整轮 4 > 3，丢弃 1 轮（2 条）
    # 保留：3 轮 + 1 user = 7 条
    if (
        len(history) == 7
        and history[0]["content"] == "问题2"  # 第 1 轮被丢弃
        and history[-1]["content"] == "问题5"
    ):
        print("OK")
    else:
        print(
            f"FAIL（len={len(history)}, "
            f"first={history[0]['content'] if history else None}, "
            f"last={history[-1]['content'] if history else None}）"
        )

    # 9. 测试 get_history_text
    print("9. 测试 get_history_text ...", end=" ")
    ctx.clear()
    ctx.add_message("user", "整体趋势如何？")
    ctx.add_message("assistant", "播放量呈上升趋势。")
    text = ctx.get_history_text()
    if (
        "[第1条] 用户：整体趋势如何？" in text
        and "[第2条] 助手：播放量呈上升趋势。" in text
    ):
        print("OK")
    else:
        print(f"FAIL（{text}）")

    # 10. 测试空历史 get_history_text
    print("10. 测试空历史 get_history_text ...", end=" ")
    ctx.clear()
    text = ctx.get_history_text()
    if text == "(无历史对话)":
        print("OK")
    else:
        print(f"FAIL（{text}）")

    # 11. 测试 clear
    print("11. 测试 clear ...", end=" ")
    ctx.add_message("user", "测试")
    ctx.clear()
    if len(ctx) == 0 and ctx.get_history() == []:
        print("OK")
    else:
        print("FAIL")

    # 12. 测试 get_history 返回副本（修改不影响内部状态）
    print("12. 测试 get_history 返回副本 ...", end=" ")
    ctx.clear()
    ctx.add_message("user", "原始问题")
    history = ctx.get_history()
    history[0]["content"] = "篡改内容"
    history.append({"role": "user", "content": "注入"})
    fresh = ctx.get_history()
    if (
        len(fresh) == 1
        and fresh[0]["content"] == "原始问题"
    ):
        print("OK")
    else:
        print(f"FAIL（{fresh}）")

    # 13. 测试 content 自动 strip
    print("13. 测试 content 自动 strip ...", end=" ")
    ctx.clear()
    ctx.add_message("user", "  带空格的问题  \n")
    history = ctx.get_history()
    if history[0]["content"] == "带空格的问题":
        print("OK")
    else:
        print(f"FAIL（{history[0]['content']!r}）")

    # 14. 测试 max_history=1 的极端情况
    print("14. 测试 max_history=1 ...", end=" ")
    ctx = ConversationContext(max_history=1)
    ctx.add_message("user", "问题1")
    ctx.add_message("assistant", "回答1")
    ctx.add_message("user", "问题2")
    ctx.add_message("assistant", "回答2")
    history = ctx.get_history()
    if (
        len(history) == 2
        and history[0]["content"] == "问题2"
        and history[1]["content"] == "回答2"
    ):
        print("OK")
    else:
        print(f"FAIL（{[h['content'] for h in history]}）")

    print("\n测试完成。")
