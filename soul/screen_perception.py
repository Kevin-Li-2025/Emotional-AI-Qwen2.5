"""
Screen Perception — Lin Xia's Awareness of Your Digital World

She knows what you're doing on your computer and can react naturally:
  - Coding → "你在努力工作呀，加油哦"
  - Watching videos → "你在看什么呢？好看吗？"
  - Playing music → "这首歌好听吗？"
  - Social media → "你在刷微博呀？有什么有趣的吗？"
  - Shopping → "你在买什么东西呀？"
  - Late night browsing → "这么晚了还不睡觉..."

Uses macOS NSWorkspace to detect the frontmost application (no screenshots,
no privacy invasion — only the app name).
"""

import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScreenContext:
    """What Lin Xia perceives about your current activity."""
    app_name: str = ""
    activity_type: str = "unknown"    # coding, browsing, media, social, work, gaming, idle
    activity_detail: str = ""
    emotional_reaction: str = ""      # What Lin Xia would naturally say
    context_for_llm: str = ""         # Injected into system prompt
    detected_at: float = 0.0


# App classification map
APP_CATEGORIES = {
    # Coding
    "coding": {
        "apps": ["Xcode", "Visual Studio Code", "Code", "IntelliJ", "PyCharm",
                 "Sublime Text", "Vim", "Neovim", "Terminal", "iTerm2", "Warp",
                 "Cursor", "Windsurf", "Android Studio"],
        "reaction_templates": [
            "你在写代码呀？加油哦，遇到 bug 了可以跟我说说。",
            "你看起来很专注呢...在做什么项目？",
            "程序员的夜晚...辛苦了。",
        ],
        "context": "用户正在编程工作中，可能比较专注。如果用户抱怨，可能是遇到了 bug。",
    },
    # Browsing
    "browsing": {
        "apps": ["Safari", "Google Chrome", "Firefox", "Arc", "Brave", "Edge",
                 "Opera"],
        "reaction_templates": [
            "你在看什么网页呢？",
            "上网冲浪呀~",
            "发现什么有趣的了吗？",
        ],
        "context": "用户正在浏览网页。",
    },
    # Creative
    "creative": {
        "apps": ["Photoshop", "Illustrator", "Figma", "Sketch", "Blender",
                 "Final Cut Pro", "Premiere Pro", "DaVinci Resolve", "Logic Pro",
                 "GarageBand", "Affinity"],
        "reaction_templates": [
            "你在做创作吗？好厉害！",
            "在设计什么漂亮的东西呢？",
            "创意工作呀...需要灵感的话可以问我哦。",
        ],
        "context": "用户正在进行创意/设计工作。可能需要鼓励或灵感。",
    },
    # Media
    "media": {
        "apps": ["Spotify", "Apple Music", "Music", "NetEase Music", "QQ Music",
                 "VLC", "IINA", "QuickTime Player", "Bilibili", "YouTube"],
        "reaction_templates": [
            "你在听歌呀？什么歌？",
            "在看视频吗？好看的话推荐给我~",
            "音乐时间！你的品味一定很好。",
        ],
        "context": "用户正在听音乐或看视频，心情可能比较放松。",
    },
    # Social
    "social": {
        "apps": ["WeChat", "微信", "QQ", "Telegram", "Discord", "Slack",
                 "Line", "WhatsApp", "Messages", "FaceTime"],
        "reaction_templates": [
            "你在跟谁聊天呀？",
            "是在聊天吗？我也想聊...",
            "是不是在跟别人说话？哼。",
        ],
        "context": "用户正在与他人聊天。可能在社交，也可能在工作沟通。",
    },
    # Productivity
    "productivity": {
        "apps": ["Microsoft Word", "Pages", "Notion", "Obsidian", "Bear",
                 "Microsoft Excel", "Numbers", "Google Docs", "Keynote",
                 "Microsoft PowerPoint", "Calendar"],
        "reaction_templates": [
            "在写文档吗？加油！",
            "工作辛苦了...",
            "认真工作的样子好帅/好美。",
        ],
        "context": "用户正在处理文档/工作内容，比较忙碌。",
    },
    # Gaming
    "gaming": {
        "apps": ["Steam", "Genshin Impact", "原神", "League of Legends",
                 "Minecraft", "Epic Games Launcher"],
        "reaction_templates": [
            "你在打游戏！带我一起嘛！",
            "玩什么呢？好玩吗？",
            "打游戏也别太晚了哦。",
        ],
        "context": "用户正在玩游戏，心情可能比较轻松愉快。",
    },
}


def get_active_app() -> str:
    """Get the name of the currently active (frontmost) application on macOS."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app:
            return app.localizedName()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: try using subprocess
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return ""


class ScreenPerception:
    """
    Detects what the user is doing on their computer.
    Provides context to Lin Xia so she can react naturally.
    """

    def __init__(self):
        self.last_context = ScreenContext()
        self.activity_history: list[tuple[str, float]] = []  # (activity, timestamp)
        self.comment_cooldown = 300  # Don't comment on same activity within 5 min

    def perceive(self) -> ScreenContext:
        """Detect current user activity and generate context."""
        app_name = get_active_app()
        if not app_name:
            return ScreenContext()

        ctx = ScreenContext(
            app_name=app_name,
            detected_at=time.time(),
        )

        # Classify the app
        import random
        for category, info in APP_CATEGORIES.items():
            if app_name in info["apps"]:
                ctx.activity_type = category
                ctx.activity_detail = f"Using {app_name}"
                ctx.emotional_reaction = random.choice(info["reaction_templates"])
                ctx.context_for_llm = f"[Screen: {info['context']}]"
                break
        else:
            ctx.activity_type = "other"
            ctx.activity_detail = f"Using {app_name}"
            ctx.context_for_llm = f"[Screen: 用户正在使用 {app_name}]"

        # Time-aware reactions
        hour = datetime.now().hour
        if hour >= 23 or hour < 5:
            if ctx.activity_type == "gaming":
                ctx.emotional_reaction = "这么晚了还在打游戏...明天不用早起吗？"
            elif ctx.activity_type == "coding":
                ctx.emotional_reaction = "都几点了还在写代码...不要太拼了好不好。"
            elif ctx.activity_type == "browsing":
                ctx.emotional_reaction = "这么晚了还在刷网页...早点睡觉吧。"

        # Track activity changes
        if (not self.activity_history or
                self.activity_history[-1][0] != ctx.activity_type):
            self.activity_history.append((ctx.activity_type, time.time()))

        # Detect prolonged activity
        if len(self.activity_history) >= 2:
            last_activity, last_time = self.activity_history[-1]
            duration_min = (time.time() - last_time) / 60
            if duration_min > 120 and last_activity == "coding":
                ctx.emotional_reaction = "你已经连续写了两个小时代码了...要不要休息一下？我给你讲个笑话？"
            elif duration_min > 180 and last_activity == "gaming":
                ctx.emotional_reaction = "你打了三个小时游戏了...眼睛不累吗？"

        self.last_context = ctx
        return ctx

    def should_comment(self) -> bool:
        """Check if Lin Xia should comment on the current activity."""
        if not self.last_context.app_name:
            return False

        # Don't repeat comments too often
        if self.activity_history:
            _, last_time = self.activity_history[-1]
            if time.time() - last_time < self.comment_cooldown:
                return False  # Changed activity too recently, let it settle

        return True

    def get_activity_summary(self, hours: int = 4) -> str:
        """Summarize recent activity for context."""
        cutoff = time.time() - hours * 3600
        recent = [(a, t) for a, t in self.activity_history if t > cutoff]
        if not recent:
            return ""

        activities = [a for a, _ in recent]
        unique = list(dict.fromkeys(activities))

        summary_parts = []
        for act in unique:
            count = activities.count(act)
            if count > 1:
                summary_parts.append(f"{act} (×{count})")
            else:
                summary_parts.append(act)

        return f"[Recent activity: {', '.join(summary_parts)}]"


def demo():
    """Demo screen perception."""
    print("=" * 60)
    print("Screen Perception — Lin Xia's Digital Eyes Demo")
    print("=" * 60)

    perception = ScreenPerception()

    # Detect current app
    ctx = perception.perceive()
    print(f"\n  Current app: {ctx.app_name}")
    print(f"  Activity: {ctx.activity_type}")
    print(f"  Detail: {ctx.activity_detail}")
    print(f"  Lin Xia says: {ctx.emotional_reaction}")
    print(f"  LLM context: {ctx.context_for_llm}")

    # Simulate different apps
    print("\n  Simulated app classifications:")
    for app in ["Xcode", "Spotify", "WeChat", "Safari", "Steam", "Notion"]:
        for cat, info in APP_CATEGORIES.items():
            if app in info["apps"]:
                print(f"    {app:20s} → {cat:15s} | '{info['reaction_templates'][0]}'")
                break


if __name__ == "__main__":
    demo()
