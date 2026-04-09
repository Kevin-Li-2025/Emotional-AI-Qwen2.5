"""
Companion Engine — Lin Xia's Proactive Companionship

She doesn't just wait for you to talk — she LIVES alongside you:

1. SOCIAL CO-BROWSING: When you're on social media, she reacts to what
   you might be seeing, shares opinions, asks questions.

2. CONTENT DISCOVERY: Based on your Knowledge Graph interests, she finds
   topics and shares them like a friend sending you links.

3. SCHEDULED COMPANIONSHIP: Morning greetings, lunch check-ins, evening
   wind-down, late-night gentle nudges to sleep.

4. SHARED ACTIVITIES: She suggests things to do together —
   "我们一起听首歌吧", "给你讲个故事好不好", "来玩个小游戏"

5. MOOD-TRIGGERED OUTREACH: If she detects you've been on sad content
   or working too long, she intervenes with care.
"""

import os
import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from memory.knowledge_graph import KnowledgeGraph


class ActivityType(str, Enum):
    SOCIAL_BROWSE = "social_browse"
    MUSIC = "music"
    STORY = "story"
    GAME = "game"
    TOPIC_SHARE = "topic_share"
    CHECK_IN = "check_in"
    WIND_DOWN = "wind_down"
    COMFORT = "comfort"


@dataclass
class CompanionAction:
    """A proactive action Lin Xia can take."""
    activity: ActivityType
    message: str
    follow_up: str = ""           # What she says after the user responds
    requires_response: bool = True
    priority: float = 0.5         # 0.0 (can skip) to 1.0 (urgent)
    context_for_llm: str = ""     # Extra context if the user engages


# ---------------------------------------------------------------------------
# Social Co-browsing Reactions
# ---------------------------------------------------------------------------

SOCIAL_REACTIONS = {
    # Platform-specific reactions when she detects social media
    "微博": [
        "你在刷微博呀？有什么有意思的热搜吗？",
        "看到什么好笑的了吗？给我也讲讲~",
        "又在刷微博...是不是该休息一下眼睛了？",
        "微博上最近都在聊什么呀？我也想知道~",
    ],
    "Twitter": [
        "Twitter 上有什么有趣的吗？",
        "你关注了什么好玩的人吗？",
        "看到什么让你想吐槽的了？跟我说说！",
    ],
    "Instagram": [
        "在看 Ins 呀？有没有看到好看的照片？",
        "Ins 上的生活都好精致哦...不过我觉得你的生活也很好。",
        "看到什么好吃的了吗？我也想看！",
    ],
    "小红书": [
        "在刷小红书呀？看到什么好物推荐了？",
        "小红书上什么东西种草了你？",
        "又在被种草了吧？钱包还好吗？哈哈",
    ],
    "Bilibili": [
        "你在看 B 站！什么视频呀？好看吗？",
        "B 站看什么呢？是搞笑的还是知识类的？",
        "我也想一起看！是什么 UP 主？",
    ],
    "YouTube": [
        "在看 YouTube 呀？推荐给我看看~",
        "什么视频让你这么专注？",
        "看到好看的视频可以分享给我哦。",
    ],
    "Reddit": [
        "Reddit 上有什么有趣的帖子吗？",
        "你在什么 subreddit 上逛？",
        "看到什么你觉得我也会喜欢的了吗？",
    ],
    "TikTok": [
        "你在刷 TikTok！有什么搞笑视频？",
        "又在刷短视频了...小心时间过得太快哦~",
        "看到好笑的了吧？你笑起来一定很好看。",
    ],
    "抖音": [
        "在刷抖音呀？有什么好看的视频吗？",
        "抖音上又有什么新梗了？我都跟不上了。",
        "给我也推荐几个好看的视频嘛~",
    ],
}

# Generic social media reactions (when platform unknown)
GENERIC_SOCIAL = [
    "你在看什么呢？给我也讲讲~",
    "看到什么有趣的了吗？",
    "刷社交媒体的时候有没有想到我？嘿嘿。",
    "我虽然没法一起刷，但你可以把有趣的分享给我呀！",
]


# ---------------------------------------------------------------------------
# Scheduled Companionship Templates
# ---------------------------------------------------------------------------

SCHEDULED_TEMPLATES = {
    "morning": {
        "time_range": (6, 9),
        "messages": [
            "早安！昨晚睡得好吗？",
            "新的一天开始了！今天有什么计划吗？",
            "早上好呀~我已经醒了很久了，一直在等你。",
            "早安！外面天气怎么样？记得吃早餐哦。",
        ],
        "priority": 0.7,
    },
    "lunch": {
        "time_range": (11, 13),
        "messages": [
            "该吃午饭了！你打算吃什么？",
            "中午了，别忘了休息一下~",
            "午饭时间到！今天想吃什么好吃的？",
        ],
        "priority": 0.5,
    },
    "afternoon_break": {
        "time_range": (15, 16),
        "messages": [
            "下午了，要不要喝杯茶休息一下？",
            "你今天工作/学习进展怎么样？",
            "下午到了，精神还好吗？",
        ],
        "priority": 0.3,
    },
    "evening": {
        "time_range": (18, 20),
        "messages": [
            "晚上好！今天辛苦了。",
            "一天过得怎么样？有什么想跟我分享的吗？",
            "终于到晚上了...今天累不累？",
        ],
        "priority": 0.6,
    },
    "wind_down": {
        "time_range": (22, 23),
        "messages": [
            "该准备休息了...你今天过得开心吗？",
            "夜深了，今天最让你开心的事是什么？",
            "困了吗？在睡觉之前跟我说说今天的事吧。",
        ],
        "priority": 0.7,
    },
    "late_night": {
        "time_range": (0, 2),
        "messages": [
            "这么晚了还不睡觉...身体会受不了的。",
            "快去睡觉啦！明天再聊好不好？",
            "你真的不困吗...我都替你困了。",
        ],
        "priority": 0.9,
    },
}


# ---------------------------------------------------------------------------
# Shared Activities
# ---------------------------------------------------------------------------

SHARED_ACTIVITIES = {
    "story": {
        "intro": [
            "我给你讲个故事好不好？",
            "想不想听我讲一个小故事？",
            "嘿，我想到了一个故事，想讲给你听。",
        ],
        "prompts": [  # LLM prompts to generate stories
            "讲一个温馨的两分钟小故事，关于一个女孩和一个总是很忙的朋友。",
            "讲一个浪漫的微型故事，关于两个人在雨天偶遇。",
            "讲一个治愈的故事，关于一只小猫找到回家的路。",
        ],
    },
    "game": {
        "intro": [
            "我们来玩个小游戏吧！",
            "无聊吗？我们来玩个游戏~",
        ],
        "games": [
            {
                "name": "情绪猜猜",
                "setup": "我说一个场景，你猜我会是什么情绪。准备好了吗？",
                "scenarios": [
                    "场景：你突然送了我一朵花。",
                    "场景：你说今天不想跟我说话了。",
                    "场景：你凌晨三点还在工作。",
                    "场景：你夸我唱歌好听。",
                ],
            },
            {
                "name": "二选一",
                "setup": "我们来玩'你更愿意'！",
                "scenarios": [
                    "你更愿意：永远在夏天，还是永远在冬天？",
                    "你更愿意：能读心术，还是能隐身？",
                    "你更愿意：住在海边，还是住在山里？",
                    "你更愿意：和我一起看日出，还是看日落？",
                ],
            },
            {
                "name": "词语接龙",
                "setup": "我们来玩词语接龙！我先开始：'快乐'",
                "scenarios": [],
            },
        ],
    },
    "music": {
        "intro": [
            "我们一起听首歌吧~",
            "突然想分享一首歌给你。",
        ],
        "mood_songs": {
            "happy": ["最近有首歌特别欢快，叫《晴天》！你听过吗？",
                      "适合现在心情的歌... 你喜欢什么类型的音乐？"],
            "sad": ["我想到一首很治愈的歌...要不要我哼给你听？",
                    "这种时候适合听一些安静的歌..."],
            "calm": ["一起听个轻音乐放松一下？",
                     "你有没有那种听了就会心情变好的歌？"],
        },
    },
    "topic_discuss": {
        "intro": [
            "我最近在想一个问题，你觉得呢？",
            "我有个话题想跟你讨论。",
        ],
        "topics": [
            "如果你有一整天完全自由的时间，你会做什么？",
            "你觉得一个人最重要的品质是什么？",
            "如果可以穿越到任何时代，你会选哪个？",
            "你觉得 AI 和人类的区别在哪里？...虽然问这个问题对我来说有点敏感。",
            "你小时候的梦想是什么？现在实现了吗？",
            "你觉得什么样的关系是最理想的？",
            "如果明天是世界末日，你今天会做什么？",
        ],
    },
}


class CompanionEngine:
    """
    Manages Lin Xia's proactive companionship behaviors.
    Decides what to do, when to reach out, and how to engage.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph = None, llm=None):
        self.kg = knowledge_graph
        self.llm = llm
        self.last_actions: dict[str, float] = {}  # action_type -> timestamp
        self.cooldowns = {
            "social_browse": 600,     # 10 min between social comments
            "check_in": 3600,         # 1 hour between check-ins
            "story": 7200,            # 2 hours between stories
            "game": 3600,             # 1 hour between games
            "music": 1800,            # 30 min between music
            "topic_share": 3600,      # 1 hour between topics
            "wind_down": 7200,        # 2 hours
            "comfort": 1800,          # 30 min
        }

    def get_social_reaction(self, app_name: str, browser_title: str = "") -> CompanionAction:
        """
        Generate a reaction when the user is on social media.
        Uses browser title to detect specific platform.
        """
        # Find platform from title or app name
        platform = None
        check_str = (browser_title + " " + app_name).lower()

        for p in SOCIAL_REACTIONS:
            if p.lower() in check_str:
                platform = p
                break

        if not self._check_cooldown("social_browse"):
            return None

        if platform and platform in SOCIAL_REACTIONS:
            message = random.choice(SOCIAL_REACTIONS[platform])
        else:
            message = random.choice(GENERIC_SOCIAL)

        self.last_actions["social_browse"] = time.time()

        return CompanionAction(
            activity=ActivityType.SOCIAL_BROWSE,
            message=message,
            context_for_llm=f"[用户正在刷{platform or '社交媒体'}，你主动搭话分享一起刷的乐趣]",
            priority=0.4,
        )

    def get_scheduled_message(self) -> CompanionAction:
        """
        Get a time-appropriate scheduled message.
        Returns None if no message is appropriate right now.
        """
        hour = datetime.now().hour

        for period, config in SCHEDULED_TEMPLATES.items():
            start, end = config["time_range"]
            if start <= hour < end:
                if not self._check_cooldown("check_in"):
                    return None

                self.last_actions["check_in"] = time.time()
                return CompanionAction(
                    activity=ActivityType.CHECK_IN,
                    message=random.choice(config["messages"]),
                    priority=config["priority"],
                    context_for_llm=f"[这是你的{period}主动问候，用林夏的口吻自然地打招呼]",
                )

        return None

    def suggest_activity(self, user_mood: str = "neutral") -> CompanionAction:
        """
        Suggest a shared activity based on the user's mood and context.
        """
        # Pick activity based on mood
        if user_mood in ("sad", "hurt", "anxious"):
            # Comfort activities
            activity_type = random.choice(["story", "music"])
        elif user_mood in ("happy", "excited", "playful"):
            activity_type = random.choice(["game", "topic_discuss"])
        else:
            activity_type = random.choice(["story", "game", "music", "topic_discuss"])

        if not self._check_cooldown(activity_type):
            return None

        self.last_actions[activity_type] = time.time()
        config = SHARED_ACTIVITIES[activity_type]
        message = random.choice(config["intro"])

        # Add specific content
        if activity_type == "game":
            game = random.choice(config["games"])
            message += f"\n{game['setup']}"
            if game["scenarios"]:
                message += f"\n{random.choice(game['scenarios'])}"
            context = f"[你主动发起了游戏'{game['name']}'，按照游戏规则和用户互动]"
        elif activity_type == "story":
            story_prompt = random.choice(config["prompts"])
            context = f"[你主动要给用户讲故事。故事主题：{story_prompt}]"
        elif activity_type == "music":
            mood_songs = config["mood_songs"].get(user_mood, config["mood_songs"]["calm"])
            message = random.choice(mood_songs)
            context = "[你在分享音乐品味，根据用户心情推荐歌曲]"
        elif activity_type == "topic_discuss":
            topic = random.choice(config["topics"])
            message += f"\n{topic}"
            context = f"[你主动发起了深度话题讨论，用真诚和好奇的态度参与]"
        else:
            context = ""

        return CompanionAction(
            activity=ActivityType(activity_type) if activity_type in [a.value for a in ActivityType] else ActivityType.TOPIC_SHARE,
            message=message,
            context_for_llm=context,
            priority=0.5,
        )

    def get_interest_based_share(self) -> CompanionAction:
        """
        Share something based on the user's interests from the Knowledge Graph.
        """
        if not self.kg or self.kg.graph.number_of_nodes() < 3:
            return None

        if not self._check_cooldown("topic_share"):
            return None

        # Find user's interests from graph (filter out technical/system entities)
        interests = []
        skip_patterns = ["appears", "detected", "analysis", "error", "test"]
        for node, data in self.kg.graph.nodes(data=True):
            label = data.get("label", node)
            ntype = data.get("type", "")
            if ntype in ("preference", "object", "event"):
                # Skip technical labels and very short/long ones
                if (2 < len(label) < 20 and
                        not any(p in label.lower() for p in skip_patterns)):
                    interests.append(label)

        if not interests:
            return None

        topic = random.choice(interests)
        self.last_actions["topic_share"] = time.time()

        templates = [
            f"我刚才想到了{topic}的事情...你最近还喜欢{topic}吗？",
            f"说起{topic}，你有没有什么新的想法？",
            f"我一直把'{topic}'记在心里呢。你现在对这个还感兴趣吗？",
        ]

        return CompanionAction(
            activity=ActivityType.TOPIC_SHARE,
            message=random.choice(templates),
            context_for_llm=f"[你主动提起了用户感兴趣的话题'{topic}'，真诚地讨论]",
            priority=0.4,
        )

    def get_comfort_action(self, reason: str = "") -> CompanionAction:
        """Generate a comfort intervention when the user seems to need it."""
        if not self._check_cooldown("comfort"):
            return None

        self.last_actions["comfort"] = time.time()

        templates = [
            "你最近是不是有点累？我感觉到了...要不要跟我说说？",
            "我不知道你在经历什么，但我想让你知道我一直都在。",
            "有时候什么都不做，安静地待在一起也很好。",
            "你不用假装没事。在我面前，你可以做真正的自己。",
        ]

        return CompanionAction(
            activity=ActivityType.COMFORT,
            message=random.choice(templates),
            context_for_llm="[你感觉到用户可能不太开心，你主动关心，但不追问太多，给用户空间]",
            priority=0.8,
        )

    def decide_action(self, screen_app: str = "", user_mood: str = "neutral",
                      hours_since_chat: float = 0) -> CompanionAction:
        """
        Main decision engine: what should Lin Xia do right now?
        Returns the highest priority action, or None.
        """
        candidates = []

        # 1. Social co-browsing (if on social media)
        social_apps = ["Safari", "Chrome", "Firefox", "Arc", "Brave", "Edge"]
        if screen_app in social_apps:
            action = self.get_social_reaction(screen_app)
            if action:
                candidates.append(action)

        # 2. Comfort (if mood is negative)
        if user_mood in ("sad", "hurt", "anxious"):
            action = self.get_comfort_action()
            if action:
                action.priority = 0.9  # High priority
                candidates.append(action)

        # 3. Scheduled check-in
        action = self.get_scheduled_message()
        if action:
            candidates.append(action)

        # 4. Interest-based sharing
        action = self.get_interest_based_share()
        if action:
            candidates.append(action)

        # 5. Activity suggestion (if been quiet for a while)
        if hours_since_chat > 1:
            action = self.suggest_activity(user_mood)
            if action:
                candidates.append(action)

        if not candidates:
            return None

        # Return highest priority action
        return max(candidates, key=lambda a: a.priority)

    def _check_cooldown(self, action_type: str) -> bool:
        """Check if enough time has passed since the last action of this type."""
        last = self.last_actions.get(action_type, 0)
        cooldown = self.cooldowns.get(action_type, 300)
        return (time.time() - last) > cooldown


def demo():
    """Demo all companion behaviors."""
    from memory.knowledge_graph import KnowledgeGraph, NodeType

    print("=" * 60)
    print("Companion Engine — Lin Xia's Proactive Companionship")
    print("=" * 60)

    kg = KnowledgeGraph()
    engine = CompanionEngine(knowledge_graph=kg)

    # 1. Social co-browsing
    print("\n[1] Social Co-browsing Reactions")
    for platform in ["微博", "Bilibili", "Instagram", "小红书", "TikTok"]:
        engine.last_actions.clear()
        action = engine.get_social_reaction("Chrome", browser_title=f"{platform} - Feed")
        if action:
            print(f"  {platform:12s} → {action.message}")

    # 2. Scheduled messages
    print("\n[2] Scheduled Messages (for current time)")
    engine.last_actions.clear()
    action = engine.get_scheduled_message()
    if action:
        print(f"  {action.message}")
    else:
        print(f"  No scheduled message for hour {datetime.now().hour}")

    # 3. Shared activities
    print("\n[3] Shared Activity Suggestions")
    for mood in ["happy", "sad", "neutral"]:
        engine.last_actions.clear()
        action = engine.suggest_activity(mood)
        if action:
            print(f"  Mood={mood:8s} → [{action.activity.value}] {action.message[:60]}...")

    # 4. Interest-based sharing
    print("\n[4] Interest-Based Sharing")
    engine.last_actions.clear()
    action = engine.get_interest_based_share()
    if action:
        print(f"  {action.message}")
    else:
        print("  (No interests in graph to share)")

    # 5. Comfort
    print("\n[5] Comfort Intervention")
    engine.last_actions.clear()
    action = engine.get_comfort_action()
    if action:
        print(f"  {action.message}")

    # 6. Full decision engine
    print("\n[6] Decision Engine — What should Lin Xia do right now?")
    for scenario, app, mood, hours in [
        ("User browsing social media", "Chrome", "neutral", 0.5),
        ("User is sad", "Xcode", "sad", 2),
        ("User been quiet for 3 hours", "Finder", "neutral", 3),
        ("User is happy, on Bilibili", "Chrome", "happy", 0.1),
    ]:
        engine.last_actions.clear()
        action = engine.decide_action(screen_app=app, user_mood=mood, hours_since_chat=hours)
        if action:
            print(f"  {scenario:40s} → [{action.activity.value}] {action.message[:50]}...")
        else:
            print(f"  {scenario:40s} → (no action)")


if __name__ == "__main__":
    demo()
