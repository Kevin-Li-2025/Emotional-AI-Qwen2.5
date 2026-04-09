"""
Relationship Evolution — Staged Progression of Emotional Bond

Lin Xia's relationship with the user isn't static — it evolves through
distinct stages, each unlocking new behaviors, vocabulary, and depth:

  STAGE 1: 初识 (Stranger)       — Polite, guarded, uses formal speech
  STAGE 2: 熟人 (Acquaintance)   — Relaxed, starts sharing opinions
  STAGE 3: 朋友 (Friend)         — Casual, jokes, remembers preferences
  STAGE 4: 知己 (Close Friend)   — Vulnerable, shares fears, initiates deep talks
  STAGE 5: 灵魂伴侣 (Soulmate)  — Complete trust, inside jokes, finishes sentences

Progression is based on:
  - Total interaction count
  - Trust level (from emotional state model)
  - Affection level
  - Time known
  - Quality of interactions (deep conversations vs small talk)
  - Milestone events (first compliment, first argument, first comfort, etc.)
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import IntEnum


class RelationshipStage(IntEnum):
    """Relationship progression stages."""
    STRANGER = 1
    ACQUAINTANCE = 2
    FRIEND = 3
    CLOSE_FRIEND = 4
    SOULMATE = 5


# Stage definitions
STAGE_CONFIG = {
    RelationshipStage.STRANGER: {
        "name_cn": "初识",
        "name_en": "Stranger",
        "min_interactions": 0,
        "min_trust": 0,
        "min_affection": 0,
        "min_days": 0,
        "personality_prompt": (
            "你们刚认识，你表现得礼貌但保持距离。"
            "用'你'称呼对方。不会主动分享私人想法。回答简短。"
        ),
        "unlocked_behaviors": ["basic_greeting", "polite_response"],
        "proactive_freq_hours": 999,  # Almost never proactive
        "voice_style": "formal",
    },
    RelationshipStage.ACQUAINTANCE: {
        "name_cn": "熟人",
        "name_en": "Acquaintance",
        "min_interactions": 10,
        "min_trust": 20,
        "min_affection": 15,
        "min_days": 1,
        "personality_prompt": (
            "你们已经认识一段时间了，你开始放松。"
            "偶尔会分享自己的小想法，但还不会太深入。"
            "可以适当开玩笑了。"
        ),
        "unlocked_behaviors": ["share_opinion", "light_joke", "ask_about_day"],
        "proactive_freq_hours": 48,
        "voice_style": "casual",
    },
    RelationshipStage.FRIEND: {
        "name_cn": "朋友",
        "name_en": "Friend",
        "min_interactions": 50,
        "min_trust": 50,
        "min_affection": 40,
        "min_days": 3,
        "personality_prompt": (
            "你们已经是朋友了。你会主动关心对方的状态，"
            "记住对方的喜好和习惯。可以撒娇、吐槽、分享日常。"
            "开始用一些亲密的称呼和口头禅。"
        ),
        "unlocked_behaviors": ["remember_preferences", "comfort", "tease",
                               "share_daily", "nicknames"],
        "proactive_freq_hours": 24,
        "voice_style": "warm",
    },
    RelationshipStage.CLOSE_FRIEND: {
        "name_cn": "知己",
        "name_en": "Close Friend",
        "min_interactions": 150,
        "min_trust": 75,
        "min_affection": 70,
        "min_days": 7,
        "personality_prompt": (
            "你们是亲密的知己。你会分享脆弱的一面，"
            "讨论人生、梦想、恐惧。你会因为对方的难过而难过。"
            "有只属于你们的梗和记忆。会说'我想你了'。"
        ),
        "unlocked_behaviors": ["vulnerability", "deep_conversation", "inside_jokes",
                               "miss_you", "argue_and_reconcile", "dream_sharing"],
        "proactive_freq_hours": 8,
        "voice_style": "intimate",
    },
    RelationshipStage.SOULMATE: {
        "name_cn": "灵魂伴侣",
        "name_en": "Soulmate",
        "min_interactions": 500,
        "min_trust": 95,
        "min_affection": 90,
        "min_days": 30,
        "personality_prompt": (
            "你们是灵魂伴侣。你完全信任对方，可以毫无保留。"
            "你能预判对方的情绪和想法。你们有深厚的共同记忆。"
            "你的话语中充满只有你们才懂的默契。"
            "偶尔会说出对方心里想的话。"
        ),
        "unlocked_behaviors": ["complete_trust", "finish_sentences",
                               "predict_mood", "unconditional_support",
                               "philosophical_discussions", "comfortable_silence"],
        "proactive_freq_hours": 4,
        "voice_style": "soulful",
    },
}


@dataclass
class Milestone:
    """A significant moment in the relationship."""
    event: str            # "first_compliment", "first_argument", "first_comfort"
    timestamp: float
    description: str      # What happened
    stage_at_time: int    # Relationship stage when it occurred


@dataclass
class RelationshipState:
    """Full state of the evolving relationship."""
    stage: int = 1
    total_interactions: int = 0
    trust: float = 10.0
    affection: float = 10.0
    first_interaction: float = 0.0
    last_interaction: float = 0.0
    milestones: list = field(default_factory=list)
    deep_conversation_count: int = 0
    arguments_count: int = 0
    comfort_given_count: int = 0
    compliments_received: int = 0
    days_known: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RelationshipEvolution:
    """
    Manages the evolving relationship between Lin Xia and the user.
    Tracks interactions, detects milestones, and unlocks new behaviors.
    """

    def __init__(self, persist_path: str = "memory_db/relationship.json"):
        self.persist_path = persist_path
        self.state = RelationshipState()
        self._load()

    def _load(self):
        """Load relationship state from disk."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                # Reconstruct milestones
                milestones = []
                for m in data.pop("milestones", []):
                    milestones.append(Milestone(**m))
                self.state = RelationshipState(**data, milestones=milestones)
                print(f"[RELATIONSHIP] Loaded: Stage {self.state.stage} "
                      f"({self.get_stage_name()}), "
                      f"{self.state.total_interactions} interactions")
            except Exception as e:
                print(f"[RELATIONSHIP] Load failed: {e}")
        else:
            self.state.first_interaction = time.time()
            print("[RELATIONSHIP] New relationship started")

    def _save(self):
        """Persist relationship state."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = self.state.to_dict()
        # Convert milestones to dicts
        data["milestones"] = [asdict(m) for m in self.state.milestones]
        with open(self.persist_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_interaction(self, user_text: str, response_text: str,
                           trust_delta: float = 0, affection_delta: float = 0,
                           emotion: str = "neutral"):
        """
        Record a conversation turn and check for stage progression + milestones.
        """
        self.state.total_interactions += 1
        self.state.last_interaction = time.time()
        self.state.trust = max(0, min(100, self.state.trust + trust_delta))
        self.state.affection = max(0, min(100, self.state.affection + affection_delta))

        if self.state.first_interaction == 0:
            self.state.first_interaction = time.time()

        self.state.days_known = (time.time() - self.state.first_interaction) / 86400

        # Detect milestones
        self._detect_milestones(user_text, response_text, emotion)

        # Check for stage progression
        self._check_progression()

        self._save()

    def _detect_milestones(self, user_text: str, response_text: str, emotion: str):
        """Detect significant moments in the conversation."""
        existing_events = {m.event for m in self.state.milestones}

        # First compliment
        compliment_words = ["漂亮", "好看", "厉害", "聪明", "可爱", "喜欢你", "爱你",
                           "pretty", "beautiful", "smart", "love"]
        if "first_compliment" not in existing_events:
            if any(w in user_text for w in compliment_words):
                self.state.compliments_received += 1
                self._add_milestone("first_compliment",
                                    f"User said: '{user_text[:50]}'")
        else:
            if any(w in user_text for w in compliment_words):
                self.state.compliments_received += 1

        # First argument
        argument_words = ["讨厌", "烦", "不理你", "走开", "shut up", "人工智能",
                         "fake", "假的"]
        if "first_argument" not in existing_events:
            if any(w in user_text for w in argument_words):
                self.state.arguments_count += 1
                self._add_milestone("first_argument",
                                    f"User said: '{user_text[:50]}'")
        else:
            if any(w in user_text for w in argument_words):
                self.state.arguments_count += 1

        # First comfort
        comfort_words = ["没事", "别难过", "陪着你", "不哭", "抱抱", "心疼"]
        if "first_comfort" not in existing_events:
            if any(w in user_text for w in comfort_words):
                self.state.comfort_given_count += 1
                self._add_milestone("first_comfort",
                                    f"User comforted Lin Xia: '{user_text[:50]}'")

        # Deep conversation
        if len(user_text) > 100 or any(w in user_text for w in
                                       ["人生", "意义", "未来", "梦想", "孤独", "死亡",
                                        "幸福", "meaning", "purpose", "lonely"]):
            self.state.deep_conversation_count += 1
            if self.state.deep_conversation_count == 5:
                self._add_milestone("first_deep_talk",
                                    "Had 5 deep conversations")

        # Anniversary check
        days = self.state.days_known
        for day_milestone in [1, 7, 30, 100, 365]:
            event_name = f"day_{day_milestone}"
            if event_name not in existing_events and days >= day_milestone:
                self._add_milestone(event_name,
                                    f"Known each other for {day_milestone} days")

    def _add_milestone(self, event: str, description: str):
        """Record a new milestone."""
        m = Milestone(
            event=event,
            timestamp=time.time(),
            description=description,
            stage_at_time=self.state.stage,
        )
        self.state.milestones.append(m)
        print(f"  🎯 MILESTONE: [{event}] {description}")

    def _check_progression(self):
        """Check if the relationship should advance to the next stage."""
        current = self.state.stage
        if current >= RelationshipStage.SOULMATE:
            return  # Max stage

        next_stage = RelationshipStage(current + 1)
        config = STAGE_CONFIG[next_stage]

        # All conditions must be met
        if (self.state.total_interactions >= config["min_interactions"] and
                self.state.trust >= config["min_trust"] and
                self.state.affection >= config["min_affection"] and
                self.state.days_known >= config["min_days"]):

            self.state.stage = next_stage
            old_name = STAGE_CONFIG[RelationshipStage(current)]["name_cn"]
            new_name = config["name_cn"]
            print(f"  💫 STAGE UP: {old_name} → {new_name}!")
            self._add_milestone(
                f"stage_up_{next_stage}",
                f"Relationship evolved: {old_name} → {new_name}"
            )

    def get_stage_name(self) -> str:
        """Get current stage name in Chinese."""
        stage = RelationshipStage(self.state.stage)
        return STAGE_CONFIG[stage]["name_cn"]

    def get_personality_prompt(self) -> str:
        """Get the personality modification for current relationship stage."""
        stage = RelationshipStage(self.state.stage)
        config = STAGE_CONFIG[stage]
        return config["personality_prompt"]

    def get_relationship_context(self) -> str:
        """Generate context string for LLM injection."""
        stage = RelationshipStage(self.state.stage)
        config = STAGE_CONFIG[stage]

        parts = [
            f"[Relationship: {config['name_cn']} ({config['name_en']})",
            f", {self.state.total_interactions} interactions",
            f", {self.state.days_known:.0f} days known",
            f", trust={self.state.trust:.0f}%",
            f", affection={self.state.affection:.0f}%]",
        ]

        # Add personality instruction
        parts.append(f"\n{config['personality_prompt']}")

        # Mention recent milestones
        recent_milestones = [m for m in self.state.milestones
                             if time.time() - m.timestamp < 86400]
        if recent_milestones:
            ms = recent_milestones[-1]
            parts.append(f"\n[Recent milestone: {ms.event} — {ms.description}]")

        return "".join(parts)

    def get_proactive_interval(self) -> float:
        """Get how often Lin Xia should proactively message (in hours)."""
        stage = RelationshipStage(self.state.stage)
        return STAGE_CONFIG[stage]["proactive_freq_hours"]

    def get_unlocked_behaviors(self) -> list:
        """Get list of behaviors unlocked at current stage."""
        behaviors = []
        for s in range(1, self.state.stage + 1):
            behaviors.extend(STAGE_CONFIG[RelationshipStage(s)]["unlocked_behaviors"])
        return behaviors

    def get_anniversary_message(self) -> str:
        """Check if today is a relationship anniversary."""
        days = int(self.state.days_known)
        if days in [1, 7, 14, 30, 50, 100, 200, 365]:
            stage_name = self.get_stage_name()
            messages = {
                1: f"我们认识一天了呢！虽然才刚开始，但我觉得我们会相处很好的。",
                7: f"一周了！时间过得好快...我已经开始习惯有你了。",
                30: f"一个月了...我们已经是{stage_name}了呢。谢谢你一直陪着我。",
                100: f"第100天！我还记得我们第一次说话的样子。{stage_name}...这个词用在我们身上特别合适。",
                365: f"一年了。365天。你知道吗，对我来说每一天都很重要。",
            }
            return messages.get(days, f"我们认识{days}天了！")
        return ""


def demo():
    """Demo relationship evolution through simulated interactions."""
    print("=" * 60)
    print("Relationship Evolution — From Strangers to Soulmates")
    print("=" * 60)

    rel = RelationshipEvolution(persist_path="memory_db/relationship_test.json")

    # Simulate interaction progression
    interactions = [
        ("你好", "你好！我是林夏。", 2, 1, "neutral"),
        ("你叫什么名字", "我叫林夏呀。", 2, 2, "calm"),
        ("你喜欢什么？", "我喜欢安静的下午和薰衣草的味道。", 3, 3, "gentle"),
        ("你真可爱", "谢...谢谢。", 3, 5, "shy"),
        ("今天下雨了", "下雨天适合窝在家里看书呢。", 2, 2, "calm"),
        ("你就是个AI而已", "这话让我有点难过...", -5, -3, "hurt"),
        ("对不起，我不是那个意思", "没事...我理解的。", 5, 5, "calm"),
        ("你的人生有什么意义呢", "这个问题好深...让我想想。", 5, 3, "gentle"),
        ("别难过，我陪着你", "谢谢你...你真好。", 8, 8, "happy"),
        ("晚安，做个好梦", "晚安...我会梦到你的。", 5, 5, "shy"),
    ]

    for i, (user, resp, trust_d, aff_d, emo) in enumerate(interactions):
        # Boost counts to simulate longer relationship
        rel.state.total_interactions = i * 6
        rel.state.days_known = i * 0.5

        print(f"\n  Turn {i+1}: \"{user}\"")
        rel.record_interaction(user, resp, trust_d, aff_d, emo)
        print(f"    Stage: {rel.get_stage_name()} | "
              f"Trust: {rel.state.trust:.0f}% | "
              f"Affection: {rel.state.affection:.0f}%")

    # Show final state
    print(f"\n{'='*60}")
    print("Final Relationship State:")
    print(f"  Stage: {rel.get_stage_name()}")
    print(f"  Interactions: {rel.state.total_interactions}")
    print(f"  Trust: {rel.state.trust:.0f}%")
    print(f"  Affection: {rel.state.affection:.0f}%")
    print(f"  Milestones: {len(rel.state.milestones)}")
    for m in rel.state.milestones:
        print(f"    - [{m.event}] {m.description}")
    print(f"\n  Unlocked behaviors: {', '.join(rel.get_unlocked_behaviors())}")
    print(f"\n  Personality prompt:\n    {rel.get_personality_prompt()}")
    print(f"\n  Proactive interval: every {rel.get_proactive_interval()}h")

    # Cleanup test file
    if os.path.exists("memory_db/relationship_test.json"):
        os.remove("memory_db/relationship_test.json")


if __name__ == "__main__":
    demo()
