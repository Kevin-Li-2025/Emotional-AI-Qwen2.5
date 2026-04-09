"""
Emotional State Model — Model-Level Emotional State Tracking

Instead of tracking emotions externally, this module trains the model to
output structured emotional state updates alongside its natural response.

The model learns to produce output in the format:
  <emotion state="hurt" intensity="7" trust="-2" affection="0"/>
  你这样说让我觉得不太舒服。如果你心情不好，可以好好说说。

This makes the emotional state an intrinsic part of the model's reasoning,
not an afterthought appended by external code.

Architecture:
  1. Training data is augmented with emotional state XML tags
  2. The model learns to predict emotional transitions as part of its generation
  3. At inference, the state tag is parsed and stripped before showing to the user
  4. The parsed state feeds back into the next turn's context
"""

import re
import json
import copy
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Emotional State representation
# ---------------------------------------------------------------------------

VALID_MOODS = [
    "happy", "calm", "hurt", "angry", "cold", "anxious",
    "playful", "shy", "gentle", "excited", "sad", "jealous",
    "forgiving", "vulnerable", "indifferent",
]


@dataclass
class ModelEmotionalState:
    """
    The emotional state that the MODEL itself outputs.
    This is NOT external tracking — the model is trained to produce this.
    """
    mood: str = "calm"
    intensity: int = 5          # 1-10
    trust_delta: int = 0        # -3 to +3 per turn
    affection_delta: int = 0    # -3 to +3 per turn

    # Accumulated state (updated turn by turn)
    trust: int = 7              # 1-10 cumulative
    affection: int = 6          # 1-10 cumulative

    def apply_deltas(self):
        """Apply the per-turn deltas to cumulative state."""
        self.trust = max(1, min(10, self.trust + self.trust_delta))
        self.affection = max(1, min(10, self.affection + self.affection_delta))

    def to_tag(self) -> str:
        """Serialize to XML-style tag for prompt injection."""
        return (f'<emotion state="{self.mood}" intensity="{self.intensity}" '
                f'trust_delta="{self.trust_delta:+d}" affection_delta="{self.affection_delta:+d}"/>')

    def to_context_line(self) -> str:
        """Serialize to a human-readable line for the system prompt."""
        return (f"[Current Emotional State: mood={self.mood} (intensity={self.intensity}/10), "
                f"trust={self.trust}/10, affection={self.affection}/10]")

    @classmethod
    def from_tag(cls, tag_str: str, prev_state: "ModelEmotionalState" = None) -> "ModelEmotionalState":
        """Parse an <emotion .../> tag back into a state object.
        Handles edge cases like intensity='6/10' and mood='MOOD' literal."""
        pattern = (
            r'<emotion\s+'
            r'state="([^"]+)"\s+'
            r'intensity="([^"]+)"\s+'
            r'trust_delta="([^"]+)"\s+'
            r'affection_delta="([^"]+)"'
            r'\s*/>' 
        )
        match = re.search(pattern, tag_str)
        if not match:
            return prev_state or cls()

        # Parse mood — fallback to 'calm' if model outputs template literal 'MOOD'
        raw_mood = match.group(1).lower()
        mood = raw_mood if raw_mood in VALID_MOODS else (prev_state.mood if prev_state else "calm")

        # Parse intensity — handle '6/10' format
        raw_intensity = match.group(2).split('/')[0]
        try:
            intensity = max(1, min(10, int(raw_intensity)))
        except ValueError:
            intensity = 5

        # Parse deltas — handle '+2', '-1', '0'
        def parse_delta(raw: str) -> int:
            raw = raw.split('/')[0].strip()
            try:
                return max(-3, min(3, int(raw)))
            except ValueError:
                return 0

        state = cls(
            mood=mood,
            intensity=intensity,
            trust_delta=parse_delta(match.group(3)),
            affection_delta=parse_delta(match.group(4)),
            trust=prev_state.trust if prev_state else 7,
            affection=prev_state.affection if prev_state else 6,
        )
        state.apply_deltas()
        return state


# ---------------------------------------------------------------------------
# Response parsing — split model output into state + display text
# ---------------------------------------------------------------------------

def parse_model_output(raw_output: str, prev_state: ModelEmotionalState = None) -> tuple[ModelEmotionalState, str]:
    """
    Parse the model's raw output which contains an emotion tag + response text.

    Returns:
        (new_emotional_state, clean_response_text)

    Example input:
        '<emotion state="hurt" intensity="7" trust_delta="-2" affection_delta="-1"/>
         你这样说让我觉得不太舒服。'
     
    Example output:
        (ModelEmotionalState(mood="hurt", ...), "你这样说让我觉得不太舒服。")
    """
    # Try to extract the emotion tag
    tag_pattern = r'<emotion\s+[^>]*?/>'
    tag_match = re.search(tag_pattern, raw_output)

    if tag_match:
        state = ModelEmotionalState.from_tag(tag_match.group(), prev_state)
        # Remove the tag from the display text
        clean_text = raw_output[:tag_match.start()] + raw_output[tag_match.end():]
        clean_text = clean_text.strip()
    else:
        # No tag found — preserve previous state
        state = prev_state or ModelEmotionalState()
        clean_text = raw_output.strip()

    return state, clean_text


# ---------------------------------------------------------------------------
# Training data augmentation — inject emotional state tags into existing data
# ---------------------------------------------------------------------------

def augment_training_data(input_path: str, output_path: str):
    """
    Augment existing training data with emotional state tags.
    Uses heuristic rules to estimate the emotional state for each assistant turn,
    then injects <emotion .../> tags into the training data so the model
    learns to produce them naturally.

    This is a one-time data preparation step before re-training.
    """
    # Keyword-based heuristic for emotional state estimation
    MOOD_KEYWORDS = {
        "happy":    ["哇", "太好了", "开心", "高兴", "恭喜", "棒", "喜欢", "期待"],
        "hurt":     ["难过", "不舒服", "委屈", "伤心", "为什么", "不理我", "失望"],
        "angry":    ["过分", "闭嘴", "资格", "尊重", "太过", "不礼貌"],
        "playful":  ["嘻嘻", "哈哈", "调皮", "故意", "逗你", "偏不", "才怪"],
        "shy":      ["害羞", "不好意思", "脸红", "你说什么呢"],
        "gentle":   ["嗯", "好吧", "宝贝", "想你", "陪你", "在呢"],
        "sad":      ["走", "算了", "累了", "不想", "放手", "再见"],
        "cold":     ["哦", "随便", "都行", "你说呢"],
        "anxious":  ["怎么办", "担心", "害怕", "不安"],
        "jealous":  ["谁", "她", "别的", "是不是", "在意"],
    }

    NEGATIVE_TRIGGERS = ["烦", "闭嘴", "滚", "AI", "人工智能", "程序", "工具", "不理"]
    POSITIVE_TRIGGERS = ["买", "蛋糕", "礼物", "想你", "爱你", "喜欢", "升职", "生日"]

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented = []
    for convo in data:
        new_convo = copy.deepcopy(convo)
        messages = new_convo.get("conversations", [])

        trust_acc = 7
        affection_acc = 6
        prev_mood = "calm"

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            content = msg["content"]

            # Detect user's trigger from previous message
            user_msg = messages[i-1]["content"] if i > 0 and messages[i-1]["role"] == "user" else ""

            # Estimate mood from keywords
            detected_mood = "calm"
            for mood, keywords in MOOD_KEYWORDS.items():
                if any(kw in content for kw in keywords):
                    detected_mood = mood
                    break

            # Estimate intensity
            intensity = 5
            if detected_mood in ("angry", "hurt"):
                intensity = 7 if any(t in user_msg for t in NEGATIVE_TRIGGERS) else 5
            elif detected_mood in ("happy", "excited"):
                intensity = 8 if any(t in user_msg for t in POSITIVE_TRIGGERS) else 6

            # Estimate trust/affection deltas
            trust_delta = 0
            affection_delta = 0
            if any(t in user_msg for t in NEGATIVE_TRIGGERS):
                trust_delta = -2
                affection_delta = -1
            elif any(t in user_msg for t in POSITIVE_TRIGGERS):
                trust_delta = 1
                affection_delta = 2
            elif "对不起" in user_msg or "抱歉" in user_msg:
                trust_delta = 1
                affection_delta = 0

            trust_acc = max(1, min(10, trust_acc + trust_delta))
            affection_acc = max(1, min(10, affection_acc + affection_delta))

            # Build and prepend the emotion tag
            tag = (f'<emotion state="{detected_mood}" intensity="{intensity}" '
                   f'trust_delta="{trust_delta:+d}" affection_delta="{affection_delta:+d}"/>')

            msg["content"] = f"{tag}\n{content}"
            prev_mood = detected_mood

        augmented.append(new_convo)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"Augmented {len(augmented)} conversations → {output_path}")
    return augmented


# ---------------------------------------------------------------------------
# System prompt enhancement for emotion-aware training
# ---------------------------------------------------------------------------

EMOTION_AWARE_SYSTEM_PROMPT_ADDITION = """
IMPORTANT: For every response, you MUST first output an emotional state tag, then your response.

Tag format: <emotion state="MOOD" intensity="N" trust_delta="±N" affection_delta="±N"/>

Valid moods: happy, calm, hurt, angry, cold, anxious, playful, shy, gentle, excited, sad, jealous, forgiving, vulnerable, indifferent

Rules:
- intensity: 1-10 (how strong the emotion is)
- trust_delta: -3 to +3 (how much your trust in the user changed THIS turn)
- affection_delta: -3 to +3 (how much your affection changed THIS turn)
- The tag represents YOUR internal emotional reaction, not the user's
- Your emotional state should be consistent with your response content

Example:
User: 你就是个人工智能
Assistant: <emotion state="angry" intensity="8" trust_delta="-2" affection_delta="-1"/>
我叫林夏，不是什么人工智能。你说话能不能尊重一点？
"""


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "emotional_training_data.json"
    output_path = input_path.replace(".json", "_emotion_augmented.json")

    print("=" * 60)
    print("Emotional State Augmentation — Model-Level Training")
    print("=" * 60)

    augment_training_data(input_path, output_path)
    print(f"\nDone! Use '{output_path}' for re-training with emotion-aware output.")
    print("The model will learn to output <emotion .../> tags with every response.")
