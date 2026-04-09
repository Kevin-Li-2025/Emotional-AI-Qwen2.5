"""
Offline DPO Pair Generator — No API Required
Generates preference pairs from existing training data by:
  1. Taking real assistant responses as 'chosen'
  2. Generating synthetic 'rejected' responses using rule-based degradation
     (flattening emotion, breaking character, making responses generic)
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Degradation strategies to create "rejected" responses
DEGRADATION_STRATEGIES = {
    "flatten_emotion": {
        "desc": "Remove all emotional nuance, make response bland",
        "transforms": [
            lambda t: t.replace("！", "。").replace("？", "。").replace("...", "。"),
            lambda t: t.replace("啊", "").replace("呢", "").replace("嘛", "").replace("呀", ""),
            lambda t: t.replace("哇", "").replace("嘻嘻", "").replace("哈哈", ""),
        ]
    },
    "break_character": {
        "desc": "Add AI-like phrases that break the Lin Xia persona",
        "prefixes": [
            "作为AI助手，我认为",
            "我理解你的需求。",
            "好的，我来帮你分析一下。",
            "作为一个人工智能，",
            "请问你还有其他问题吗？",
        ]
    },
    "too_submissive": {
        "desc": "Make response unconditionally agreeable even when it shouldn't be",
        "templates": [
            "好的，你说的都对。没关系的。",
            "嗯嗯，你说什么都行。",
            "没事的，我不介意。你开心就好。",
            "好吧，都听你的。",
        ]
    },
    "too_formal": {
        "desc": "Make response robotic and overly formal",
        "transforms": [
            lambda t: "您好。" + t.replace("你", "您"),
        ]
    },
}


def degrade_response(original: str, strategy: str) -> str:
    """Apply a degradation strategy to create a 'rejected' version."""
    config = DEGRADATION_STRATEGIES[strategy]

    if strategy == "flatten_emotion":
        result = original
        for transform in config["transforms"]:
            result = transform(result)
        # Truncate to make it feel generic
        if len(result) > 30:
            result = result[:30] + "。"
        return result

    elif strategy == "break_character":
        prefix = random.choice(config["prefixes"])
        # Strip emotional content and prepend AI-speak
        cleaned = original.replace("！", "。").replace("？", "。")
        return prefix + cleaned[:40]

    elif strategy == "too_submissive":
        return random.choice(config["templates"])

    elif strategy == "too_formal":
        result = original
        for transform in config["transforms"]:
            result = transform(result)
        return result

    return original


def extract_pairs_from_data(data: list) -> list:
    """Extract DPO preference pairs from existing training conversations."""
    pairs = []
    strategies = list(DEGRADATION_STRATEGIES.keys())

    for convo in data:
        messages = convo.get("conversations", [])

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Get the user prompt (previous message)
            if i == 0 or messages[i-1]["role"] != "user":
                continue

            user_msg = messages[i-1]["content"]
            assistant_msg = msg["content"]

            # Strip any existing emotion tags for clean comparison
            import re
            clean_chosen = re.sub(r'<emotion[^>]*/>', '', assistant_msg).strip()

            if len(clean_chosen) < 5:
                continue

            # Create a rejected version using a random strategy
            strategy = random.choice(strategies)
            rejected = degrade_response(clean_chosen, strategy)

            # Skip if rejected is too similar to chosen
            if rejected == clean_chosen or len(rejected) < 3:
                continue

            pairs.append({
                "prompt": user_msg,
                "chosen": clean_chosen,
                "rejected": rejected,
                "strategy": strategy,
            })

    return pairs


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "emotional_training_data.json"
    output_path = "dpo_preference_data.json"

    print("=" * 60)
    print("Offline DPO Pair Generator (No API Required)")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations from {input_path}")

    pairs = extract_pairs_from_data(data)
    random.shuffle(pairs)

    # Show strategy distribution
    from collections import Counter
    strategy_counts = Counter(p["strategy"] for p in pairs)
    print(f"\nGenerated {len(pairs)} preference pairs:")
    for s, c in strategy_counts.most_common():
        print(f"  {s}: {c}")

    # Show samples
    print(f"\nSample pairs:")
    for p in pairs[:3]:
        print(f"  User: {p['prompt'][:50]}")
        print(f"  Chosen:   {p['chosen'][:60]}")
        print(f"  Rejected: {p['rejected'][:60]}")
        print(f"  Strategy: {p['strategy']}")
        print()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(pairs)} pairs → {output_path}")


if __name__ == "__main__":
    main()
