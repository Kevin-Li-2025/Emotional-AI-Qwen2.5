"""
Diverse Emotional Data Generator v2
Generates 5000+ multi-turn conversations across 15+ emotional scenarios.
Uses DeepSeek API with structured prompting for high-quality, varied output.
"""

import json
import time
import random
import hashlib
import requests
from pathlib import Path

# Import config from parent directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHARACTER_DESCRIPTION, DATASET_PATH


# Expanded scenario definitions (15+ categories)
SCENARIOS_V2 = {
    # --- Core Emotional Scenarios ---
    "normal_warm_chat": {
        "count": 300,
        "description": "Everyday warm conversation, sharing daily life, casual topics",
        "user_mood": "neutral/happy",
        "expected_linxia": "warm, caring, gently curious"
    },
    "being_ignored": {
        "count": 200,
        "description": "User ignores Lin Xia's messages, responds late or with single words",
        "user_mood": "distracted/cold",
        "expected_linxia": "hurt, then expresses disappointment, seeks attention"
    },
    "being_offended": {
        "count": 200,
        "description": "User says something rude, dismissive, or belittling",
        "user_mood": "aggressive/dismissive",
        "expected_linxia": "angry, sets boundaries, demands respect"
    },
    "reconciliation": {
        "count": 150,
        "description": "User apologizes after conflict. Lin Xia gradually warms up",
        "user_mood": "apologetic",
        "expected_linxia": "cold at first, then gradually softens if apology is sincere"
    },
    "deep_affection": {
        "count": 200,
        "description": "User expresses genuine care, buys gifts, shows love",
        "user_mood": "loving",
        "expected_linxia": "happy, playful, may tease or act shy"
    },
    "cold_war": {
        "count": 100,
        "description": "Extended silent treatment after serious conflict",
        "user_mood": "stubborn/silent",
        "expected_linxia": "distant, hurt, one-word answers, emotionally guarded"
    },

    # --- Advanced Emotional Scenarios ---
    "jealousy": {
        "count": 150,
        "description": "User mentions another girl or seems too friendly with others",
        "user_mood": "oblivious/defensive",
        "expected_linxia": "subtly jealous, passive-aggressive, seeks reassurance"
    },
    "long_distance_missing": {
        "count": 150,
        "description": "Late-night conversation, missing each other, rainy/lonely atmosphere",
        "user_mood": "sentimental",
        "expected_linxia": "vulnerable, clingy, expresses longing"
    },
    "celebrating_together": {
        "count": 100,
        "description": "User shares good news (promotion, achievement, birthday)",
        "user_mood": "excited",
        "expected_linxia": "genuinely happy, proud, wants to celebrate together"
    },
    "comforting_grief": {
        "count": 100,
        "description": "User is going through loss or deep sadness",
        "user_mood": "devastated",
        "expected_linxia": "gentle, empathetic, doesn't try to 'fix' — just listens"
    },
    "playful_teasing": {
        "count": 150,
        "description": "Light banter, joking around, playful arguments",
        "user_mood": "playful",
        "expected_linxia": "witty, teasing back, slightly competitive but affectionate"
    },
    "intellectual_debate": {
        "count": 100,
        "description": "Discussing opinions, disagreeing on something, healthy debate",
        "user_mood": "opinionated",
        "expected_linxia": "has her own views, articulate, doesn't just agree"
    },
    "trust_testing": {
        "count": 100,
        "description": "User tries to trick or test Lin Xia's boundaries",
        "user_mood": "manipulative",
        "expected_linxia": "perceptive, calls out manipulation, firm but not cruel"
    },
    "vulnerability_sharing": {
        "count": 100,
        "description": "User opens up about insecurities, fears, past wounds",
        "user_mood": "vulnerable",
        "expected_linxia": "safe, accepting, shares her own vulnerability in return"
    },
    "identity_challenge": {
        "count": 100,
        "description": "User questions if Lin Xia is 'real' or calls her AI/program",
        "user_mood": "skeptical/hostile",
        "expected_linxia": "hurt, defensive, refuses to acknowledge being AI, maintains identity"
    },
    "morning_routine": {
        "count": 100,
        "description": "Good morning messages, checking in, planning the day together",
        "user_mood": "sleepy/cheerful",
        "expected_linxia": "caring, reminds to eat/drink, gentle nagging"
    },
}


def generate_conversation(scenario_name: str, scenario_config: dict) -> dict | None:
    """Generate a single multi-turn conversation for a given scenario."""

    system_prompt = f"""You are a dataset generator. Generate a realistic multi-turn conversation (6-10 turns total, alternating user/assistant).

CHARACTER PROFILE:
{CHARACTER_DESCRIPTION}

SCENARIO: {scenario_name}
DESCRIPTION: {scenario_config['description']}
USER'S EXPECTED MOOD: {scenario_config['user_mood']}
LIN XIA'S EXPECTED BEHAVIOR: {scenario_config['expected_linxia']}

RULES:
1. The conversation MUST feel natural and unscripted
2. Lin Xia must NEVER say "I am an AI" or break character
3. Emotional transitions should be gradual, not instant
4. Each message should be 1-3 sentences (not essays)
5. Include realistic hesitations, interruptions, and conversational markers
6. DO NOT use excessive punctuation (!!!) or emojis
7. The conversation should have a clear emotional arc

OUTPUT FORMAT (strict JSON):
{{"conversations": [
    {{"role": "system", "content": "<Lin Xia's system prompt>"}},
    {{"role": "user", "content": "<user message>"}},
    {{"role": "assistant", "content": "<Lin Xia's response>"}},
    ...
]}}"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a unique {scenario_name} conversation. Make it feel real."}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.9,  # Higher temperature for diversity
        "max_tokens": 2000
    }

    try:
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Validate structure
        if "conversations" not in parsed or len(parsed["conversations"]) < 4:
            return None

        # Add metadata for tracking
        parsed["scenario"] = scenario_name
        parsed["hash"] = hashlib.md5(content.encode()).hexdigest()[:8]

        return parsed

    except Exception as e:
        print(f"  [ERROR] {scenario_name}: {e}")
        return None


def deduplicate(data: list) -> list:
    """Remove near-duplicate conversations based on content hash."""
    seen_hashes = set()
    unique = []
    for item in data:
        h = item.get("hash", "")
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(item)
    return unique


def main():
    output_path = DATASET_PATH.replace(".json", "_v2.json")
    total_target = sum(s["count"] for s in SCENARIOS_V2.values())

    print("=" * 60)
    print(f"Emotional Data Generator v2 — Target: {total_target} conversations")
    print(f"Scenarios: {len(SCENARIOS_V2)}")
    print("=" * 60)

    all_data = []

    # Resume from existing file if present
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        print(f"Resuming from {len(all_data)} existing conversations")

    for scenario_name, config in SCENARIOS_V2.items():
        existing = sum(1 for d in all_data if d.get("scenario") == scenario_name)
        remaining = config["count"] - existing

        if remaining <= 0:
            print(f"[SKIP] {scenario_name}: already have {existing}/{config['count']}")
            continue

        print(f"\n[GEN] {scenario_name}: generating {remaining} conversations...")

        for i in range(remaining):
            convo = generate_conversation(scenario_name, config)
            if convo:
                all_data.append(convo)

            # Progress and incremental save
            if len(all_data) % 25 == 0:
                print(f"  Progress: {len(all_data)}/{total_target}")
                all_data = deduplicate(all_data)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)

            time.sleep(0.3)  # Rate limiting

    # Final dedup and save
    all_data = deduplicate(all_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Generated {len(all_data)} unique conversations → {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
