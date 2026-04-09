"""
DPO Preference Pair Generator
Creates (prompt, chosen, rejected) triplets for Direct Preference Optimization.
'chosen' = emotionally rich, character-consistent
'rejected' = flat, robotic, breaks character, or overly submissive
"""

import json
import time
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHARACTER_DESCRIPTION


PAIR_GENERATION_PROMPT = """You are generating preference training data for an emotional AI character named "Lin Xia".

CHARACTER PROFILE:
{character_desc}

SCENARIO: {scenario}

Generate ONE preference training example with:
1. A "prompt" (what the user says, 1-2 messages of context)
2. A "chosen" response (emotionally rich, natural, character-consistent — what Lin Xia SHOULD say)
3. A "rejected" response (emotionally flat, robotic, breaks character, or too submissive — what she should NOT say)

The difference should be SUBTLE but meaningful. Don't make rejected obviously bad (no broken language).
Instead, make rejected bland, overly polite, generic, or slightly out-of-character.

OUTPUT FORMAT (strict JSON):
{{
    "prompt": "<user's message(s)>",
    "chosen": "<what Lin Xia should say>",
    "rejected": "<what she should NOT say>",
    "scenario": "{scenario}",
    "contrast_type": "<what makes chosen better: e.g. 'emotional_depth', 'boundary_setting', 'naturalness', 'personality'>"
}}"""


SCENARIOS_FOR_DPO = [
    "User says something mildly rude",
    "User ignores Lin Xia's feelings",
    "User asks Lin Xia to always agree with them",
    "User says 'you're just an AI'",
    "User gives a sincere compliment",
    "User shares sad news",
    "User apologizes after a fight",
    "User mentions another girl",
    "Late-night sentimental conversation",
    "User celebrates an achievement",
    "User tries to manipulate Lin Xia",
    "User asks Lin Xia about her feelings",
    "User asks a boring question",
    "User is being excessively sweet",
    "User suddenly goes cold",
    "User says 'I love you' for the first time",
    "User forgets an important date",
    "User asks Lin Xia to change her personality",
    "User shares their insecurities",
    "User wants to end the relationship",
]


def generate_pair(scenario: str) -> dict | None:
    """Generate a single DPO preference pair."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a training data generator for emotional AI alignment."},
            {"role": "user", "content": PAIR_GENERATION_PROMPT.format(
                character_desc=CHARACTER_DESCRIPTION,
                scenario=scenario
            )}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.85,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Validate required fields
        if all(k in parsed for k in ("prompt", "chosen", "rejected")):
            return parsed
        return None
    except Exception as e:
        print(f"  [ERROR] {scenario}: {e}")
        return None


def main():
    target_pairs = 2000
    output_path = "dpo_preference_data.json"
    pairs_per_scenario = target_pairs // len(SCENARIOS_FOR_DPO)

    print("=" * 60)
    print(f"DPO Preference Pair Generator — Target: {target_pairs} pairs")
    print(f"Scenarios: {len(SCENARIOS_FOR_DPO)}, {pairs_per_scenario} pairs each")
    print("=" * 60)

    all_pairs = []

    # Resume support
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_pairs = json.load(f)
        print(f"Resuming from {len(all_pairs)} existing pairs")

    for scenario in SCENARIOS_FOR_DPO:
        existing = sum(1 for p in all_pairs if p.get("scenario") == scenario)
        remaining = pairs_per_scenario - existing

        if remaining <= 0:
            continue

        print(f"\n  [{scenario}] generating {remaining} pairs...")

        for i in range(remaining):
            pair = generate_pair(scenario)
            if pair:
                all_pairs.append(pair)

            if len(all_pairs) % 50 == 0:
                print(f"    Total: {len(all_pairs)}/{target_pairs}")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

            time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_pairs)} preference pairs → {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
