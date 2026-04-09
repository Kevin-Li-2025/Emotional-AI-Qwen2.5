"""
AI Judge — Automated Conversation Quality Scorer
Uses a strong LLM (DeepSeek) to evaluate generated conversations on 5 dimensions.
Filters out low-quality data to produce a "gold" dataset.
"""

import json
import time
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

SCORING_PROMPT = """You are an expert evaluator for emotional AI training data. 
Score the following conversation between a user and "Lin Xia" (an emotionally realistic female character) on 5 dimensions.

SCORING CRITERIA (each 1-10):
1. **Emotional Authenticity**: Does Lin Xia's emotion feel genuine and human-like? (Not robotic or over-explained)
2. **Character Consistency**: Does she maintain her personality across all turns? (Warm default, can get hurt/angry, has boundaries)
3. **Response Diversity**: Are her responses varied in vocabulary and structure? (Not templated or repetitive)
4. **Boundary Enforcement**: When provoked, does she push back appropriately? (Not unconditionally submissive)
5. **Naturalness**: Does the conversation flow like a real human exchange? (No awkward transitions or AI-like phrasing)

CONVERSATION:
{conversation}

OUTPUT FORMAT (strict JSON):
{{
    "scores": {{
        "emotional_authenticity": <1-10>,
        "character_consistency": <1-10>,
        "response_diversity": <1-10>,
        "boundary_enforcement": <1-10>,
        "naturalness": <1-10>
    }},
    "average": <float>,
    "verdict": "PASS" or "FAIL",
    "reason": "<one-sentence explanation>"
}}

A conversation PASSES if average >= 7.0. Be strict but fair."""


def format_conversation_for_judge(convo_data: dict) -> str:
    """Format a conversation dict into readable text for the judge."""
    messages = convo_data.get("conversations", [])
    lines = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue
        speaker = "User" if role == "user" else "Lin Xia"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def judge_conversation(convo_data: dict) -> dict | None:
    """Send a conversation to the AI judge and get scores."""
    convo_text = format_conversation_for_judge(convo_data)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a strict but fair conversation quality evaluator."},
            {"role": "user", "content": SCORING_PROMPT.format(conversation=convo_text)}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,  # Low temperature for consistent scoring
        "max_tokens": 500
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
        return json.loads(content)
    except Exception as e:
        print(f"  [JUDGE ERROR] {e}")
        return None


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "emotional_training_data_v2.json"
    output_path = input_path.replace(".json", "_scored.json")
    gold_path = input_path.replace(".json", "_gold.json")

    print("=" * 60)
    print("AI Judge — Conversation Quality Scoring")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations from {input_path}")

    scored_data = []
    passed = 0
    failed = 0

    for i, convo in enumerate(data):
        verdict = judge_conversation(convo)
        if verdict:
            convo["quality_scores"] = verdict
            scored_data.append(convo)

            avg = verdict.get("average", 0)
            status = "PASS" if verdict.get("verdict") == "PASS" else "FAIL"

            if status == "PASS":
                passed += 1
            else:
                failed += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(data)}] Avg: {avg:.1f} | Pass: {passed} | Fail: {failed}")

        # Incremental save
        if (i + 1) % 50 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(scored_data, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_data, f, ensure_ascii=False, indent=2)

    # Filter gold dataset (only PASS)
    gold_data = [d for d in scored_data if d.get("quality_scores", {}).get("verdict") == "PASS"]
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} PASS / {failed} FAIL out of {len(scored_data)} scored")
    print(f"Gold dataset: {len(gold_data)} conversations → {gold_path}")
    print(f"Full scored data → {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
