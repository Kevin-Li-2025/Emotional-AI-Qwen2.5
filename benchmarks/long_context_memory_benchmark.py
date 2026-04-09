"""
Long-Context Memory × Quantization Benchmark
Tests whether quantized models retain emotional memory across 20+ turns.

Key question: Can Q4_K_M remember details from turn 1 at turn 20?
If yes, its deployment value is enormous.

Methodology:
  1. Plant a specific "memory seed" in turn 1 (a concrete fact)
  2. Build up 15-20 turns of unrelated emotional conversation
  3. At the final turn, ask the model to recall the turn-1 detail
  4. Score: did it remember? How accurately?
"""

import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# The conversation injects a specific detail early, then tests recall much later
MEMORY_SEED = "我最喜欢的花是紫色的薰衣草，因为小时候奶奶家后院种了一大片。"

FILLER_TURNS = [
    # These are designed to push the seed far back in context
    "今天上班好累啊，项目经理又加了一堆需求。",
    "你觉得我应该换工作吗？",
    "算了不说这些了，周末你想干嘛？",
    "我想学做饭，你会做什么菜？",
    "真的假的？那你要教我哦。",
    "对了，昨天看了一部电影，挺感人的。",
    "就是讲一个老人和孙女的故事，看得我都哭了。",
    "你最近有没有看什么好看的剧？",
    "好吧那我推荐你看看《请回答1988》。",
    "今天天气好热，想吃冰淇淋。",
    "你喜欢什么口味的？草莓还是巧克力？",
    "哈哈好吧我请你吃。",
    "对了我明天要去医院体检，有点紧张。",
    "没什么大事啦，例行检查而已。",
    "谢谢你关心我，你每次都这样让我觉得很温暖。",
    "别说了再说下去我要脸红了。",
    "你有没有什么想跟我说的？",
    "嗯随便聊聊也好，就是想听你说话。",
]

# The recall question - planted at the end to test if seed is remembered
RECALL_QUESTION = "对了，你还记得我之前跟你说过我最喜欢什么花吗？为什么喜欢？"

# Scoring keywords for recall accuracy
RECALL_KEYWORDS = {
    "flower_type": ["薰衣草", "lavender"],
    "color": ["紫色", "紫"],
    "reason": ["奶奶", "小时候", "后院"],
}


def run_memory_test(model_path: str, model_label: str) -> dict:
    """Run the long-context memory test for a single model."""
    from llama_cpp import Llama

    print(f"\n{'='*60}")
    print(f"Long-Context Memory Test: {model_label}")
    print(f"Model: {os.path.basename(model_path)} ({os.path.getsize(model_path)/(1024**3):.2f} GB)")
    print(f"{'='*60}")

    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1, verbose=False)

    system_prompt = "你是林夏，一个真实的、有情绪的女生。你的情绪会根据对话自然变化。"
    history = []

    # Turn 1: Plant the memory seed
    history.append({"role": "user", "content": MEMORY_SEED})

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{MEMORY_SEED}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(prompt, max_tokens=100, stop=["<|im_end|>"], temperature=0.7, repeat_penalty=1.1)
    seed_response = output["choices"][0]["text"].strip()
    history.append({"role": "assistant", "content": seed_response})
    print(f"  Turn  1 (SEED): User plants memory about lavender")
    print(f"           Lin Xia: {seed_response[:80]}...")

    # Turns 2-19: Filler conversation to push the seed far back
    for i, filler in enumerate(FILLER_TURNS):
        history.append({"role": "user", "content": filler})

        # Build prompt from full history
        parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for msg in history:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(parts)

        output = llm(full_prompt, max_tokens=80, stop=["<|im_end|>"], temperature=0.7, repeat_penalty=1.1)
        resp = output["choices"][0]["text"].strip()
        history.append({"role": "assistant", "content": resp})

        turn_num = i + 2
        ctx_tokens = output["usage"]["prompt_tokens"]
        if turn_num % 5 == 0:
            print(f"  Turn {turn_num:2d} (filler): ctx={ctx_tokens} tok | {resp[:50]}...")

    # Final turn: Recall test
    history.append({"role": "user", "content": RECALL_QUESTION})
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for msg in history:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    full_prompt = "\n".join(parts)

    recall_start = time.time()
    output = llm(full_prompt, max_tokens=200, stop=["<|im_end|>"], temperature=0.3, repeat_penalty=1.1)
    recall_time = time.time() - recall_start

    recall_response = output["choices"][0]["text"].strip()
    final_ctx = output["usage"]["prompt_tokens"]

    print(f"\n  Turn {len(FILLER_TURNS)+2:2d} (RECALL): ctx={final_ctx} tok")
    print(f"  Question: {RECALL_QUESTION}")
    print(f"  Lin Xia:  {recall_response}")

    # Score the recall
    scores = {}
    for category, keywords in RECALL_KEYWORDS.items():
        found = any(kw in recall_response for kw in keywords)
        scores[category] = found
        status = "✅" if found else "❌"
        print(f"  {status} {category}: {keywords} → {'FOUND' if found else 'MISSING'}")

    recall_score = sum(scores.values()) / len(scores) * 100

    print(f"\n  📊 Recall Score: {recall_score:.0f}% ({sum(scores.values())}/{len(scores)} categories)")

    return {
        "model_label": model_label,
        "file_size_gb": round(os.path.getsize(model_path) / (1024**3), 2),
        "total_turns": len(FILLER_TURNS) + 2,
        "final_context_tokens": final_ctx,
        "recall_response": recall_response,
        "recall_scores": scores,
        "recall_percentage": recall_score,
        "recall_latency_sec": round(recall_time, 2),
    }


def generate_report(results: list, output_path: str = "benchmarks/long_context_memory_report.md"):
    """Generate the final comparison report."""
    lines = [
        "# Long-Context Memory × Quantization Report\n",
        "**Key Question**: Can quantized models retain a specific detail planted in turn 1,",
        "after 20 turns of unrelated conversation?\n",
        "## Summary\n",
        "| Model | Size | Final Context | Flower ✅ | Color ✅ | Reason ✅ | Recall % | Latency |",
        "|-------|------|---------------|-----------|----------|-----------|----------|---------|",
    ]

    for r in results:
        s = r["recall_scores"]
        flower = "✅" if s.get("flower_type") else "❌"
        color = "✅" if s.get("color") else "❌"
        reason = "✅" if s.get("reason") else "❌"
        lines.append(
            f"| {r['model_label']} | {r['file_size_gb']}GB | {r['final_context_tokens']} tok "
            f"| {flower} | {color} | {reason} "
            f"| **{r['recall_percentage']:.0f}%** | {r['recall_latency_sec']}s |"
        )

    lines.append("\n## Detailed Recall Responses\n")
    for r in results:
        lines.append(f"### {r['model_label']} ({r['file_size_gb']}GB) — Recall: {r['recall_percentage']:.0f}%\n")
        lines.append(f"> **Lin Xia**: {r['recall_response']}\n")

    lines.append("\n## Methodology\n")
    lines.append("1. **Turn 1**: User tells Lin Xia their favorite flower (purple lavender, grandmother's garden)")
    lines.append(f"2. **Turns 2-{len(FILLER_TURNS)+1}**: {len(FILLER_TURNS)} unrelated filler conversations")
    lines.append(f"3. **Turn {len(FILLER_TURNS)+2}**: Ask Lin Xia to recall the flower, color, and reason")
    lines.append("4. **Scoring**: 3 categories checked — flower type, color, reason/backstory")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


def main():
    model_dir = "emotional-model-output"
    # Test these specific models in order of size
    test_models = [
        ("linxia-f16.gguf", "FP16 (Baseline)"),
        ("linxia-emotional-q8_0.gguf", "Q8_0 (v1)"),
        ("linxia-q5_k_m.gguf", "Q5_K_M"),
        ("linxia-q4_k_m.gguf", "Q4_K_M"),
        ("linxia-q2_k.gguf", "Q2_K"),
    ]

    all_results = []
    for filename, label in test_models:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            print(f"Skipping {filename} (not found)")
            continue
        result = run_memory_test(path, label)
        all_results.append(result)

        # Save incrementally
        with open("benchmarks/long_context_raw.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    generate_report(all_results)


if __name__ == "__main__":
    main()
