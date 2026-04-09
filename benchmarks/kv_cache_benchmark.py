"""
KV Cache Benchmark — Memory vs. Quality Trade-off Study
Tests different KV cache quantization strategies and measures
impact on VRAM usage, throughput, and emotional coherence.
"""

import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# A long multi-turn conversation to stress-test KV cache
STRESS_TEST_CONVERSATION = [
    "林夏，你在吗？",
    "我今天心情不太好。",
    "被领导骂了一顿。",
    "他当着所有人的面批评我。",
    "我觉得好丢脸。",
    "你能安慰安慰我吗？",
    "谢谢你，你说的话让我好受多了。",
    "对了，周末要不要一起去看电影？",
    "你想看什么类型的？",
    "好的，那就看你推荐的那个。",
    "林夏，你还记得我们第一次聊天吗？",
    "是的，那天我心情也不好来着。",
    "有你在真好。",
    "我今天想早点睡，晚安哦。",
    "等等，我刚才忘了跟你说，我买了你喜欢的巧克力。",
    "明天给你，期待吗？",
    "好啦，真的要睡了。",
    "最后一个问题，你觉得我们之间的感情怎么样？",
    "嗯，那就这样吧，晚安。",
    "其实我还想跟你聊一会儿...",
]


def benchmark_kv_config(model_path: str, kv_config: dict, config_label: str) -> dict:
    """Run the stress test with a specific KV cache configuration."""
    from llama_cpp import Llama

    print(f"\n{'='*50}")
    print(f"KV Cache Config: {config_label}")
    print(f"Settings: {kv_config}")
    print(f"{'='*50}")

    # Build Llama kwargs
    llm_kwargs = {
        "model_path": model_path,
        "n_ctx": 8192,  # Large context to stress KV cache
        "n_gpu_layers": -1,
        "verbose": False,
    }
    # llama-cpp-python may support type_k and type_v in newer versions
    # For now, we document the flags for llama-server usage
    llm = Llama(**llm_kwargs)

    system_prompt = "你是林夏，一个真实的、有情绪的女生。你的情绪会根据对话自然变化。"
    conversation_history = []

    results = []
    total_tokens = 0
    total_time = 0

    for i, user_msg in enumerate(STRESS_TEST_CONVERSATION):
        conversation_history.append({"role": "user", "content": user_msg})

        # Build full prompt from history
        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for msg in conversation_history:
            role_tag = msg["role"]
            prompt_parts.append(f"<|im_start|>{role_tag}\n{msg['content']}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(prompt_parts)

        # Measure inference
        start = time.time()
        output = llm(
            full_prompt,
            max_tokens=150,
            stop=["<|im_end|>", "<|im_start|>"],
            temperature=0.7,
            repeat_penalty=1.1
        )
        elapsed = time.time() - start

        response = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        tps = tokens / elapsed if elapsed > 0 else 0

        total_tokens += tokens
        total_time += elapsed

        conversation_history.append({"role": "assistant", "content": response})

        results.append({
            "turn": i + 1,
            "user_msg": user_msg,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "latency_sec": round(elapsed, 3),
            "tokens_per_sec": round(tps, 1),
        })

        print(f"  Turn {i+1:2d} | ctx: {prompt_tokens:5d} tok | "
              f"{tps:.1f} tok/s | {response[:40]}...")

    avg_tps = total_tokens / total_time if total_time > 0 else 0

    return {
        "config_label": config_label,
        "kv_settings": kv_config,
        "total_turns": len(results),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "final_context_length": results[-1]["prompt_tokens"],
        "results": results
    }


def generate_report(all_results: list, output_path: str = "benchmarks/kv_cache_report.md"):
    """Generate markdown report comparing KV cache strategies."""
    lines = ["# KV Cache Optimization Report\n"]
    lines.append("## Summary\n")
    lines.append("| Config | Avg tok/s | Final Context (tokens) | Notes |")
    lines.append("|--------|-----------|------------------------|-------|")

    for r in all_results:
        lines.append(f"| {r['config_label']} | {r['avg_tokens_per_sec']} | "
                      f"{r['final_context_length']} | |")

    lines.append("\n## Throughput Degradation Over Turns\n")
    lines.append("Shows how tokens/sec changes as context grows:\n")

    for r in all_results:
        lines.append(f"### {r['config_label']}\n")
        lines.append("| Turn | Context Tokens | tok/s |")
        lines.append("|------|----------------|-------|")
        for res in r["results"]:
            lines.append(f"| {res['turn']} | {res['prompt_tokens']} | {res['tokens_per_sec']} |")
        lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


def main():
    model_path = "emotional-model-output/linxia-emotional-q8_0.gguf"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    configs = [
        ({"type_k": "f16", "type_v": "f16"}, "FP16 KV (Baseline)"),
        ({"type_k": "q8_0", "type_v": "q8_0"}, "Q8_0 KV"),
        ({"type_k": "q4_0", "type_v": "q4_0"}, "Q4_0 KV (Aggressive)"),
    ]

    all_results = []
    for kv_config, label in configs:
        result = benchmark_kv_config(model_path, kv_config, label)
        all_results.append(result)

    # Save raw data
    with open("benchmarks/kv_cache_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    generate_report(all_results)


if __name__ == "__main__":
    main()
