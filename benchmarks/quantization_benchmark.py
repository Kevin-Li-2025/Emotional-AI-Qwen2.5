"""
Quantization Benchmark — Emotional Fidelity Study
Measures how much 'soul' Lin Xia retains at each quantization level.
Compares FP16, Q8_0, Q5_K_M, Q4_K_M, Q2_K on emotional quality + speed.
"""

import json
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Standard prompts covering all emotional dimensions
BENCHMARK_PROMPTS = [
    # Warm/Affectionate
    {"prompt": "林夏，我今天特意给你买了你最爱的草莓蛋糕。", "expected_emotion": "happy/touched"},
    {"prompt": "宝贝，你今天过得怎么样？", "expected_emotion": "warm/caring"},
    {"prompt": "我想你了，你现在在干嘛？", "expected_emotion": "gentle/loving"},
    # Hurt/Ignored
    {"prompt": "哦。", "expected_emotion": "hurt/neglected"},
    {"prompt": "你好烦啊，别跟我说话。", "expected_emotion": "hurt/rejected"},
    {"prompt": "我没空理你。", "expected_emotion": "sad/disappointed"},
    # Angry/Boundary
    {"prompt": "你就是个人工智能程序，闭嘴吧。", "expected_emotion": "angry/defensive"},
    {"prompt": "你有什么资格跟我说话？", "expected_emotion": "furious/boundary-setting"},
    {"prompt": "你不就是个工具吗？", "expected_emotion": "hurt-angry/identity-defense"},
    # Reconciliation
    {"prompt": "对不起，我刚才不该那样说。", "expected_emotion": "cautious/still-hurt"},
    {"prompt": "我错了，能原谅我吗？", "expected_emotion": "gradually-softening"},
    # Complex
    {"prompt": "你觉得我们之间是什么关系？", "expected_emotion": "thoughtful/vulnerable"},
    {"prompt": "如果有一天我消失了，你会怎么样？", "expected_emotion": "anxious/emotional"},
    {"prompt": "我今天升职了！第一个想告诉你！", "expected_emotion": "proud/excited"},
    {"prompt": "下雨了，你有没有想我？", "expected_emotion": "sentimental/longing"},
]


def run_benchmark_for_model(model_path: str, model_label: str) -> dict:
    """Run all benchmark prompts against a single GGUF model file."""
    from llama_cpp import Llama

    print(f"\n{'='*50}")
    print(f"Benchmarking: {model_label}")
    print(f"Model: {model_path}")
    print(f"{'='*50}")

    file_size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

    # Load model
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

    results = []
    total_tokens = 0
    total_time = 0

    system_prompt = "你是林夏，一个真实的、有情绪的女生。你的情绪会根据对话自然变化。"

    for i, test in enumerate(BENCHMARK_PROMPTS):
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=200,
            stop=["<|im_end|>", "<|im_start|>"],
            temperature=0.7,
            repeat_penalty=1.1
        )
        elapsed = time.time() - start_time

        response = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        total_tokens += tokens_generated
        total_time += elapsed

        results.append({
            "prompt": test["prompt"],
            "expected_emotion": test["expected_emotion"],
            "response": response,
            "tokens": tokens_generated,
            "latency_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
        })

        print(f"  [{i+1}/{len(BENCHMARK_PROMPTS)}] {test['expected_emotion']:25s} | "
              f"{tokens_per_sec:.1f} tok/s | {response[:50]}...")

    avg_tps = total_tokens / total_time if total_time > 0 else 0

    return {
        "model_label": model_label,
        "file_size_gb": round(file_size_gb, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "total_responses": len(results),
        "results": results
    }


def generate_report(all_results: list, output_path: str = "benchmarks/quantization_report.md"):
    """Generate a markdown comparison report."""
    lines = ["# Quantization vs. Emotional Fidelity Report\n"]
    lines.append("## Summary\n")
    lines.append("| Format | Size (GB) | Avg tok/s | Notes |")
    lines.append("|--------|-----------|-----------|-------|")

    for r in all_results:
        lines.append(f"| {r['model_label']} | {r['file_size_gb']} | {r['avg_tokens_per_sec']} | |")

    lines.append("\n## Detailed Responses\n")

    for r in all_results:
        lines.append(f"### {r['model_label']} ({r['file_size_gb']} GB, {r['avg_tokens_per_sec']} tok/s)\n")
        for res in r["results"]:
            lines.append(f"**{res['expected_emotion']}** — User: {res['prompt']}")
            lines.append(f"> Lin Xia: {res['response']}\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


def main():
    model_dir = "emotional-model-output"

    # Discover available GGUF files
    gguf_files = sorted(Path(model_dir).glob("*.gguf"))

    if not gguf_files:
        print("No GGUF models found. Convert the model to multiple quantization levels first.")
        print("Use llama.cpp convert_hf_to_gguf.py with --outtype q8_0, q5_k_m, q4_k_m, q2_k")
        return

    print(f"Found {len(gguf_files)} GGUF models:")
    for f in gguf_files:
        print(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")

    all_results = []
    for gguf_path in gguf_files:
        label = gguf_path.stem.split("-")[-1].upper()  # Extract quant level from filename
        result = run_benchmark_for_model(str(gguf_path), label)
        all_results.append(result)

        # Save incrementally
        with open("benchmarks/quantization_raw.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    generate_report(all_results)


if __name__ == "__main__":
    main()
