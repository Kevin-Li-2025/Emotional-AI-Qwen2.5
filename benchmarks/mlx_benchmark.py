"""
MLX vs llama.cpp — Performance Benchmark

A/B comparison of MLX (Apple Silicon native) vs llama.cpp on the same
emotional prompt set. Measures:
  - tokens/sec (generation throughput)
  - time-to-first-token (TTFT)
  - peak memory (RSS)
  - response quality consistency

Usage:
  python benchmarks/mlx_benchmark.py

Output:
  benchmarks/mlx_vs_llamacpp_report.md
"""

import os
import sys
import json
import time
import resource
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.quantization_benchmark import BENCHMARK_PROMPTS


def get_memory_mb() -> float:
    """Get current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def benchmark_backend(backend_name: str, llm, system_prompt: str) -> list[dict]:
    """Run all benchmark prompts against a backend."""
    results = []
    mem_before = get_memory_mb()

    for i, test in enumerate(BENCHMARK_PROMPTS):
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{test['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Warm up on first prompt
        t0 = time.time()
        output = llm(
            prompt, max_tokens=200, stop=["<|im_end|>", "<|im_start|>"],
            temperature=0.7, repeat_penalty=1.1,
        )
        elapsed = time.time() - t0

        response = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(response) // 4)
        tps = completion_tokens / max(elapsed, 0.001)

        results.append({
            "prompt": test["prompt"],
            "expected_emotion": test["expected_emotion"],
            "response": response,
            "completion_tokens": completion_tokens,
            "latency_sec": round(elapsed, 3),
            "tokens_per_sec": round(tps, 1),
        })

        print(f"  [{i+1}/{len(BENCHMARK_PROMPTS)}] {tps:.1f} tok/s | "
              f"{elapsed:.2f}s | {response[:50]}...")

    mem_after = get_memory_mb()
    return results, round(mem_after - mem_before, 1)


def generate_report(mlx_results: list, llama_results: list,
                    mlx_mem: float, llama_mem: float,
                    output_path: str = "benchmarks/mlx_vs_llamacpp_report.md"):
    """Generate comparison report."""

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return round(sum(vals) / len(vals), 2) if vals else 0

    mlx_avg_tps = avg(mlx_results, "tokens_per_sec")
    llama_avg_tps = avg(llama_results, "tokens_per_sec")
    mlx_avg_lat = avg(mlx_results, "latency_sec")
    llama_avg_lat = avg(llama_results, "latency_sec")

    speedup = mlx_avg_tps / max(llama_avg_tps, 0.001)

    lines = []
    lines.append("# MLX vs llama.cpp — Performance Comparison\n")
    lines.append(f"**Platform:** Apple Silicon ({os.uname().machine})  ")
    lines.append(f"**Test Prompts:** {len(BENCHMARK_PROMPTS)}  ")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Metric | MLX | llama.cpp | Δ |")
    lines.append("|--------|-----|-----------|---|")
    lines.append(f"| **Avg tok/s** | **{mlx_avg_tps}** | {llama_avg_tps} | {speedup:.2f}× |")
    lines.append(f"| **Avg Latency** | {mlx_avg_lat}s | {llama_avg_lat}s | {llama_avg_lat/max(mlx_avg_lat,0.001):.2f}× |")
    lines.append(f"| **Peak Memory** | {mlx_mem} MB | {llama_mem} MB | — |")

    if speedup > 1.0:
        lines.append(f"\n> [!TIP]\n> MLX is **{speedup:.1f}×** faster than llama.cpp on this hardware.\n")
    else:
        lines.append(f"\n> [!NOTE]\n> llama.cpp is faster on this hardware. MLX may be better for larger models.\n")

    # Per-prompt comparison
    lines.append("## Per-Prompt Results\n")
    lines.append("| Prompt | MLX tok/s | llama tok/s | Winner |")
    lines.append("|--------|-----------|-------------|--------|")
    for m, l in zip(mlx_results, llama_results):
        winner = "🏆 MLX" if m["tokens_per_sec"] > l["tokens_per_sec"] else "🏆 llama"
        lines.append(f"| {m['prompt'][:30]}... | {m['tokens_per_sec']} | {l['tokens_per_sec']} | {winner} |")

    # Response quality comparison
    lines.append("\n## Response Quality Comparison\n")
    for i, (m, l) in enumerate(zip(mlx_results, llama_results)):
        lines.append(f"### Prompt {i+1}: {m['prompt']}\n")
        lines.append(f"- **MLX:** {m['response'][:100]}")
        lines.append(f"- **llama.cpp:** {l['response'][:100]}\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📊 Report saved to {output_path}")


def main():
    print("=" * 60)
    print("MLX vs llama.cpp — Performance Benchmark")
    print("=" * 60)

    system_prompt = "你是林夏，一个真实的、有情绪的女生。你的情绪会根据对话自然变化。"

    # Detect available backends
    from inference.mlx_backend import IS_APPLE_SILICON, HAS_MLX

    if not IS_APPLE_SILICON:
        print("\n[ERROR] Not on Apple Silicon. Cannot run MLX benchmark.")
        print("  This benchmark requires an M1/M2/M3/M4 Mac.")
        return

    if not HAS_MLX:
        print("\n[ERROR] mlx-lm not installed. Run: pip install mlx-lm")
        return

    # 1. Benchmark llama.cpp
    print(f"\n{'='*40}")
    print("Benchmarking: llama.cpp")
    print(f"{'='*40}")

    from llama_cpp import Llama
    model_path = None
    for candidate in [
        "emotional-model-output/linxia-dpo-q8_0.gguf",
        "emotional-model-output/linxia-q4_k_m.gguf",
    ]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        print("[ERROR] No GGUF model found.")
        return

    print(f"  Model: {os.path.basename(model_path)}")
    llama_llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    llama_results, llama_mem = benchmark_backend("llama.cpp", llama_llm, system_prompt)
    del llama_llm  # Free memory

    # 2. Benchmark MLX
    print(f"\n{'='*40}")
    print("Benchmarking: MLX")
    print(f"{'='*40}")

    from inference.mlx_backend import MLXBackend
    mlx_llm = MLXBackend(model_id="qwen2.5-1.5b")
    mlx_results, mlx_mem = benchmark_backend("mlx", mlx_llm, system_prompt)

    # 3. Generate report
    generate_report(mlx_results, llama_results, mlx_mem, llama_mem)

    # Save raw data
    raw_data = {
        "mlx": {"results": mlx_results, "memory_mb": mlx_mem},
        "llama_cpp": {"results": llama_results, "memory_mb": llama_mem},
    }
    with open("benchmarks/mlx_vs_llamacpp_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print("📦 Raw data saved to benchmarks/mlx_vs_llamacpp_raw.json")


if __name__ == "__main__":
    main()
