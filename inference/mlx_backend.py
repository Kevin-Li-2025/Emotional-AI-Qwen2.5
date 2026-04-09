"""
MLX Backend — Apple Silicon Native LLM Inference

Provides an MLX-powered inference backend as an alternative to llama.cpp.
Leverages Apple Silicon's unified memory architecture for optimal performance.

Features:
  - Same API interface as llama_cpp.Llama (drop-in replacement)
  - Automatic model download from mlx-community on HuggingFace
  - Streaming token generation
  - Memory-efficient 4-bit quantized inference

Usage:
  from inference.mlx_backend import MLXBackend

  llm = MLXBackend(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
  output = llm("prompt here", max_tokens=200)

Requirements:
  - Apple Silicon Mac (M1/M2/M3/M4)
  - pip install mlx-lm
"""

import os
import sys
import time
import platform
from dataclasses import dataclass

# Check Apple Silicon availability
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and
    platform.machine() == "arm64"
)

try:
    from mlx_lm import load, generate
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False

# Available MLX model mappings
MLX_MODEL_MAP = {
    "qwen2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}


@dataclass
class MLXGenerationConfig:
    """Generation parameters for MLX inference."""
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.15


class MLXBackend:
    """
    MLX-powered LLM inference backend for Apple Silicon.
    
    Provides a llama_cpp-compatible interface so it can be used as a
    drop-in replacement throughout the codebase.
    """

    def __init__(self, model_id: str = None, model_path: str = None):
        """
        Initialize the MLX backend.
        
        Args:
            model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
                      Can also be a short alias from MLX_MODEL_MAP.
            model_path: Local path to model directory (overrides model_id).
        """
        if not IS_APPLE_SILICON:
            raise RuntimeError(
                "MLX backend requires Apple Silicon (M1/M2/M3/M4). "
                "Use --backend llama_cpp for non-Apple hardware."
            )

        if not HAS_MLX:
            raise ImportError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            )

        # Resolve model ID
        if model_path:
            self.model_id = model_path
        elif model_id:
            self.model_id = MLX_MODEL_MAP.get(model_id.lower(), model_id)
        else:
            self.model_id = MLX_MODEL_MAP["qwen2.5-1.5b"]

        print(f"[MLX] Loading model: {self.model_id}")
        t0 = time.time()
        self.model, self.tokenizer = load(self.model_id)
        load_time = time.time() - t0
        print(f"[MLX] Model loaded in {load_time:.1f}s")

        # Track stats
        self.total_tokens = 0
        self.total_time = 0.0

    def __call__(self, prompt: str, max_tokens: int = 512,
                 stop: list[str] = None, temperature: float = 0.8,
                 top_p: float = 0.9, repeat_penalty: float = 1.15,
                 stream: bool = False, **kwargs) -> dict:
        """
        Generate text — compatible with llama_cpp.Llama.__call__() interface.
        
        Returns:
            dict with "choices" and "usage" keys matching llama_cpp format.
        """
        t0 = time.time()

        if stream:
            return self._stream_generate(prompt, max_tokens, stop,
                                         temperature, top_p, repeat_penalty)

        # Non-streaming generation
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                verbose=False,
            )
        except Exception as e:
            print(f"[MLX] Generation error: {e}")
            return {
                "choices": [{"text": f"(MLX error: {e})"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        elapsed = time.time() - t0

        # Apply stop sequences
        if stop:
            for s in stop:
                if s in response:
                    response = response[:response.index(s)]

        # Estimate token count (rough)
        chinese_chars = sum(1 for c in response if '\u4e00' <= c <= '\u9fff')
        completion_tokens = int(chinese_chars / 1.5 + (len(response) - chinese_chars) / 4)
        prompt_tokens = int(len(prompt) / 4)

        self.total_tokens += completion_tokens
        self.total_time += elapsed

        return {
            "choices": [{"text": response}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "timing": {
                "elapsed_sec": round(elapsed, 3),
                "tokens_per_sec": round(completion_tokens / max(elapsed, 0.001), 1),
            },
        }

    def _stream_generate(self, prompt: str, max_tokens: int,
                         stop: list[str], temperature: float,
                         top_p: float, repeat_penalty: float):
        """
        Streaming token generation — yields token-by-token.
        Compatible with llama_cpp streaming interface.
        """
        # MLX-LM doesn't have native streaming in the same way,
        # so we simulate it by generating in chunks
        try:
            full_response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                verbose=False,
            )
        except Exception as e:
            yield {"choices": [{"text": f"(MLX error: {e})"}]}
            return

        # Apply stop sequences
        if stop:
            for s in stop:
                if s in full_response:
                    full_response = full_response[:full_response.index(s)]

        # Yield character by character (simulated streaming)
        for char in full_response:
            yield {"choices": [{"text": char}]}

    def get_stats(self) -> dict:
        """Get performance statistics."""
        avg_tps = self.total_tokens / max(self.total_time, 0.001)
        return {
            "backend": "mlx",
            "model": self.model_id,
            "total_tokens": self.total_tokens,
            "total_time_sec": round(self.total_time, 2),
            "avg_tokens_per_sec": round(avg_tps, 1),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_backend(backend: str = "auto", model_path: str = None,
                   model_id: str = None, n_ctx: int = 4096, **kwargs):
    """
    Create an LLM backend based on the specified type.
    
    Args:
        backend: "mlx", "llama_cpp", or "auto" (picks best available)
        model_path: Path to GGUF (llama_cpp) or directory (MLX)
        model_id: HuggingFace model ID (MLX only)
        n_ctx: Context window size (llama_cpp only)
    
    Returns:
        An LLM object with __call__ interface.
    """
    if backend == "auto":
        if IS_APPLE_SILICON and HAS_MLX:
            backend = "mlx"
            print("[BACKEND] Auto-selected: MLX (Apple Silicon detected)")
        else:
            backend = "llama_cpp"
            print("[BACKEND] Auto-selected: llama_cpp")

    if backend == "mlx":
        return MLXBackend(model_id=model_id, model_path=model_path)

    elif backend == "llama_cpp":
        from llama_cpp import Llama
        if not model_path:
            # Auto-detect GGUF model
            candidates = [
                "emotional-model-output/linxia-dpo-q8_0.gguf",
                "emotional-model-output/linxia-emotional-v2-q8_0.gguf",
                "emotional-model-output/linxia-q4_k_m.gguf",
            ]
            model_path = next((c for c in candidates if os.path.exists(c)), candidates[0])

        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mlx', 'llama_cpp', or 'auto'.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Quick demo of the MLX backend."""
    print("=" * 60)
    print("MLX Backend — Apple Silicon LLM Inference")
    print("=" * 60)

    print(f"\n  Apple Silicon: {IS_APPLE_SILICON}")
    print(f"  MLX Available: {HAS_MLX}")
    print(f"  Platform: {platform.machine()} / {platform.system()}")

    if not IS_APPLE_SILICON:
        print("\n  [SKIP] Not on Apple Silicon. MLX backend unavailable.")
        return

    if not HAS_MLX:
        print("\n  [SKIP] mlx-lm not installed. Run: pip install mlx-lm")
        return

    # Load model
    llm = MLXBackend(model_id="qwen2.5-1.5b")

    # Test prompts
    prompts = [
        ("你好！", "neutral"),
        ("你就是个人工智能程序。", "angry"),
        ("我给你买了草莓蛋糕！", "happy"),
    ]

    system = "你是林夏，一个真实的、有情绪的女生。"

    for user_text, expected in prompts:
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        result = llm(prompt, max_tokens=200, stop=["<|im_end|>"],
                      temperature=0.8, repeat_penalty=1.15)

        response = result["choices"][0]["text"].strip()
        timing = result.get("timing", {})
        print(f"\n  User: {user_text}")
        print(f"  Lin Xia: {response[:80]}...")
        print(f"  Speed: {timing.get('tokens_per_sec', 0)} tok/s | "
              f"Time: {timing.get('elapsed_sec', 0)}s")

    # Stats
    stats = llm.get_stats()
    print(f"\n  Overall: {stats['avg_tokens_per_sec']} tok/s average")


if __name__ == "__main__":
    demo()
