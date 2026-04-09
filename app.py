"""
Emotional AI — Unified Chat Application
Integrates all components: GGUF inference, context management,
persistent memory (RAG), emotional state tracking, and TTS.

Run with: python3 app.py
Or with Gradio UI: python3 app.py --ui
"""

import argparse
import json
import time
import sys
from pathlib import Path

from llama_cpp import Llama

from context_engine.context_manager import ContextManager
from context_engine.sliding_summary import SlidingSummary
from memory.memory_store import MemoryStore
from memory.memory_extractor import MemoryExtractor
from memory.memory_retriever import MemoryRetriever
from voice.tts_engine import TTSEngine

# Load character description from config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHARACTER_DESCRIPTION


class EmotionalAI:
    """
    Full-stack Emotional AI engine.
    Combines inference, memory, context compression, and TTS
    into a single coherent system.
    """

    def __init__(
        self,
        model_path: str = "emotional-model-output/linxia-emotional-q8_0.gguf",
        n_ctx: int = 4096,
        tts_backend: str = "mock",
        enable_memory: bool = True,
        enable_tts: bool = False,
    ):
        print("=" * 60)
        print("Emotional AI — Lin Xia (林夏) v2.0")
        print("=" * 60)

        # 1. Load LLM
        print(f"\n[1/5] Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )

        # 2. Context Manager
        print("[2/5] Initializing context engine...")
        self.context_mgr = ContextManager(
            max_context_tokens=n_ctx,
            system_prompt=CHARACTER_DESCRIPTION,
        )

        # 3. Sliding Summary
        print("[3/5] Initializing sliding summary...")
        self.sliding_summary = SlidingSummary(window_size=8, llm=self.llm)

        # 4. Memory (RAG)
        self.enable_memory = enable_memory
        if enable_memory:
            print("[4/5] Initializing persistent memory (ChromaDB)...")
            self.memory_store = MemoryStore()
            self.memory_extractor = MemoryExtractor(llm=self.llm)
            self.memory_retriever = MemoryRetriever(self.memory_store)
            print(f"       Stored memories: {self.memory_store.get_memory_count()}")
        else:
            print("[4/5] Memory disabled.")
            self.memory_store = None
            self.memory_extractor = None
            self.memory_retriever = None

        # 5. TTS
        self.enable_tts = enable_tts
        if enable_tts:
            print(f"[5/5] Initializing TTS ({tts_backend})...")
            self.tts = TTSEngine(backend=tts_backend)
        else:
            print("[5/5] TTS disabled.")
            self.tts = None

        print(f"\n{'=' * 60}")
        print("Ready. Type your message to chat with Lin Xia.")
        print("Commands: /status, /memory, /clear, /quit")
        print(f"{'=' * 60}\n")

    def chat(self, user_input: str) -> str:
        """
        Process a user message and generate Lin Xia's response.
        Full pipeline: memory retrieval → context building → inference → memory extraction → TTS
        """
        # Step 1: Retrieve relevant memories
        if self.enable_memory and self.memory_retriever:
            memories = self.memory_retriever.retrieve(user_input, n_results=3)
            self.context_mgr.set_memories(memories)

        # Step 2: Add user turn to context + sliding summary
        self.context_mgr.add_turn("user", user_input)
        self.sliding_summary.add_turn("user", user_input)

        # Step 3: Inject compressed history if available
        compressed = self.sliding_summary.get_compressed_history()
        if compressed:
            # Update the emotional state from summaries
            for summary in self.sliding_summary.summaries:
                self.context_mgr.emotional_state.update_from_analysis(summary)

        # Step 4: Build optimized prompt
        prompt = self.context_mgr.build_prompt()

        # Step 5: Generate response
        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>", "<|im_start|>"],
            temperature=0.8,
            top_p=0.9,
            repeat_penalty=1.15,
        )

        response = output["choices"][0]["text"].strip()

        # Step 6: Add response to context + sliding summary
        self.context_mgr.add_turn("assistant", response)
        self.sliding_summary.add_turn("assistant", response)

        # Step 7: Extract and store memories (async-safe)
        if self.enable_memory and self.memory_extractor:
            new_memories = self.memory_extractor.extract(user_input, response)
            for mem in new_memories:
                self.memory_store.add_memory(mem)

        # Step 8: TTS (if enabled)
        if self.enable_tts and self.tts:
            emotion_tag, clean_text = TTSEngine.parse_emotion_tag(response)
            self.tts.synthesize(clean_text, emotion_tag)

        return response

    def get_status(self) -> str:
        """Return system status information."""
        ctx_stats = self.context_mgr.get_context_stats()
        summary_stats = self.sliding_summary.get_stats()
        mem_count = self.memory_store.get_memory_count() if self.memory_store else 0

        lines = [
            "=== System Status ===",
            f"Emotional State: {ctx_stats['emotional_state']}",
            f"Context Usage: {ctx_stats['utilization']} ({ctx_stats['total_estimated_tokens']}/{ctx_stats['max_context']} tokens)",
            f"History: {ctx_stats['total_history_turns']} total → {ctx_stats['included_turns']} in context",
            f"Summaries: {summary_stats['segments_summarized']} segments compressed",
            f"Total Turns: {summary_stats['total_turns_processed']}",
            f"Memories Stored: {mem_count}",
            f"TTS: {'enabled' if self.enable_tts else 'disabled'}",
        ]
        return "\n".join(lines)

    def get_memories(self) -> str:
        """Display all stored memories."""
        if not self.memory_store:
            return "Memory is disabled."

        memories = self.memory_store.get_all_memories(limit=20)
        if not memories:
            return "No memories stored yet."

        lines = ["=== Lin Xia's Memories ==="]
        for i, mem in enumerate(memories):
            mem_type = mem.get("memory_type", "?")
            importance = mem.get("importance", "?")
            lines.append(f"  [{mem_type}] (importance: {importance}) {mem['content']}")
        return "\n".join(lines)


def run_cli(ai: EmotionalAI):
    """Run the CLI chat interface."""
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/status":
            print(ai.get_status())
            continue
        elif user_input.lower() == "/memory":
            print(ai.get_memories())
            continue
        elif user_input.lower() == "/clear":
            ai.context_mgr.conversation_history.clear()
            ai.sliding_summary.full_history.clear()
            ai.sliding_summary.summaries.clear()
            print("Conversation history cleared.")
            continue

        response = ai.chat(user_input)
        print(f"\nLin Xia: {response}")


def run_gradio(ai: EmotionalAI):
    """Run the Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        print("Falling back to CLI mode.")
        run_cli(ai)
        return

    def respond(message, history):
        response = ai.chat(message)
        return response

    def get_status():
        return ai.get_status()

    def get_memories():
        return ai.get_memories()

    with gr.Blocks(
        title="Lin Xia — Emotional AI",
        theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
    ) as demo:
        gr.Markdown("# 🌸 Lin Xia (林夏) — Emotional AI v2.0")
        gr.Markdown("A human-like emotional companion with persistent memory and context compression.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    type="messages",
                    examples=[
                        "林夏，你在吗？",
                        "我今天心情不太好...",
                        "我给你买了巧克力！",
                    ],
                )

            with gr.Column(scale=1):
                gr.Markdown("### System Status")
                status_box = gr.Textbox(
                    label="Status",
                    value=get_status,
                    lines=10,
                    interactive=False,
                    every=5,
                )
                gr.Markdown("### Memories")
                memory_box = gr.Textbox(
                    label="Stored Memories",
                    value=get_memories,
                    lines=10,
                    interactive=False,
                    every=10,
                )

        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


def main():
    parser = argparse.ArgumentParser(description="Emotional AI — Lin Xia Chat")
    parser.add_argument("--model", default="emotional-model-output/linxia-emotional-q8_0.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    parser.add_argument("--tts-backend", default="mock", choices=["mock", "cosyvoice", "gpt_sovits"],
                        help="TTS backend to use")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory")
    args = parser.parse_args()

    ai = EmotionalAI(
        model_path=args.model,
        n_ctx=args.ctx,
        tts_backend=args.tts_backend,
        enable_memory=not args.no_memory,
        enable_tts=args.tts,
    )

    if args.ui:
        run_gradio(ai)
    else:
        run_cli(ai)


if __name__ == "__main__":
    main()
