"""
Emotional AI — Lin Xia v3.0 — Product App
Full-stack chat app with:
  - Text chat with emotional AI
  - Image upload → emotional vision reaction
  - Voice synthesis auto-playback
  - Real-time emotion dashboard (mood/trust/affection)
  - Knowledge graph viewer (Mermaid)
  - Model-level emotion tag parsing

Run: python3 app.py         (CLI)
     python3 app.py --ui    (Gradio Web UI)
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

from llama_cpp import Llama

from context_engine.context_manager import ContextManager
from context_engine.sliding_summary import SlidingSummary
from context_engine.emotional_state_model import parse_model_output, ModelEmotionalState
from context_engine.smart_context import SmartContextBuilder
from memory.memory_store import MemoryStore
from memory.memory_extractor import MemoryExtractor
from memory.memory_retriever import MemoryRetriever
from memory.knowledge_graph import KnowledgeGraph
from memory.graph_extractor import GraphExtractor
from voice.tts_engine import TTSEngine
from vision.vision_engine import VisionEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHARACTER_DESCRIPTION


class EmotionalAI:
    """
    Full-stack Emotional AI Engine v3.0.
    Text + Vision + Voice + Knowledge Graph + Emotion Tracking.
    """

    def __init__(
        self,
        model_path: str = "emotional-model-output/linxia-dpo-q8_0.gguf",
        n_ctx: int = 4096,
        enable_memory: bool = True,
        enable_tts: bool = True,
        enable_vision: bool = True,
    ):
        print("=" * 60)
        print("  Emotional AI — Lin Xia (林夏) v3.0")
        print("  Text + Vision + Voice + Knowledge Graph")
        print("=" * 60)

        # 1. Load LLM
        print(f"\n[1/7] Loading model: {os.path.basename(model_path)}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )

        # 2. Context Manager
        print("[2/7] Context engine...")
        self.context_mgr = ContextManager(
            max_context_tokens=n_ctx,
            system_prompt=CHARACTER_DESCRIPTION,
        )

        # 3. Smart Context (Attention Sink)
        self.smart_ctx = SmartContextBuilder(max_ctx=n_ctx, recent_window=8)

        # 4. Sliding Summary
        print("[3/7] Sliding summary...")
        self.sliding_summary = SlidingSummary(window_size=8, llm=self.llm)

        # 5. Emotional State
        self.emotional_state = ModelEmotionalState()

        # 6. Memory (ChromaDB + Knowledge Graph)
        self.enable_memory = enable_memory
        if enable_memory:
            print("[4/7] Persistent memory (ChromaDB + Knowledge Graph)...")
            self.memory_store = MemoryStore()
            self.memory_extractor = MemoryExtractor(llm=self.llm)
            self.memory_retriever = MemoryRetriever(self.memory_store)
            self.knowledge_graph = KnowledgeGraph()
            self.graph_extractor = GraphExtractor(self.knowledge_graph)
            mem_count = self.memory_store.get_memory_count()
            kg_stats = self.knowledge_graph.get_stats()
            print(f"       Memories: {mem_count} | Graph: {kg_stats['nodes']} nodes, {kg_stats['edges']} edges")
        else:
            print("[4/7] Memory disabled.")
            self.memory_store = None
            self.memory_extractor = None
            self.memory_retriever = None
            self.knowledge_graph = KnowledgeGraph()
            self.graph_extractor = None

        # 7. TTS
        self.enable_tts = enable_tts
        if enable_tts:
            print("[5/7] TTS (Edge TTS)...")
            self.tts = TTSEngine()
        else:
            print("[5/7] TTS disabled.")
            self.tts = None

        # 8. Vision
        self.enable_vision = enable_vision
        if enable_vision:
            print("[6/7] Vision engine...")
            self.vision = VisionEngine(strategy="metadata")
        else:
            print("[6/7] Vision disabled.")
            self.vision = None

        print("[7/7] Ready!")
        print(f"\n{'=' * 60}")
        print("Type your message to chat. Commands: /status /memory /graph /quit")
        print(f"{'=' * 60}\n")

    def chat(self, user_input: str, image_path: str = None) -> dict:
        """
        Process user message + optional image.

        Returns:
            {
                "text": str,           # Clean response text
                "emotion": str,        # Detected emotion
                "intensity": int,
                "trust": int,
                "affection": int,
                "audio_path": str,     # Path to generated audio
                "image_reaction": str, # Vision reaction (if image provided)
            }
        """
        result = {"audio_path": "", "image_reaction": ""}

        # Step 0: Vision — if image provided
        vision_context = ""
        if image_path and self.vision:
            img_info = self.vision.analyze_image(image_path)
            vision_reaction = self.vision.generate_emotional_reaction(img_info, llm=self.llm)
            result["image_reaction"] = vision_reaction
            vision_context = f"\n[用户发来了一张图片：{img_info.get('description', '一张照片')}]"

            # Store in graph
            if self.graph_extractor:
                self.graph_extractor.extract_from_image(img_info.get("description", ""))

        # Step 1: Retrieve memories
        if self.enable_memory and self.memory_retriever:
            memories = self.memory_retriever.retrieve(user_input, n_results=3)
            self.context_mgr.set_memories(memories)

        # Step 1b: Get graph context
        graph_context = ""
        if self.knowledge_graph:
            graph_context = self.knowledge_graph.to_context_string(user_input)

        # Step 2: Add turn
        full_input = user_input + vision_context
        self.context_mgr.add_turn("user", full_input)
        self.sliding_summary.add_turn("user", full_input)

        # Step 3: Build prompt with emotion state + graph
        system = CHARACTER_DESCRIPTION
        system += "\n" + self.emotional_state.to_context_line()
        if graph_context:
            system += "\n" + graph_context

        # Add emotion instruction for V2 model
        system += (
            "\n\n每次回复时先输出情绪标签："
            '<emotion state="MOOD" intensity="N" trust_delta="±N" affection_delta="±N"/>'
            "\n有效mood: happy, calm, hurt, angry, cold, playful, shy, gentle, sad, anxious"
        )

        prompt = self.context_mgr.build_prompt()

        # Step 4: Generate
        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>", "<|im_start|>"],
            temperature=0.8,
            top_p=0.9,
            repeat_penalty=1.15,
        )
        raw_response = output["choices"][0]["text"].strip()

        # Step 5: Parse emotion tag
        new_state, clean_text = parse_model_output(raw_response, self.emotional_state)
        self.emotional_state = new_state

        # If image was provided and no text input, use vision reaction
        if image_path and not user_input.strip():
            clean_text = result["image_reaction"] or clean_text

        result["text"] = clean_text
        result["emotion"] = self.emotional_state.mood
        result["intensity"] = self.emotional_state.intensity
        result["trust"] = self.emotional_state.trust
        result["affection"] = self.emotional_state.affection

        # Step 6: Store in history
        self.context_mgr.add_turn("assistant", clean_text)
        self.sliding_summary.add_turn("assistant", clean_text)

        # Step 7: Extract memories + graph
        if self.enable_memory:
            if self.memory_extractor:
                new_memories = self.memory_extractor.extract(user_input, clean_text)
                for mem in new_memories:
                    self.memory_store.add_memory(mem)

            if self.graph_extractor:
                self.graph_extractor.extract_from_turn(user_input, clean_text)

        # Step 8: TTS
        if self.enable_tts and self.tts:
            audio = self.tts.speak(clean_text, self.emotional_state.mood)
            result["audio_path"] = audio

        return result

    def get_status(self) -> str:
        es = self.emotional_state
        mem_count = self.memory_store.get_memory_count() if self.memory_store else 0
        kg = self.knowledge_graph.get_stats() if self.knowledge_graph else {}
        return (
            f"💭 Mood: {es.mood} (intensity {es.intensity}/10)\n"
            f"💛 Trust: {es.trust}/10 | Affection: {es.affection}/10\n"
            f"🧠 Memories: {mem_count} | Graph: {kg.get('nodes', 0)} nodes, {kg.get('edges', 0)} edges\n"
            f"🔊 TTS: {'on' if self.enable_tts else 'off'} | 👁 Vision: {'on' if self.enable_vision else 'off'}"
        )

    def get_graph_mermaid(self) -> str:
        if self.knowledge_graph:
            return self.knowledge_graph.to_mermaid()
        return "graph LR\n  Empty[No data]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli(ai: EmotionalAI):
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break
        elif user_input.lower() == "/status":
            print(ai.get_status())
            continue
        elif user_input.lower() == "/memory":
            if ai.memory_store:
                mems = ai.memory_store.get_all_memories(limit=10)
                for m in mems:
                    print(f"  [{m.get('memory_type','?')}] {m['content']}")
            continue
        elif user_input.lower() == "/graph":
            print(ai.get_graph_mermaid())
            continue

        result = ai.chat(user_input)
        print(f"\nLin Xia [{result['emotion']}]: {result['text']}")


# ---------------------------------------------------------------------------
# Gradio Web UI
# ---------------------------------------------------------------------------

def run_gradio(ai: EmotionalAI):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        run_cli(ai)
        return

    def respond(message, image, history):
        img_path = image if image else None
        result = ai.chat(message or "", image_path=img_path)

        response_text = result["text"]
        if result.get("image_reaction") and image:
            response_text = f"👁 {result['image_reaction']}\n\n{response_text}"

        history = history or []
        history.append({"role": "user", "content": message or "[图片]"})
        history.append({"role": "assistant", "content": response_text})

        audio = result.get("audio_path", "")
        return history, audio, ai.get_status()

    def get_graph():
        return ai.get_graph_mermaid()

    css = """
    .gradio-container { max-width: 1200px !important; }
    .emotion-panel { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                     border-radius: 12px; padding: 16px; color: #e0e0e0; }
    """

    with gr.Blocks(
        title="Lin Xia — Emotional AI v3.0",
        theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
        css=css,
    ) as demo:
        gr.Markdown("# 🌸 Lin Xia (林夏) — Emotional AI v3.0")
        gr.Markdown("*Text + Vision + Voice + Knowledge Graph*")

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat with Lin Xia",
                    height=500,
                    type="messages",
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="Message",
                        placeholder="跟林夏说话...",
                        scale=4,
                    )
                    img_input = gr.Image(
                        label="📷",
                        type="filepath",
                        scale=1,
                        height=80,
                    )
                send_btn = gr.Button("发送 💬", variant="primary")
                audio_output = gr.Audio(label="🔊 Lin Xia's Voice", autoplay=True)

            # Side panel
            with gr.Column(scale=1):
                gr.Markdown("### 💭 Emotional State")
                status_box = gr.Textbox(
                    label="Status",
                    value=ai.get_status(),
                    lines=5,
                    interactive=False,
                    elem_classes=["emotion-panel"],
                )

                gr.Markdown("### 🧠 Knowledge Graph")
                graph_btn = gr.Button("Refresh Graph")
                graph_display = gr.Code(
                    label="Mermaid Diagram",
                    language=None,
                    value=ai.get_graph_mermaid(),
                    lines=10,
                )

                gr.Markdown("### ℹ️ Quick Actions")
                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Chat")
                    examples_btn = gr.Button("💡 Examples")

        # Event handlers
        send_btn.click(
            fn=respond,
            inputs=[msg_box, img_input, chatbot],
            outputs=[chatbot, audio_output, status_box],
        ).then(
            fn=lambda: ("", None),
            outputs=[msg_box, img_input],
        )

        msg_box.submit(
            fn=respond,
            inputs=[msg_box, img_input, chatbot],
            outputs=[chatbot, audio_output, status_box],
        ).then(
            fn=lambda: ("", None),
            outputs=[msg_box, img_input],
        )

        graph_btn.click(fn=get_graph, outputs=graph_display)
        clear_btn.click(fn=lambda: ([], "", ai.get_status()), outputs=[chatbot, graph_display, status_box])

        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Emotional AI — Lin Xia v3.0")
    parser.add_argument("--model", default=None, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision")
    args = parser.parse_args()

    # Auto-detect best model
    if args.model is None:
        candidates = [
            "emotional-model-output/linxia-dpo-q8_0.gguf",     # Best: DPO aligned
            "emotional-model-output/linxia-emotional-v2-q8_0.gguf", # V2: emotion tags
            "emotional-model-output/linxia-emotional-q8_0.gguf",    # V1: SFT only
            "emotional-model-output/linxia-q4_k_m.gguf",            # Lightweight
        ]
        args.model = next((c for c in candidates if os.path.exists(c)), candidates[0])

    ai = EmotionalAI(
        model_path=args.model,
        n_ctx=args.ctx,
        enable_memory=not args.no_memory,
        enable_tts=not args.no_tts,
        enable_vision=not args.no_vision,
    )

    if args.ui:
        run_gradio(ai)
    else:
        run_cli(ai)


if __name__ == "__main__":
    main()
