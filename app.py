"""
Emotional AI — Lin Xia v4.0 — Agentic Cognitive Architecture
Full-stack chat app with:
  - LangGraph cognitive graph (inner monologue + self-check guardrail)
  - Multi-agent subconscious bus (Memory, Emotion, Metabolism, Companion)
  - Pluggable inference backend (llama.cpp / MLX Apple Silicon)
  - RAGAS evaluation framework for RAG quality
  - Text chat + Vision + Voice + Knowledge Graph + Emotion Tracking
  - Full-duplex voice interaction with interruption handling

Run: python3 app.py                          (CLI, auto-detect backend)
     python3 app.py --ui                     (Gradio Web UI)
     python3 app.py --ui --backend mlx       (MLX Apple Silicon)
     python3 app.py --ui --backend llama_cpp  (llama.cpp)
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path

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
from vision.face_memory import FaceMemory
from soul.metabolism import BioClock, MemoryConsolidator, DreamEngine, ProactiveEngine
from soul.relationship import RelationshipEvolution
from soul.screen_perception import ScreenPerception
from soul.screen_share import ScreenShareEngine
from soul.companion import CompanionEngine
from soul.health_perception import HealthPerception
from avatar.avatar_engine import AvatarEngine
from avatar.desktop_app import DesktopLinXia

# v4.0 — Cognitive Graph + Multi-Agent
try:
    from soul.cognitive_graph import LinXiaCognitiveGraph, CognitiveNodes
    HAS_COGNITIVE_GRAPH = True
except ImportError:
    HAS_COGNITIVE_GRAPH = False

try:
    from soul.subconscious_bus import SubconsciousBus, SyncBusAdapter
    from soul.agents import (
        AgentOrchestra, MemoryAgent, EmotionAnalystAgent,
        MetabolismAgent, CompanionAgent,
    )
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False

try:
    from audio.interruption_handler import InterruptionHandler
    HAS_INTERRUPTION = True
except ImportError:
    HAS_INTERRUPTION = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHARACTER_DESCRIPTION


class EmotionalAI:
    """
    Full-stack Emotional AI Engine v4.0.
    LangGraph Cognitive Graph + Multi-Agent Bus + Pluggable Backend.
    """

    def __init__(
        self,
        model_path: str = "emotional-model-output/linxia-dpo-q8_0.gguf",
        n_ctx: int = 4096,
        enable_memory: bool = True,
        enable_tts: bool = True,
        enable_vision: bool = True,
        enable_soul: bool = True,
        backend: str = "llama_cpp",
    ):
        print("=" * 60)
        print("  Emotional AI — Lin Xia (林夏) v4.0")
        print("  Cognitive Graph + Multi-Agent + Pluggable Backend")
        print("=" * 60)

        self.backend_name = backend

        # 1. Load LLM (pluggable backend)
        print(f"\n[1/9] Loading model ({backend}): {os.path.basename(model_path)}")
        if backend == "mlx":
            from inference.mlx_backend import MLXBackend
            self.llm = MLXBackend(model_path=model_path if model_path != "emotional-model-output/linxia-dpo-q8_0.gguf" else None)
        else:
            from llama_cpp import Llama
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

        # 8. Vision + Face Memory
        self.enable_vision = enable_vision
        if enable_vision:
            print("[6/7] Vision engine + Face memory...")
            self.vision = VisionEngine(strategy="metadata")
            self.face_memory = FaceMemory(knowledge_graph=self.knowledge_graph)
        else:
            print("[6/7] Vision disabled.")
            self.vision = None
            self.face_memory = None

        # 9. Soul System (Bio-clock, Relationship, Screen, Health)
        self.enable_soul = enable_soul
        if enable_soul:
            print("[7/8] Soul Engine & Metabolism...")
            self.bio_clock = BioClock()
            self.relationship = RelationshipEvolution()
            self.screen_percept = ScreenPerception()
            self.screen_share = ScreenShareEngine()
            self.companion = CompanionEngine(knowledge_graph=self.knowledge_graph)
            self.health = HealthPerception()
            self.desktop_app = DesktopLinXia()
        else:
            self.bio_clock = None
            self.relationship = None
            self.screen_percept = None
            self.screen_share = None
            self.companion = None
            self.health = None
            self.desktop_app = None

        # 10. Cognitive Graph (LangGraph)
        self.cognitive_graph = None
        if HAS_COGNITIVE_GRAPH:
            print("[8/10] Cognitive Graph (LangGraph)...")
            try:
                nodes = CognitiveNodes(
                    llm=self.llm,
                    memory_retriever=self.memory_retriever if enable_memory else None,
                    memory_store=self.memory_store,
                    memory_extractor=self.memory_extractor,
                    knowledge_graph=self.knowledge_graph,
                    graph_extractor=self.graph_extractor if enable_memory else None,
                    context_mgr=self.context_mgr,
                    sliding_summary=self.sliding_summary,
                    emotional_state_model=self.emotional_state,
                    tts_engine=self.tts if enable_tts else None,
                    vision_engine=self.vision if enable_vision else None,
                    face_memory=self.face_memory if enable_vision else None,
                    bio_clock=self.bio_clock if enable_soul else None,
                    relationship=self.relationship if enable_soul else None,
                    screen_percept=self.screen_percept if enable_soul else None,
                    screen_share=self.screen_share if enable_soul else None,
                    health=self.health if enable_soul else None,
                    companion=self.companion if enable_soul else None,
                    smart_ctx=self.smart_ctx,
                    system_prompt=CHARACTER_DESCRIPTION,
                )
                self.cognitive_graph = LinXiaCognitiveGraph(nodes)
                print("       Cognitive Graph: ✓ (Inner Monologue + Self-Check)")
            except Exception as e:
                print(f"       Cognitive Graph failed: {e}")
        else:
            print("[8/10] Cognitive Graph: disabled (install langgraph)")

        # 11. Multi-Agent Subconscious Bus
        self.orchestra = None
        if HAS_AGENTS and enable_soul:
            print("[9/10] Multi-Agent Subconscious Bus...")
            try:
                bus = SubconsciousBus(debug=False)
                self.bus_adapter = SyncBusAdapter(bus)
                self.orchestra = AgentOrchestra(bus)
                self.orchestra.register(MemoryAgent(
                    bus, self.memory_store, self.memory_retriever,
                ))
                self.orchestra.register(EmotionAnalystAgent(
                    bus, health_perception=self.health,
                ))
                self.orchestra.register(MetabolismAgent(
                    bus, bio_clock=self.bio_clock,
                ))
                self.orchestra.register(CompanionAgent(
                    bus, companion_engine=self.companion,
                    relationship=self.relationship,
                ))
                self.bus_adapter.start()
                print(f"       Agents: {len(self.orchestra.agents)} active")
            except Exception as e:
                print(f"       Agent system failed: {e}")
                self.orchestra = None
        else:
            print("[9/10] Multi-Agent Bus: disabled")

        # 12. Interruption Handler
        self.interruption_handler = None
        if HAS_INTERRUPTION:
            self.interruption_handler = InterruptionHandler()
            print("[10/10] Interruption Handler: ✓")
        else:
            print("[10/10] Interruption Handler: disabled")

        print(f"\n{'=' * 60}")
        print("Type your message to chat. Commands: /status /memory /graph /trace /quit")
        print(f"{'=' * 60}\n")

    def chat(self, user_input: str, image_path: str = None) -> dict:
        """
        Process user message + optional image.
        v4.0: Routes through LangGraph cognitive graph if available.

        Returns:
            {
                "text": str,           # Clean response text
                "emotion": str,        # Detected emotion
                "intensity": int,
                "trust": int,
                "affection": int,
                "audio_path": str,     # Path to generated audio
                "image_reaction": str, # Vision reaction (if image provided)
                "inner_monologue": str,# v4.0: hidden CoT (for debug)
                "trace": list,         # v4.0: cognitive graph trace
            }
        """
        # v4.0: Notify agents of user input
        if self.orchestra and hasattr(self, 'bus_adapter'):
            self.bus_adapter.publish_sync(
                "user.input", {"text": user_input}, source="MainApp"
            )

        # v4.0: Use cognitive graph if available
        if self.cognitive_graph:
            return self._chat_cognitive(user_input, image_path)

        # Fallback: original v3.0 linear pipeline
        return self._chat_legacy(user_input, image_path)

    def _chat_cognitive(self, user_input: str, image_path: str = None) -> dict:
        """v4.0: Route through LangGraph cognitive graph."""
        try:
            state = self.cognitive_graph.invoke(user_input, image_path or "")
            es = state.get("emotional_state", {})
            self.emotional_state = ModelEmotionalState(
                mood=es.get("mood", "calm"),
                intensity=es.get("intensity", 5),
                trust=es.get("trust", 7),
                affection=es.get("affection", 6),
            )
            return {
                "text": state.get("response_text", ""),
                "emotion": es.get("mood", "calm"),
                "intensity": es.get("intensity", 5),
                "trust": es.get("trust", 7),
                "affection": es.get("affection", 6),
                "audio_path": state.get("audio_path", ""),
                "image_reaction": state.get("image_reaction", ""),
                "inner_monologue": state.get("inner_monologue", ""),
                "trace": state.get("trace", []),
            }
        except Exception as e:
            print(f"[COGNITIVE] Graph error, falling back: {e}")
            return self._chat_legacy(user_input, image_path)

    def _chat_legacy(self, user_input: str, image_path: str = None) -> dict:
        """v3.0 legacy linear pipeline (fallback)."""
        result = {"audio_path": "", "image_reaction": "", "inner_monologue": "", "trace": []}

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

        # Step 1c: Get Soul Context (Bio-clock + Relationship + Screen + Health)
        soul_context = ""
        if self.enable_soul:
            # Time-aware bio state
            bio_state = self.bio_clock.get_state()
            soul_context += f"\n{bio_state['context_line']}"
            
            # Relationship stage
            soul_context += f"\n{self.relationship.get_relationship_context()}"
            
            # Screen/App awareness
            app_ctx = self.screen_percept.perceive()
            soul_context += f"\n{app_ctx.context_for_llm}"
            
            # Screen visual context (if share enabled)
            if self.screen_share.enabled:
                screen_snap = self.screen_share.capture_and_analyze()
                if screen_snap:
                    soul_context += f"\n{screen_snap.to_context_string()}"
            
            # Health/Bio-sensing
            health_ctx = self.health.get_health_context()
            if health_ctx:
                soul_context += f"\n{health_ctx}"

        # Step 2: Add turn
        full_input = user_input + vision_context
        self.context_mgr.add_turn("user", full_input)
        self.sliding_summary.add_turn("user", full_input)

        # Step 3: Build prompt with emotion state + graph + soul
        system = CHARACTER_DESCRIPTION
        system += "\n" + self.emotional_state.to_context_line()
        # (Note: graph_context variable was missing in original snippet, assuming it's handled by context_mgr)
        if soul_context:
            system += "\n" + soul_context

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

        # Step 7: Record relationship progression
        if self.enable_soul:
            self.relationship.record_interaction(
                user_input, clean_text, 
                trust_delta=self.emotional_state.trust - result.get("trust", 10),
                affection_delta=self.emotional_state.affection - result.get("affection", 10),
                emotion=self.emotional_state.mood
            )

        # Step 8: TTS (with bio-clock voice modulation)
        if self.enable_tts and self.tts:
            voice_rate = 0
            voice_pitch = 0
            if self.enable_soul:
                bio_state = self.bio_clock.get_state()
                voice_rate = bio_state.get("voice_rate", 0)
                voice_pitch = bio_state.get("voice_pitch", 0)

            audio = self.tts.speak(
                clean_text, 
                self.emotional_state.mood,
                rate=voice_rate,
                pitch=voice_pitch
            )
            result["audio_path"] = audio

        # Step 9: Update Desktop App if running
        if self.enable_soul and self.desktop_app and self.desktop_app._running:
            self.desktop_app.update_emotion(self.emotional_state.mood)
            self.desktop_app.say(clean_text)
            if self.health.last_state.status == "tracking":
                self.desktop_app.update_health(self.health.last_state.bpm, self.health.last_state.stress_level)

        return result

    def get_status(self) -> str:
        es = self.emotional_state
        mem_count = self.memory_store.get_memory_count() if self.memory_store else 0
        kg = self.knowledge_graph.get_stats() if self.knowledge_graph else {}
        parts = [
            f"💭 Mood: {es.mood} (intensity {es.intensity}/10)",
            f"💛 Trust: {es.trust}/10 | Affection: {es.affection}/10",
            f"🧠 Memories: {mem_count} | Graph: {kg.get('nodes', 0)} nodes, {kg.get('edges', 0)} edges",
            f"🔊 TTS: {'on' if self.enable_tts else 'off'} | 👁 Vision: {'on' if self.enable_vision else 'off'}",
            f"🧬 Backend: {self.backend_name} | 🧩 CogGraph: {'✓' if self.cognitive_graph else '✗'}",
        ]
        if self.orchestra:
            stats = self.orchestra.bus.get_stats() if hasattr(self.orchestra, 'bus') else {}
            parts.append(f"🤖 Agents: {len(self.orchestra.agents)} | Events: {stats.get('published', 0)}")
        return "\n".join(parts)

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
    import threading
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        run_cli(ai)
        return

    # Background processing state
    state = {
        "bio_sensing": False,
        "screen_share": False,
        "desktop_mode": False
    }

    def bio_sensing_loop():
        """Background thread for webcam health tracking."""
        import cv2
        cap = None
        while True:
            if state["bio_sensing"]:
                if cap is None:
                    cap = cv2.VideoCapture(0)
                
                ret, frame = cap.read()
                if ret:
                    ai.health.process_frame(frame)
                    # Update desktop app if running
                    if ai.desktop_app and ai.desktop_app._running:
                        hs = ai.health.last_state
                        if hs.status == "tracking":
                            ai.desktop_app.update_health(hs.bpm, hs.stress_level)
            else:
                if cap is not None:
                    cap.release()
                    cap = None
            
            time.sleep(0.1)

    # Start bio thread
    bio_thread = threading.Thread(target=bio_sensing_loop, daemon=True)
    bio_thread.start()

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
        
        # Format bio info for status
        bio_info = ""
        if state["bio_sensing"]:
            hs = ai.health.last_state
            if hs.status == "tracking":
                bio_info = f"\n💓 Body: {hs.bpm:.0f} BPM | Stress: {hs.stress_level:.1%}"
            else:
                bio_info = f"\n💓 Body: {hs.status}"

        return history, audio, ai.get_status() + bio_info

    def get_graph():
        return ai.get_graph_mermaid()

    css = """
    .gradio-container { max-width: 1200px !important; }
    .emotion-panel { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                     border-radius: 12px; padding: 16px; color: #e0e0e0; }
    """

    with gr.Blocks(
        title="Lin Xia — Emotional AI v4.0",
        theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
        css=css,
    ) as demo:
        gr.Markdown("# 🌸 Lin Xia (林夏) — Emotional AI v4.0")
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

                gr.Markdown("### 🛠️ v6.0 Modules")
                with gr.Group():
                    bio_toggle = gr.Checkbox(label="💓 Bio-Sensing (Webcam rPPG)", value=False)
                    share_toggle = gr.Checkbox(label="🖥️ Screen Sharing", value=False)
                    desktop_btn = gr.Button("🚀 Launch Desktop Mode", variant="secondary")
                
                def toggle_bio(val):
                    state["bio_sensing"] = val
                    return f"Bio-Sensing: {'ON' if val else 'OFF'}"
                
                def toggle_share(val):
                    state["screen_share"] = val
                    if ai.screen_share:
                        if val: ai.screen_share.enable()
                        else: ai.screen_share.disable()
                    return f"Screen Share: {'ON' if val else 'OFF'}"
                
                def launch_desktop():
                    if ai.desktop_app:
                        # Launch in a separate thread to not block Gradio
                        def run():
                            ai.desktop_app.launch(
                                initial_emotion=ai.emotional_state.mood,
                                initial_message="我来桌面陪你啦！"
                            )
                        threading.Thread(target=run, daemon=True).start()
                        state["desktop_mode"] = True
                        return "Desktop Mode Launched!"
                    return "Desktop App Not Available"

                bio_toggle.change(fn=toggle_bio, inputs=bio_toggle, outputs=status_box)
                share_toggle.change(fn=toggle_share, inputs=share_toggle, outputs=status_box)
                desktop_btn.click(fn=launch_desktop, outputs=status_box)

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
    parser = argparse.ArgumentParser(description="Emotional AI — Lin Xia v4.0")
    parser.add_argument("--model", default=None, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision")
    parser.add_argument("--backend", default="llama_cpp",
                        choices=["llama_cpp", "mlx", "auto"],
                        help="Inference backend (default: llama_cpp)")
    args = parser.parse_args()

    # Auto-detect best model
    if args.model is None and args.backend != "mlx":
        candidates = [
            "emotional-model-output/linxia-dpo-q8_0.gguf",     # Best: DPO aligned
            "emotional-model-output/linxia-emotional-v2-q8_0.gguf", # V2: emotion tags
            "emotional-model-output/linxia-emotional-q8_0.gguf",    # V1: SFT only
            "emotional-model-output/linxia-q4_k_m.gguf",            # Lightweight
        ]
        args.model = next((c for c in candidates if os.path.exists(c)), candidates[0])
    elif args.model is None:
        args.model = "emotional-model-output/linxia-dpo-q8_0.gguf"  # Placeholder for MLX

    ai = EmotionalAI(
        model_path=args.model,
        n_ctx=args.ctx,
        enable_memory=not args.no_memory,
        enable_tts=not args.no_tts,
        enable_vision=not args.no_vision,
        backend=args.backend,
    )

    if args.ui:
        run_gradio(ai)
    else:
        run_cli(ai)


if __name__ == "__main__":
    main()
