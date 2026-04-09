"""
Cognitive Graph — Lin Xia's Agentic Thinking Architecture (LangGraph)

Replaces the linear chat() pipeline with a graph-based cognitive architecture.
Lin Xia now "thinks" through a directed graph:

  UserInput → MemoryRetrieval → SoulSensing → InnerMonologue → Response → SelfCheck → Output
                                                                            ↑          │
                                                                            └──(retry)──┘

Key innovations:
  1. INNER MONOLOGUE: Before responding, Lin Xia generates hidden internal
     reasoning ("他看起来很累，我应该温柔一点..."). This is injected into
     the response prompt but not shown to the user.

  2. SELF-CHECK GUARDRAIL: After generating, a validator checks if the
     response contradicts the current relationship stage. If "初识" stage
     says "我好想你" → reject and regenerate (max 2 retries).

  3. FULL TRACEABILITY: Every node's input/output is recorded in the State,
     enabling debugging and observability.

Requires: pip install langgraph
"""

import re
import time
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("[COGNITIVE] langgraph not installed. Run: pip install langgraph")


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class LinXiaState(TypedDict):
    """Shared state flowing through the cognitive graph."""
    # Input
    user_input: str
    image_path: str

    # Memory retrieval
    memories: list[str]
    graph_context: str

    # Soul sensing
    soul_context: str
    bio_state: dict
    relationship_stage: str

    # Inner monologue (hidden from user)
    inner_monologue: str

    # Response generation
    emotional_state: dict
    response_text: str
    raw_response: str

    # Self-check
    self_check_passed: bool
    self_check_reason: str
    retry_count: int

    # Output
    audio_path: str
    image_reaction: str

    # Trace
    trace: list[dict]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

class CognitiveNodes:
    """
    Collection of node functions for the cognitive graph.
    Each node takes the full state and returns a partial update.
    """

    def __init__(self, llm, memory_retriever, memory_store, memory_extractor,
                 knowledge_graph, graph_extractor, context_mgr, sliding_summary,
                 emotional_state_model, tts_engine, vision_engine, face_memory,
                 bio_clock, relationship, screen_percept, screen_share,
                 health, companion, smart_ctx, system_prompt):
        self.llm = llm
        self.memory_retriever = memory_retriever
        self.memory_store = memory_store
        self.memory_extractor = memory_extractor
        self.knowledge_graph = knowledge_graph
        self.graph_extractor = graph_extractor
        self.context_mgr = context_mgr
        self.sliding_summary = sliding_summary
        self.emotional_state_model = emotional_state_model
        self.tts_engine = tts_engine
        self.vision_engine = vision_engine
        self.face_memory = face_memory
        self.bio_clock = bio_clock
        self.relationship = relationship
        self.screen_percept = screen_percept
        self.screen_share = screen_share
        self.health = health
        self.companion = companion
        self.smart_ctx = smart_ctx
        self.system_prompt = system_prompt

    def memory_retrieval(self, state: LinXiaState) -> dict:
        """Node: Retrieve relevant memories from ChromaDB + Knowledge Graph."""
        t0 = time.time()
        memories = []
        graph_context = ""

        if self.memory_retriever:
            memories = self.memory_retriever.retrieve(state["user_input"], n_results=3)

        if self.knowledge_graph:
            graph_context = self.knowledge_graph.to_context_string(state["user_input"])

        return {
            "memories": memories,
            "graph_context": graph_context,
            "trace": state.get("trace", []) + [{
                "node": "memory_retrieval",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "memories_found": len(memories),
            }],
        }

    def soul_sensing(self, state: LinXiaState) -> dict:
        """Node: Gather environmental/biological context."""
        t0 = time.time()
        soul_parts = []
        bio_state = {}
        relationship_stage = ""

        if self.bio_clock:
            try:
                bs = self.bio_clock.get_state()
                if isinstance(bs, dict):
                    soul_parts.append(bs.get("context_line", ""))
                    bio_state = bs
                else:
                    soul_parts.append(bs.to_context_string())
                    bio_state = {"time_of_day": bs.time_of_day, "energy": bs.energy}
            except Exception:
                pass

        if self.relationship:
            try:
                soul_parts.append(self.relationship.get_relationship_context())
                relationship_stage = self.relationship.get_stage_name()
            except Exception:
                pass

        if self.screen_percept:
            try:
                app_ctx = self.screen_percept.perceive()
                soul_parts.append(app_ctx.context_for_llm)
            except Exception:
                pass

        if self.health:
            try:
                health_ctx = self.health.get_health_context()
                if health_ctx:
                    soul_parts.append(health_ctx)
            except Exception:
                pass

        return {
            "soul_context": "\n".join(p for p in soul_parts if p),
            "bio_state": bio_state,
            "relationship_stage": relationship_stage,
            "trace": state.get("trace", []) + [{
                "node": "soul_sensing",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "relationship_stage": relationship_stage,
            }],
        }

    def inner_monologue(self, state: LinXiaState) -> dict:
        """
        Node: Generate hidden inner reasoning (not shown to user).

        This is the key cognitive innovation — Lin Xia "thinks" before speaking:
          "用户说他很累...他心率也偏高。我应该关心他，但不能太刻意。"

        This monologue is injected into the response prompt for better coherence.
        """
        t0 = time.time()

        # Build context for inner thought
        memory_str = "\n".join(f"- {m}" for m in state.get("memories", []))
        soul = state.get("soul_context", "")
        relationship = state.get("relationship_stage", "未知")

        prompt = (
            "<|im_start|>system\n"
            "你是林夏的内心。现在用户说了一句话，你要快速思考：\n"
            "1. 用户的情绪和意图是什么？\n"
            "2. 我现在的状态（关系阶段、心情）适合怎么回应？\n"
            "3. 有什么相关的记忆可以引用？\n"
            "4. 我应该用什么语气和态度？\n\n"
            "注意：这是你的内心独白，不会展示给用户。简短即可，2-3句话。\n"
            f"\n[当前关系: {relationship}]\n"
            f"[相关记忆]\n{memory_str}\n"
            f"[环境感知]\n{soul}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n用户说: \"{state['user_input']}\"<|im_end|>\n"
            "<|im_start|>assistant\n"
            "内心想法: "
        )

        try:
            output = self.llm(
                prompt, max_tokens=100, stop=["<|im_end|>"],
                temperature=0.7, repeat_penalty=1.1,
            )
            monologue = output["choices"][0]["text"].strip()
            # Strip any emotion tags leaked
            monologue = re.sub(r'<emotion[^>]*/>', '', monologue).strip()
        except Exception as e:
            monologue = f"(思考中断: {e})"

        return {
            "inner_monologue": monologue,
            "trace": state.get("trace", []) + [{
                "node": "inner_monologue",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "thought": monologue[:80],
            }],
        }

    def emotional_response(self, state: LinXiaState) -> dict:
        """Node: Generate Lin Xia's visible response with emotion tags."""
        t0 = time.time()

        # Build enhanced system prompt
        system = self.system_prompt
        if state.get("inner_monologue"):
            system += f"\n\n[你的内心想法（参考但不要直接说出来）]\n{state['inner_monologue']}"
        if state.get("graph_context"):
            system += f"\n{state['graph_context']}"
        if state.get("soul_context"):
            system += f"\n{state['soul_context']}"

        # Add emotion instruction
        system += (
            "\n\n每次回复时先输出情绪标签："
            '<emotion state="MOOD" intensity="N" trust_delta="±N" affection_delta="±N"/>'
            "\n有效mood: happy, calm, hurt, angry, cold, playful, shy, gentle, sad, anxious"
        )

        # Inject memories
        if state.get("memories"):
            memory_str = "\n".join(f"- {m}" for m in state["memories"])
            system += f"\n\n[你的记忆]\n{memory_str}"

        # Build conversation context
        if self.context_mgr:
            self.context_mgr.add_turn("user", state["user_input"])
            prompt = self.context_mgr.build_prompt()
        else:
            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{state['user_input']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        try:
            output = self.llm(
                prompt, max_tokens=512,
                stop=["<|im_end|>", "<|im_start|>"],
                temperature=0.8, top_p=0.9, repeat_penalty=1.15,
            )
            raw_response = output["choices"][0]["text"].strip()
        except Exception as e:
            raw_response = f"(生成失败: {e})"

        # Parse emotion tag
        from context_engine.emotional_state_model import parse_model_output, ModelEmotionalState
        prev_state = ModelEmotionalState(**state.get("emotional_state", {})) if state.get("emotional_state") else None
        new_state, clean_text = parse_model_output(raw_response, prev_state)

        return {
            "raw_response": raw_response,
            "response_text": clean_text,
            "emotional_state": {
                "mood": new_state.mood,
                "intensity": new_state.intensity,
                "trust": new_state.trust,
                "affection": new_state.affection,
                "trust_delta": new_state.trust_delta,
                "affection_delta": new_state.affection_delta,
            },
            "trace": state.get("trace", []) + [{
                "node": "emotional_response",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "emotion": new_state.mood,
                "intensity": new_state.intensity,
            }],
        }

    def self_check(self, state: LinXiaState) -> dict:
        """
        Node: Guardrail — verify response matches relationship stage.

        Rules:
          - STRANGER stage: No "我想你", "宝贝", intimate language
          - ACQUAINTANCE: No deep vulnerability
          - If violated: retry with feedback (max 2 retries)
        """
        t0 = time.time()
        response = state.get("response_text", "")
        stage = state.get("relationship_stage", "")
        retry = state.get("retry_count", 0)

        passed = True
        reason = ""

        # Stage-specific boundaries
        STAGE_BOUNDARIES = {
            "初识": {
                "forbidden": ["想你", "宝贝", "亲爱的", "我好喜欢你", "抱抱", "么么"],
                "reason": "关系等级是'初识'，不能说太亲密的话",
            },
            "熟人": {
                "forbidden": ["我爱你", "灵魂伴侣", "你是我的一切"],
                "reason": "关系等级是'熟人'，不能说深层情感表白",
            },
        }

        if stage in STAGE_BOUNDARIES:
            boundary = STAGE_BOUNDARIES[stage]
            for word in boundary["forbidden"]:
                if word in response:
                    passed = False
                    reason = f"{boundary['reason']}（触发词: '{word}'）"
                    break

        # Check for nonsensical/empty responses
        if len(response.strip()) < 3:
            passed = False
            reason = "回复过短"

        return {
            "self_check_passed": passed,
            "self_check_reason": reason,
            "retry_count": retry + (0 if passed else 1),
            "trace": state.get("trace", []) + [{
                "node": "self_check",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "passed": passed,
                "reason": reason,
            }],
        }

    def output(self, state: LinXiaState) -> dict:
        """Node: Final output — TTS, memory storage, relationship update."""
        t0 = time.time()
        audio_path = ""

        # TTS
        if self.tts_engine:
            try:
                mood = state.get("emotional_state", {}).get("mood", "calm")
                audio_path = self.tts_engine.speak(state["response_text"], mood)
            except Exception:
                pass

        # Store memories
        if self.memory_extractor and self.memory_store:
            try:
                new_memories = self.memory_extractor.extract(
                    state["user_input"], state["response_text"]
                )
                for mem in new_memories:
                    self.memory_store.add_memory(mem)
            except Exception:
                pass

        # Update knowledge graph
        if self.graph_extractor:
            try:
                self.graph_extractor.extract_from_turn(
                    state["user_input"], state["response_text"]
                )
            except Exception:
                pass

        # Record conversation
        if self.context_mgr:
            self.context_mgr.add_turn("assistant", state["response_text"])
        if self.sliding_summary:
            self.sliding_summary.add_turn("user", state["user_input"])
            self.sliding_summary.add_turn("assistant", state["response_text"])

        # Update relationship
        if self.relationship:
            try:
                es = state.get("emotional_state", {})
                self.relationship.record_interaction(
                    state["user_input"], state["response_text"],
                    trust_delta=es.get("trust_delta", 0),
                    affection_delta=es.get("affection_delta", 0),
                    emotion=es.get("mood", "neutral"),
                )
            except Exception:
                pass

        return {
            "audio_path": audio_path,
            "trace": state.get("trace", []) + [{
                "node": "output",
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "audio_generated": bool(audio_path),
            }],
        }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class LinXiaCognitiveGraph:
    """
    Builds and manages the LangGraph cognitive architecture.
    """

    def __init__(self, nodes: CognitiveNodes):
        self.nodes = nodes
        self.graph = None
        self.compiled = None

        if HAS_LANGGRAPH:
            self._build()
        else:
            print("[COGNITIVE] LangGraph not available. Using fallback mode.")

    def _build(self):
        """Build the cognitive state graph."""
        builder = StateGraph(LinXiaState)

        # Add nodes
        builder.add_node("memory_retrieval", self.nodes.memory_retrieval)
        builder.add_node("soul_sensing", self.nodes.soul_sensing)
        builder.add_node("inner_monologue", self.nodes.inner_monologue)
        builder.add_node("emotional_response", self.nodes.emotional_response)
        builder.add_node("self_check", self.nodes.self_check)
        builder.add_node("output", self.nodes.output)

        # Set entry point
        builder.set_entry_point("memory_retrieval")

        # Linear edges
        builder.add_edge("memory_retrieval", "soul_sensing")
        builder.add_edge("soul_sensing", "inner_monologue")
        builder.add_edge("inner_monologue", "emotional_response")
        builder.add_edge("emotional_response", "self_check")

        # Conditional edge: self_check → output OR retry
        def should_retry(state: LinXiaState) -> str:
            if state.get("self_check_passed", True):
                return "output"
            if state.get("retry_count", 0) >= 2:
                return "output"  # Give up after 2 retries
            return "emotional_response"  # Retry

        builder.add_conditional_edges(
            "self_check",
            should_retry,
            {
                "output": "output",
                "emotional_response": "emotional_response",
            }
        )

        builder.add_edge("output", END)

        self.compiled = builder.compile()
        print("[COGNITIVE] LangGraph cognitive graph compiled ✓")

    def invoke(self, user_input: str, image_path: str = "") -> dict:
        """
        Run the full cognitive graph for a single user turn.
        Returns the final state with response text, emotion, audio, and trace.
        """
        initial_state: LinXiaState = {
            "user_input": user_input,
            "image_path": image_path,
            "memories": [],
            "graph_context": "",
            "soul_context": "",
            "bio_state": {},
            "relationship_stage": "",
            "inner_monologue": "",
            "emotional_state": {},
            "response_text": "",
            "raw_response": "",
            "self_check_passed": False,
            "self_check_reason": "",
            "retry_count": 0,
            "audio_path": "",
            "image_reaction": "",
            "trace": [],
        }

        if self.compiled:
            final_state = self.compiled.invoke(initial_state)
        else:
            # Fallback: run nodes sequentially without LangGraph
            state = initial_state
            for node_fn in [
                self.nodes.memory_retrieval,
                self.nodes.soul_sensing,
                self.nodes.inner_monologue,
                self.nodes.emotional_response,
                self.nodes.self_check,
                self.nodes.output,
            ]:
                update = node_fn(state)
                state = {**state, **update}
            final_state = state

        return final_state

    def get_trace_summary(self, state: dict) -> str:
        """Format the execution trace for debugging."""
        trace = state.get("trace", [])
        lines = ["[Cognitive Graph Trace]"]
        total_ms = 0
        for step in trace:
            node = step.get("node", "?")
            ms = step.get("duration_ms", 0)
            total_ms += ms
            details = {k: v for k, v in step.items() if k not in ("node", "duration_ms")}
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            lines.append(f"  {node:25s} {ms:6.1f}ms  {detail_str}")
        lines.append(f"  {'TOTAL':25s} {total_ms:6.1f}ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Quick demo with mock components."""
    print("=" * 60)
    print("Cognitive Graph — Lin Xia's Thinking Architecture")
    print("=" * 60)

    if not HAS_LANGGRAPH:
        print("\n[ERROR] langgraph not installed. Run: pip install langgraph")
        print("  Testing fallback mode instead...\n")

    # Create mock nodes for demo
    class MockLLM:
        def __call__(self, prompt, **kwargs):
            if "内心" in prompt:
                return {"choices": [{"text": "用户在和我打招呼，我应该热情一点回应。他今天可能心情不错。"}]}
            return {"choices": [{"text": '<emotion state="happy" intensity="6" trust_delta="+1" affection_delta="+1"/>\n你好呀！今天过得怎么样？'}]}

    class MockRetriever:
        def retrieve(self, q, n_results=3):
            return ["用户喜欢紫色的薰衣草", "用户住在伦敦"]

    class MockKG:
        def to_context_string(self, q):
            return "[知识图谱: 薰衣草→用户→伦敦]"

    # Build minimal nodes
    nodes = CognitiveNodes(
        llm=MockLLM(),
        memory_retriever=MockRetriever(),
        memory_store=None, memory_extractor=None,
        knowledge_graph=MockKG(), graph_extractor=None,
        context_mgr=None, sliding_summary=None,
        emotional_state_model=None, tts_engine=None,
        vision_engine=None, face_memory=None,
        bio_clock=None, relationship=None,
        screen_percept=None, screen_share=None,
        health=None, companion=None,
        smart_ctx=None, system_prompt="你是林夏。",
    )

    graph = LinXiaCognitiveGraph(nodes)

    # Run
    print("\n  User: 你好！")
    result = graph.invoke("你好！")

    print(f"\n  Inner Monologue: {result.get('inner_monologue', 'N/A')}")
    print(f"  Response: {result.get('response_text', 'N/A')}")
    print(f"  Emotion: {result.get('emotional_state', {}).get('mood', 'N/A')}")
    print(f"  Self-Check: {'✅ PASS' if result.get('self_check_passed') else '❌ FAIL'}")
    print(f"\n{graph.get_trace_summary(result)}")


if __name__ == "__main__":
    demo()
