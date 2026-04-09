"""
Context Manager — Central context window controller.
Manages the allocation of the limited context window across:
  - System prompt (fixed)
  - Emotional state summary (dynamic, compressed)
  - Retrieved memories (from RAG)
  - Recent conversation turns (sliding window)

Implements the "compressed infinite context" illusion.
"""

import json
import time
from dataclasses import dataclass, field


@dataclass
class EmotionalState:
    """Tracks Lin Xia's current emotional state as a numerical vector."""
    mood: str = "calm"           # Current dominant emotion
    mood_intensity: float = 5.0  # 1-10 scale
    trust_level: float = 7.0     # Trust toward user (1-10)
    affection: float = 6.0       # Affection level (1-10)
    recent_events: list = field(default_factory=list)  # Last N emotional events
    last_updated: float = 0.0

    def to_prompt_block(self) -> str:
        """Convert emotional state to a compact string for injection into system prompt."""
        events_str = "; ".join(self.recent_events[-3:]) if self.recent_events else "No recent events"
        return (
            f"[Emotional State: mood={self.mood} (intensity: {self.mood_intensity:.0f}/10), "
            f"trust={self.trust_level:.0f}/10, affection={self.affection:.0f}/10. "
            f"Recent: {events_str}]"
        )

    def update_from_analysis(self, analysis: dict):
        """Update state based on LLM analysis of the latest exchange."""
        if "mood" in analysis:
            self.mood = analysis["mood"]
        if "mood_intensity" in analysis:
            self.mood_intensity = float(analysis["mood_intensity"])
        if "trust_delta" in analysis:
            self.trust_level = max(1, min(10, self.trust_level + float(analysis["trust_delta"])))
        if "affection_delta" in analysis:
            self.affection = max(1, min(10, self.affection + float(analysis["affection_delta"])))
        if "event_summary" in analysis:
            self.recent_events.append(analysis["event_summary"])
            self.recent_events = self.recent_events[-5:]  # Keep last 5
        self.last_updated = time.time()


class ContextManager:
    """
    Manages the context window budget for optimal information density.

    Budget allocation (configurable):
      - System prompt:     ~15% of context
      - Emotional state:   ~5% of context
      - Retrieved memories:~20% of context
      - Recent turns:      ~60% of context
    """

    def __init__(self, max_context_tokens: int = 2048, system_prompt: str = ""):
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt
        self.emotional_state = EmotionalState()
        self.conversation_history: list[dict] = []
        self.retrieved_memories: list[str] = []

        # Budget allocation (as fractions of max_context)
        self.budget = {
            "system": 0.15,
            "emotion": 0.05,
            "memory": 0.20,
            "turns": 0.60,
        }

    def add_turn(self, role: str, content: str):
        """Add a conversation turn to history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def set_memories(self, memories: list[str]):
        """Set retrieved memories for the current turn."""
        self.retrieved_memories = memories

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (Chinese: ~1.5 chars/token, English: ~4 chars/token)."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _build_memory_block(self, token_budget: int) -> str:
        """Build memory injection block within token budget."""
        if not self.retrieved_memories:
            return ""

        block = "[Relevant Memories]\n"
        for mem in self.retrieved_memories:
            candidate = block + f"- {mem}\n"
            if self._estimate_tokens(candidate) > token_budget:
                break
            block = candidate
        return block

    def _select_recent_turns(self, token_budget: int) -> list[dict]:
        """Select as many recent turns as fit within the token budget."""
        selected = []
        used_tokens = 0

        for turn in reversed(self.conversation_history):
            turn_tokens = self._estimate_tokens(turn["content"]) + 10  # Overhead
            if used_tokens + turn_tokens > token_budget:
                break
            selected.insert(0, turn)
            used_tokens += turn_tokens

        return selected

    def build_prompt(self) -> str:
        """
        Build the complete prompt with optimal context allocation.
        Returns a Qwen chat-template-compatible prompt string.
        """
        # Calculate token budgets
        budget_tokens = {
            k: int(v * self.max_context_tokens)
            for k, v in self.budget.items()
        }

        # 1. System prompt (fixed)
        system_block = self.system_prompt

        # 2. Emotional state (dynamic)
        emotion_block = self.emotional_state.to_prompt_block()

        # 3. Memory retrieval (dynamic)
        memory_block = self._build_memory_block(budget_tokens["memory"])

        # Combine system prompt components
        full_system = f"{system_block}\n\n{emotion_block}"
        if memory_block:
            full_system += f"\n\n{memory_block}"

        # 4. Recent turns — use remaining budget
        used_tokens = self._estimate_tokens(full_system)
        remaining_budget = self.max_context_tokens - used_tokens - 200  # Reserve for generation
        recent_turns = self._select_recent_turns(remaining_budget)

        # Build Qwen chat template
        parts = [f"<|im_start|>system\n{full_system}<|im_end|>"]
        for turn in recent_turns:
            parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def get_context_stats(self) -> dict:
        """Return statistics about current context usage."""
        prompt = self.build_prompt()
        total_tokens = self._estimate_tokens(prompt)
        return {
            "total_estimated_tokens": total_tokens,
            "max_context": self.max_context_tokens,
            "utilization": f"{total_tokens / self.max_context_tokens * 100:.1f}%",
            "total_history_turns": len(self.conversation_history),
            "included_turns": len(self._select_recent_turns(
                int(self.max_context_tokens * self.budget["turns"])
            )),
            "emotional_state": self.emotional_state.to_prompt_block(),
            "memories_injected": len(self.retrieved_memories),
        }
