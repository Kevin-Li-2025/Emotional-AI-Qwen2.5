"""
Sliding Summary — Compresses old conversation turns into emotional state summaries.
Uses the LLM itself to extract the "emotional gist" of older exchanges,
allowing infinite conversation length without losing emotional continuity.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SUMMARIZE_PROMPT = """Analyze the following conversation segment between a user and Lin Xia (an emotional AI character).
Extract a compact emotional summary.

CONVERSATION SEGMENT:
{conversation}

OUTPUT FORMAT (strict JSON):
{{
    "mood": "<Lin Xia's current dominant emotion: e.g. happy, hurt, angry, calm, anxious, playful>",
    "mood_intensity": <1-10>,
    "trust_delta": <-3 to +3, how much trust changed in this segment>,
    "affection_delta": <-3 to +3, how much affection changed>,
    "event_summary": "<one-sentence summary of what happened emotionally>",
    "unresolved": "<any unresolved emotional tension, or 'none'>"
}}"""


class SlidingSummary:
    """
    Maintains a sliding window of full conversation turns,
    summarizing older turns into compressed emotional state blocks.
    """

    def __init__(self, window_size: int = 8, llm=None):
        """
        Args:
            window_size: Number of recent turns to keep in full detail.
            llm: A llama_cpp.Llama instance for generating summaries.
        """
        self.window_size = window_size
        self.llm = llm
        self.full_history: list[dict] = []  # Complete history
        self.summaries: list[dict] = []     # Compressed summaries of old segments

    def add_turn(self, role: str, content: str):
        """Add a turn and trigger summarization if needed."""
        self.full_history.append({"role": role, "content": content})

        # When history exceeds 2x window, summarize the overflow
        if len(self.full_history) > self.window_size * 2:
            overflow = self.full_history[:self.window_size]
            self.full_history = self.full_history[self.window_size:]
            self._summarize_segment(overflow)

    def _summarize_segment(self, segment: list[dict]):
        """Summarize a segment of conversation into an emotional state block."""
        if self.llm is None:
            # Fallback: simple concatenation
            texts = [f"{m['role']}: {m['content']}" for m in segment]
            self.summaries.append({
                "mood": "unknown",
                "event_summary": f"({len(segment)} turns summarized)",
                "raw_turns": len(segment)
            })
            return

        # Format segment for the LLM
        convo_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Lin Xia'}: {m['content']}"
            for m in segment
        )

        prompt = f"<|im_start|>system\nYou are a conversation analyst.<|im_end|>\n<|im_start|>user\n{SUMMARIZE_PROMPT.format(conversation=convo_text)}<|im_end|>\n<|im_start|>assistant\n"

        try:
            output = self.llm(
                prompt,
                max_tokens=300,
                stop=["<|im_end|>"],
                temperature=0.1,  # Low temp for factual extraction
            )
            response = output["choices"][0]["text"].strip()
            summary = json.loads(response)
            summary["raw_turns"] = len(segment)
            self.summaries.append(summary)
        except Exception as e:
            # Graceful fallback
            self.summaries.append({
                "mood": "unknown",
                "event_summary": f"({len(segment)} turns, parse error)",
                "raw_turns": len(segment),
                "error": str(e)
            })

    def get_recent_turns(self) -> list[dict]:
        """Get the most recent turns (full detail, within the window)."""
        return self.full_history[-self.window_size:]

    def get_compressed_history(self) -> str:
        """Get the full compressed history as a string for prompt injection."""
        if not self.summaries:
            return ""

        lines = ["[Conversation History Summary]"]
        for i, s in enumerate(self.summaries):
            event = s.get("event_summary", "unknown event")
            mood = s.get("mood", "unknown")
            lines.append(f"  Segment {i+1}: {event} (mood: {mood})")

        unresolved = [s.get("unresolved", "none") for s in self.summaries if s.get("unresolved") != "none"]
        if unresolved:
            lines.append(f"  Unresolved tensions: {'; '.join(unresolved)}")

        return "\n".join(lines)

    def get_total_turns_processed(self) -> int:
        """Total turns across summaries and current window."""
        summarized = sum(s.get("raw_turns", 0) for s in self.summaries)
        return summarized + len(self.full_history)

    def get_stats(self) -> dict:
        return {
            "total_turns_processed": self.get_total_turns_processed(),
            "segments_summarized": len(self.summaries),
            "current_window_size": len(self.full_history),
            "window_limit": self.window_size,
        }
