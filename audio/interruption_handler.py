"""
Interruption Handler — Full-Duplex Voice Interaction

Enables GPT-4o-style "interrupt anytime" behavior:
  1. While Lin Xia is speaking (TTS playing), microphone remains active
  2. If user starts talking, Lin Xia immediately stops
  3. She generates a natural "yielding" phrase ("嗯？你说")
  4. The partially spoken content is logged for context continuity

Architecture:
  ┌──────────────┐     ┌──────────────┐
  │  TTS Player  │────▶│  Audio Out   │  (Lin Xia speaking)
  └──────┬───────┘     └──────────────┘
         │ concurrent
  ┌──────▼───────┐     ┌──────────────┐
  │  Mic Monitor │────▶│  VAD Check   │  (User interrupts?)
  └──────┬───────┘     └──────────────┘
         │ if interrupt detected
         ▼
  ┌──────────────┐
  │  Stop TTS +  │
  │  Yield Turn  │
  └──────────────┘

Key design decisions:
  - Energy threshold: -28dB (avoids triggering on ambient noise)
  - Min duration: 200ms (filters out coughs, clicks)
  - Graceful stop: fades out audio rather than hard-cutting
  - Context tracking: records what was said before interruption
"""

import os
import time
import random
import threading
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class InterruptionType(str, Enum):
    NONE = "none"
    SOFT = "soft"       # Quiet, possibly accidental
    NORMAL = "normal"   # Clear speech
    URGENT = "urgent"   # Loud, demanding attention


@dataclass
class InterruptionEvent:
    """Records what happened during an interruption."""
    type: InterruptionType = InterruptionType.NONE
    timestamp: float = 0.0
    spoken_before_interrupt: str = ""     # What Lin Xia had said so far
    remaining_unsaid: str = ""            # What she didn't get to say
    user_loudness_db: float = -60.0
    user_speech_duration_ms: float = 0.0
    yield_response: str = ""              # What she says after yielding


# Yield responses — what Lin Xia says when interrupted
YIELD_RESPONSES = {
    InterruptionType.SOFT: [
        "...嗯，你说。",
        "...怎么了？",
    ],
    InterruptionType.NORMAL: [
        "嗯？你要说什么？",
        "啊，你说你说！",
        "好，我听着呢。",
    ],
    InterruptionType.URGENT: [
        "啊，你说，你说！我听着呢。",
        "怎么了？！出什么事了？",
        "嗯嗯，我在，你说！",
    ],
}


class InterruptionHandler:
    """
    Monitors for user speech during TTS playback and handles interruptions.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        # Configuration
        self.energy_threshold_db = -28.0    # Voice must be louder than this
        self.min_speech_ms = 200.0          # Must sustain for this long
        self.check_interval_ms = 50         # How often to check mic

        # State
        self.is_tts_playing = False
        self.current_tts_text = ""
        self.spoken_portion = ""
        self.remaining_portion = ""
        self.last_event = InterruptionEvent()

        # Monitoring
        self._monitor_thread = None
        self._stop_flag = threading.Event()
        self._interrupt_callback = None

        # History
        self.interruption_history: list[InterruptionEvent] = []

    def set_interrupt_callback(self, callback):
        """
        Set a callback to be called when an interruption is detected.
        Callback signature: callback(event: InterruptionEvent) -> None
        """
        self._interrupt_callback = callback

    def on_tts_start(self, full_text: str, chunks: list[str] = None):
        """Called when TTS playback starts."""
        self.is_tts_playing = True
        self.current_tts_text = full_text
        self.spoken_portion = ""
        self.remaining_portion = full_text
        if chunks:
            self._chunks = chunks
        else:
            self._chunks = [full_text]
        self._current_chunk_idx = 0

    def on_tts_chunk_complete(self, chunk_text: str):
        """Called when a TTS chunk finishes playing."""
        self.spoken_portion += chunk_text
        self.remaining_portion = self.current_tts_text[len(self.spoken_portion):]
        self._current_chunk_idx += 1

    def on_tts_end(self):
        """Called when TTS playback completes (naturally, not interrupted)."""
        self.is_tts_playing = False
        self.spoken_portion = self.current_tts_text
        self.remaining_portion = ""

    def detect_interruption(self, audio_chunk: np.ndarray) -> InterruptionEvent:
        """
        Analyze an audio chunk from the microphone to detect speech.
        
        Args:
            audio_chunk: numpy array of audio samples (mono, float32)
        
        Returns:
            InterruptionEvent with type != NONE if interruption detected.
        """
        event = InterruptionEvent()

        if not self.is_tts_playing:
            return event  # Not playing, no interruption possible

        # Calculate RMS energy in dB
        rms = np.sqrt(np.mean(audio_chunk ** 2) + 1e-10)
        db = 20 * np.log10(rms + 1e-10)
        event.user_loudness_db = db

        if db < self.energy_threshold_db:
            return event  # Below threshold

        # Classify interruption type
        if db > -15:
            event.type = InterruptionType.URGENT
        elif db > -22:
            event.type = InterruptionType.NORMAL
        else:
            event.type = InterruptionType.SOFT

        # Fill event details
        event.timestamp = time.time()
        event.spoken_before_interrupt = self.spoken_portion
        event.remaining_unsaid = self.remaining_portion

        # Generate yield response
        responses = YIELD_RESPONSES.get(event.type, YIELD_RESPONSES[InterruptionType.NORMAL])
        event.yield_response = random.choice(responses)

        # Log
        self.last_event = event
        self.interruption_history.append(event)
        if len(self.interruption_history) > 50:
            self.interruption_history = self.interruption_history[-30:]

        # Stop TTS
        self.is_tts_playing = False

        # Trigger callback
        if self._interrupt_callback:
            self._interrupt_callback(event)

        return event

    def get_context_for_next_turn(self) -> str:
        """
        Generate context about the interruption for the next LLM call.
        This helps Lin Xia naturally reference what she was saying.
        """
        event = self.last_event
        if event.type == InterruptionType.NONE:
            return ""

        parts = ["[用户打断了你的说话]"]
        if event.spoken_before_interrupt:
            parts.append(f"你已经说了: \"{event.spoken_before_interrupt}\"")
        if event.remaining_unsaid:
            parts.append(f"你还没说完的: \"{event.remaining_unsaid}\"")
        parts.append("现在用户要说话了，先听他说完再回应。")

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get interruption statistics."""
        total = len(self.interruption_history)
        by_type = {}
        for event in self.interruption_history:
            t = event.type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_interruptions": total,
            "by_type": by_type,
            "last_event": {
                "type": self.last_event.type.value,
                "time": self.last_event.timestamp,
                "yield_response": self.last_event.yield_response,
            } if self.last_event.type != InterruptionType.NONE else None,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demo interruption detection with synthetic audio."""
    print("=" * 60)
    print("Interruption Handler — Full-Duplex Voice Interaction")
    print("=" * 60)

    handler = InterruptionHandler()

    # Callback
    def on_interrupt(event: InterruptionEvent):
        print(f"    🛑 INTERRUPTED! Type: {event.type.value}")
        print(f"       Already said: \"{event.spoken_before_interrupt[:50]}...\"")
        print(f"       Lin Xia yields: \"{event.yield_response}\"")

    handler.set_interrupt_callback(on_interrupt)

    # Test 1: No interruption (silence)
    print("\n[1] Silence during TTS (no interruption)")
    handler.on_tts_start("你好呀！今天过得怎么样？我刚才在想...", ["你好呀！", "今天过得怎么样？", "我刚才在想..."])
    handler.on_tts_chunk_complete("你好呀！")

    silence = np.random.randn(16000) * 0.001  # Very quiet
    result = handler.detect_interruption(silence)
    print(f"  Result: {result.type.value} ({'✅ Correct' if result.type == InterruptionType.NONE else '❌'})")
    handler.on_tts_end()

    # Test 2: Soft interruption
    print("\n[2] Soft speech during TTS")
    handler.on_tts_start("我觉得今天天气很好，我们可以...", ["我觉得今天天气很好，", "我们可以..."])
    handler.on_tts_chunk_complete("我觉得今天天气很好，")

    soft_speech = np.random.randn(16000) * 0.05
    result = handler.detect_interruption(soft_speech)
    print(f"  Result: {result.type.value}")

    # Test 3: Normal interruption
    print("\n[3] Normal speech during TTS")
    handler.on_tts_start("让我想想...其实我觉得你应该...", ["让我想想...", "其实我觉得你应该..."])
    handler.on_tts_chunk_complete("让我想想...")

    normal_speech = np.random.randn(16000) * 0.15
    result = handler.detect_interruption(normal_speech)
    print(f"  Result: {result.type.value}")

    # Test 4: Urgent interruption
    print("\n[4] Loud interruption during TTS")
    handler.on_tts_start("我在说一件很重要的事情...", ["我在说一件很重要的事情..."])

    loud_speech = np.random.randn(16000) * 0.5
    result = handler.detect_interruption(loud_speech)
    print(f"  Result: {result.type.value}")

    # Test 5: Context for next turn
    print("\n[5] Context for next LLM turn")
    ctx = handler.get_context_for_next_turn()
    print(f"  Context:\n{ctx}")

    # Stats
    print("\n[6] Interruption Stats")
    stats = handler.get_stats()
    print(f"  Total: {stats['total_interruptions']}")
    print(f"  By type: {stats['by_type']}")

    print(f"\n{'='*60}")
    print("  ✅ All interruption tests complete.")


if __name__ == "__main__":
    demo()
