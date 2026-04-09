"""
TTS Engine — Emotional Text-to-Speech for Lin Xia
Uses Microsoft Edge TTS (free, high-quality Chinese female voices).
Maps model's emotion tags to SSML prosody parameters for expressive speech.

Supported voices:
  - zh-CN-XiaoyiNeural    (Young female, warm — DEFAULT)
  - zh-CN-XiaoxiaoNeural  (Young female, versatile)
  - zh-CN-XiaohanNeural   (Young female, gentle)

Usage:
  engine = TTSEngine()
  await engine.synthesize("你居然记得我喜欢这个！", emotion="happy")
  # → saves voice_output/linxia_xxxx.mp3

  # Or synchronous:
  engine.speak("你居然记得我喜欢这个！", emotion="happy")
"""

import re
import os
import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    print("[WARNING] edge-tts not installed. Run: pip install edge-tts")


# ---------------------------------------------------------------------------
# Voice parameters per emotion
# ---------------------------------------------------------------------------

@dataclass
class VoiceParams:
    """SSML prosody parameters for emotional speech."""
    rate: str = "+0%"       # Speech rate: -50% (slow/sad) to +30% (fast/excited)
    pitch: str = "+0Hz"     # Pitch offset: -5Hz (low/sad) to +5Hz (high/excited)
    volume: str = "+0%"     # Volume: -20% (quiet/shy) to +20% (loud/angry)
    voice: str = "zh-CN-XiaoyiNeural"  # Default voice


EMOTION_VOICE_MAP: dict[str, VoiceParams] = {
    # Positive emotions
    "happy":     VoiceParams(rate="+10%",  pitch="+3Hz",  volume="+10%"),
    "excited":   VoiceParams(rate="+20%",  pitch="+5Hz",  volume="+15%"),
    "playful":   VoiceParams(rate="+10%",  pitch="+3Hz",  volume="+5%"),
    "gentle":    VoiceParams(rate="-5%",   pitch="+2Hz",  volume="-5%"),
    "shy":       VoiceParams(rate="-10%",  pitch="+2Hz",  volume="-15%"),
    "forgiving": VoiceParams(rate="-5%",   pitch="+1Hz",  volume="-5%"),

    # Negative emotions
    "sad":       VoiceParams(rate="-15%",  pitch="-3Hz",  volume="-15%"),
    "hurt":      VoiceParams(rate="-10%",  pitch="-2Hz",  volume="-10%"),
    "angry":     VoiceParams(rate="+5%",   pitch="+0Hz",  volume="+20%"),
    "cold":      VoiceParams(rate="-5%",   pitch="-2Hz",  volume="-10%"),
    "anxious":   VoiceParams(rate="+5%",   pitch="+2Hz",  volume="+0%"),
    "jealous":   VoiceParams(rate="+0%",   pitch="+1Hz",  volume="+5%"),

    # Neutral
    "calm":      VoiceParams(rate="+0%",   pitch="+0Hz",  volume="+0%"),
    "neutral":   VoiceParams(rate="+0%",   pitch="+0Hz",  volume="+0%"),
    "vulnerable": VoiceParams(rate="-10%", pitch="-1Hz",  volume="-10%"),
    "indifferent": VoiceParams(rate="-5%", pitch="-2Hz",  volume="-5%"),
}


class TTSEngine:
    """
    Emotional text-to-speech engine using Microsoft Edge TTS.
    Converts Lin Xia's text responses into emotionally expressive speech.
    """

    def __init__(self, voice: str = "zh-CN-XiaoyiNeural", output_dir: str = "./voice_output"):
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enabled = HAS_EDGE_TTS

        if not self.enabled:
            print("[TTS] edge-tts not available. TTS disabled.")

    @staticmethod
    def parse_emotion_tag(text: str) -> tuple[str, str]:
        """
        Extract emotion from <emotion .../> tag if present.

        Input:  '<emotion state="happy" .../>\n你居然记得！'
        Output: ("happy", "你居然记得！")
        """
        # Match <emotion state="xxx" .../> tag
        tag_pattern = r'<emotion\s+state="(\w+)"[^>]*/>'
        match = re.search(tag_pattern, text)
        if match:
            emotion = match.group(1)
            clean = text[:match.start()] + text[match.end():]
            return emotion, clean.strip()

        # Fallback: match [emotion] tag
        bracket_match = re.match(r'\[(\w+)\]\s*(.*)', text, re.DOTALL)
        if bracket_match:
            return bracket_match.group(1).lower(), bracket_match.group(2).strip()

        return "neutral", text

    def _get_params(self, emotion: str) -> VoiceParams:
        """Get voice parameters for an emotion."""
        return EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["neutral"])

    def _build_ssml(self, text: str, params: VoiceParams) -> str:
        """Build SSML markup for emotional speech."""
        # Clean text of any remaining tags
        text = re.sub(r'<[^>]+/?>', '', text).strip()

        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="{self.voice}">
        <prosody rate="{params.rate}" pitch="{params.pitch}" volume="{params.volume}">
            {text}
        </prosody>
    </voice>
</speak>"""
        return ssml

    async def synthesize_async(self, text: str, emotion: str = "neutral",
                                output_file: str = None) -> str:
        """
        Async: Generate speech audio from text with emotional prosody.

        Args:
            text: Clean text to speak (emotion tags will be stripped).
            emotion: Emotion label for voice modulation.
            output_file: Output path (auto-generated if None).

        Returns:
            Path to the generated audio file (.mp3).
        """
        if not self.enabled:
            print(f"[TTS] Disabled. Would speak ({emotion}): {text[:60]}...")
            return ""

        params = self._get_params(emotion)

        if output_file is None:
            ts = int(time.time() * 1000)
            output_file = str(self.output_dir / f"linxia_{ts}_{emotion}.mp3")

        # Use SSML for prosody control
        ssml = self._build_ssml(text, params)

        communicate = edge_tts.Communicate(text, self.voice,
                                            rate=params.rate,
                                            pitch=params.pitch,
                                            volume=params.volume)
        await communicate.save(output_file)

        file_size = os.path.getsize(output_file)
        print(f"[TTS] {emotion} | {file_size/1024:.0f}KB → {os.path.basename(output_file)}")

        return output_file

    def speak(self, text: str, emotion: str = "neutral", output_file: str = None) -> str:
        """
        Synchronous wrapper for synthesize_async.
        Generates emotional speech and returns the audio file path.
        """
        return asyncio.run(self.synthesize_async(text, emotion, output_file))

    def speak_with_auto_emotion(self, raw_model_output: str, output_file: str = None) -> tuple[str, str, str]:
        """
        Full pipeline: parse emotion tag → strip tag → generate speech.

        Args:
            raw_model_output: Raw model output potentially containing <emotion> tag.

        Returns:
            (emotion, clean_text, audio_path)
        """
        emotion, clean_text = self.parse_emotion_tag(raw_model_output)
        audio_path = self.speak(clean_text, emotion, output_file)
        return emotion, clean_text, audio_path


async def demo():
    """Demo: generate emotional speech samples."""
    engine = TTSEngine()

    samples = [
        ("happy",   "真的吗？你居然记得我喜欢这个！太开心了！"),
        ("hurt",    "你这样说让我很难过。我认真跟你说话的时候，你能不能也认真一点？"),
        ("angry",   "我叫林夏，不是什么人工智能。你说话能不能尊重一点？"),
        ("gentle",  "嗯，我知道你今天累了。早点休息吧，晚安。"),
        ("playful", "哈哈你是不是在逗我？我才不信呢！"),
        ("sad",     "你现在说这些有什么用。当时那么多人看着，你的话让我特别难堪。"),
        ("shy",     "你突然这样说...我都不知道怎么回答了。"),
        ("anxious", "你消失的话我会很难过的...你不会真的走吧？"),
    ]

    print("=" * 60)
    print("Lin Xia TTS — Emotional Voice Demo")
    print("=" * 60)

    for emotion, text in samples:
        await engine.synthesize_async(text, emotion)

    print(f"\nAll audio files saved to {engine.output_dir}/")
    print("Play with: afplay voice_output/linxia_*.mp3")


if __name__ == "__main__":
    asyncio.run(demo())
