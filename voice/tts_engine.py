"""
TTS Engine — Emotional text-to-speech integration.
Wraps CosyVoice or GPT-SoVITS to give Lin Xia an emotionally expressive voice.
Parses emotion tags from model output and maps them to voice parameters.
"""

import re
import os
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VoiceParams:
    """Parameters that control the emotional quality of generated speech."""
    speed: float = 1.0      # 0.5 (slow/sad) to 1.5 (fast/excited)
    pitch: float = 0.0      # -2 (low/sad) to +2 (high/excited)
    energy: float = 1.0     # 0.5 (quiet/sad) to 1.5 (loud/angry)
    emotion: str = "neutral" # Emotion label for the TTS engine


# Mapping from Lin Xia's emotion tags to voice parameters
EMOTION_VOICE_MAP = {
    "gentle":   VoiceParams(speed=0.9, pitch=0.5,  energy=0.8,  emotion="gentle"),
    "happy":    VoiceParams(speed=1.1, pitch=1.0,  energy=1.2,  emotion="happy"),
    "excited":  VoiceParams(speed=1.2, pitch=1.5,  energy=1.4,  emotion="happy"),
    "sad":      VoiceParams(speed=0.8, pitch=-1.0, energy=0.6,  emotion="sad"),
    "hurt":     VoiceParams(speed=0.85, pitch=-0.5, energy=0.7, emotion="sad"),
    "angry":    VoiceParams(speed=1.0, pitch=0.0,  energy=1.5,  emotion="angry"),
    "cold":     VoiceParams(speed=0.9, pitch=-0.5, energy=0.8,  emotion="neutral"),
    "playful":  VoiceParams(speed=1.1, pitch=1.0,  energy=1.1,  emotion="happy"),
    "shy":      VoiceParams(speed=0.85, pitch=0.5, energy=0.7,  emotion="gentle"),
    "anxious":  VoiceParams(speed=1.1, pitch=0.5,  energy=1.0,  emotion="fear"),
    "neutral":  VoiceParams(speed=1.0, pitch=0.0,  energy=1.0,  emotion="neutral"),
}


class TTSEngine:
    """
    Text-to-Speech engine that converts Lin Xia's responses to emotional speech.
    Supports multiple backends (CosyVoice, GPT-SoVITS, or mocked for testing).
    """

    def __init__(self, backend: str = "mock", model_path: str = None):
        """
        Args:
            backend: "cosyvoice", "gpt_sovits", or "mock" (for testing without TTS)
            model_path: Path to the TTS model/checkpoint.
        """
        self.backend = backend
        self.model_path = model_path
        self.output_dir = Path("./voice_output")
        self.output_dir.mkdir(exist_ok=True)

        if backend == "cosyvoice":
            self._init_cosyvoice()
        elif backend == "gpt_sovits":
            self._init_gpt_sovits()

    def _init_cosyvoice(self):
        """Initialize CosyVoice backend."""
        try:
            # CosyVoice requires specific installation
            # pip install cosyvoice or clone from github
            print("[TTS] CosyVoice backend selected. Ensure CosyVoice is installed.")
        except Exception as e:
            print(f"[TTS ERROR] Failed to init CosyVoice: {e}")
            self.backend = "mock"

    def _init_gpt_sovits(self):
        """Initialize GPT-SoVITS backend."""
        try:
            print("[TTS] GPT-SoVITS backend selected. Ensure API server is running.")
        except Exception as e:
            print(f"[TTS ERROR] Failed to init GPT-SoVITS: {e}")
            self.backend = "mock"

    @staticmethod
    def parse_emotion_tag(text: str) -> tuple[str, str]:
        """
        Extract emotion tag from model output.
        E.g., "[sad] 你这样说让我很难过。" → ("sad", "你这样说让我很难过。")
        """
        match = re.match(r'\[(\w+)\]\s*(.*)', text, re.DOTALL)
        if match:
            return match.group(1).lower(), match.group(2)
        return "neutral", text

    def get_voice_params(self, emotion_tag: str) -> VoiceParams:
        """Map an emotion tag to voice parameters."""
        return EMOTION_VOICE_MAP.get(emotion_tag, EMOTION_VOICE_MAP["neutral"])

    def synthesize(self, text: str, emotion_tag: str = None, output_file: str = None) -> str:
        """
        Synthesize speech from text with emotional modulation.

        Args:
            text: The text to speak (without emotion tags).
            emotion_tag: The emotion to apply. If None, will be parsed from text.
            output_file: Output audio file path. Auto-generated if None.

        Returns:
            Path to the generated audio file.
        """
        if emotion_tag is None:
            emotion_tag, text = self.parse_emotion_tag(text)

        params = self.get_voice_params(emotion_tag)

        if output_file is None:
            import time
            output_file = str(self.output_dir / f"linxia_{int(time.time())}.wav")

        if self.backend == "mock":
            return self._mock_synthesize(text, params, output_file)
        elif self.backend == "cosyvoice":
            return self._cosyvoice_synthesize(text, params, output_file)
        elif self.backend == "gpt_sovits":
            return self._gpt_sovits_synthesize(text, params, output_file)

        return output_file

    def _mock_synthesize(self, text: str, params: VoiceParams, output_file: str) -> str:
        """Mock synthesis — logs parameters instead of producing audio."""
        print(f"[TTS Mock] Emotion: {params.emotion} | Speed: {params.speed} | "
              f"Pitch: {params.pitch} | Energy: {params.energy}")
        print(f"[TTS Mock] Text: {text[:80]}...")
        # Write metadata instead of audio
        meta_path = output_file.replace(".wav", ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"text": text, "params": vars(params)}, f, ensure_ascii=False, indent=2)
        return meta_path

    def _cosyvoice_synthesize(self, text: str, params: VoiceParams, output_file: str) -> str:
        """CosyVoice synthesis with emotion control."""
        # Placeholder for CosyVoice API integration
        # In production, this would call the CosyVoice inference API
        # with instruction-based emotion control
        instruction = f"Use a {params.emotion} tone, speak at {params.speed}x speed"
        print(f"[CosyVoice] Instruction: {instruction}")
        print(f"[CosyVoice] Text: {text[:80]}...")
        # TODO: Integrate actual CosyVoice inference
        return output_file

    def _gpt_sovits_synthesize(self, text: str, params: VoiceParams, output_file: str) -> str:
        """GPT-SoVITS synthesis via API."""
        # Placeholder for GPT-SoVITS API integration
        # Typical API: POST /tts with text, ref_audio, and speed parameters
        print(f"[GPT-SoVITS] Emotion: {params.emotion}")
        print(f"[GPT-SoVITS] Text: {text[:80]}...")
        # TODO: Integrate actual GPT-SoVITS API call
        return output_file
