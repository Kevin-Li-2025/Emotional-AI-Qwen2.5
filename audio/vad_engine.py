"""
VAD Engine — Full-Duplex Voice Activity Detection + Ambient Sensing

1. INTERRUPTION DETECTION: When the user starts speaking while Lin Xia is
   talking, she stops immediately and listens — just like a real person.
   Uses energy-based VAD (no Silero dependency required, pure librosa).

2. AMBIENT SENSING: Analyzes background noise to infer the user's environment.
   - Quiet room → whisper mode (gentle, soft voice)
   - Noisy environment → louder, clearer speech
   - Music playing → detect and comment ("你在听音乐吗？")
"""

import os
import time
import numpy as np
from dataclasses import dataclass

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class AmbientProfile:
    """Profile of the user's acoustic environment."""
    noise_level_db: float = -60.0       # Background noise floor
    is_quiet: bool = True               # < -40dB
    is_noisy: bool = False              # > -20dB
    has_music: bool = False             # Detected tonal background
    environment_hint: str = "quiet"     # "quiet", "moderate", "noisy", "music"

    # Recommended voice adjustments
    volume_adjust: str = "normal"       # "whisper", "normal", "loud"
    rate_adjust: str = "0%"             # Edge TTS rate adjustment

    def to_context_string(self) -> str:
        parts = [f"[Environment: {self.environment_hint}"]
        if self.is_quiet:
            parts.append(" — user is in a quiet space, speak softly")
        elif self.has_music:
            parts.append(" — user seems to be listening to music")
        elif self.is_noisy:
            parts.append(" — it's noisy around the user, speak clearly")
        parts.append("]")
        return "".join(parts)


@dataclass
class VADResult:
    """Result of voice activity detection on an audio segment."""
    has_speech: bool = False
    speech_ratio: float = 0.0          # Ratio of speech frames
    speech_segments: list = None       # List of (start_sec, end_sec) tuples
    is_interruption: bool = False      # User speaking during playback
    loudness_db: float = -60.0
    duration_sec: float = 0.0


class VADEngine:
    """
    Lightweight Voice Activity Detection + Ambient Analysis.
    No external VAD model required — uses energy + zero-crossing rate.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.enabled = HAS_LIBROSA

        # State for real-time monitoring
        self.is_linxia_speaking = False
        self.ambient_profile = AmbientProfile()

        # Thresholds (tunable)
        self.speech_threshold_db = -35   # Below this = silence
        self.music_threshold = 0.3       # Spectral flatness for music detection
        self.interruption_energy_db = -30  # Must be this loud to interrupt

    def detect_speech(self, audio_path: str) -> VADResult:
        """
        Detect speech segments in an audio file.
        Returns speech timestamps and whether it constitutes an interruption.
        """
        if not self.enabled:
            return VADResult()

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        result = VADResult()
        result.duration_sec = len(y) / sr

        # RMS energy per frame
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        result.loudness_db = float(np.mean(rms_db))

        # Classify frames as speech/silence
        speech_frames = rms_db > self.speech_threshold_db
        result.speech_ratio = float(np.mean(speech_frames))
        result.has_speech = result.speech_ratio > 0.1

        # Extract speech segments
        segments = []
        hop_sec = 512 / sr
        in_speech = False
        seg_start = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                seg_start = i * hop_sec
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((round(seg_start, 2), round(i * hop_sec, 2)))
                in_speech = False
        if in_speech:
            segments.append((round(seg_start, 2), round(len(speech_frames) * hop_sec, 2)))

        result.speech_segments = segments

        # Check for interruption
        if self.is_linxia_speaking and result.has_speech:
            if result.loudness_db > self.interruption_energy_db:
                result.is_interruption = True

        return result

    def analyze_ambient(self, audio_path: str) -> AmbientProfile:
        """
        Analyze the background acoustic environment.
        Used to adjust Lin Xia's voice parameters for the setting.
        """
        if not self.enabled:
            return AmbientProfile()

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        profile = AmbientProfile()

        # 1. Overall noise floor
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)

        # Find quiet sections (non-speech) for noise floor
        quiet_frames = rms_db[rms_db < self.speech_threshold_db]
        if len(quiet_frames) > 0:
            profile.noise_level_db = float(np.mean(quiet_frames))
        else:
            profile.noise_level_db = float(np.mean(rms_db))

        # 2. Classify environment
        noise = profile.noise_level_db
        if noise < -45:
            profile.is_quiet = True
            profile.is_noisy = False
            profile.environment_hint = "quiet"
            profile.volume_adjust = "whisper"
            profile.rate_adjust = "-10%"  # Slower, softer in quiet
        elif noise > -20:
            profile.is_quiet = False
            profile.is_noisy = True
            profile.environment_hint = "noisy"
            profile.volume_adjust = "loud"
            profile.rate_adjust = "+5%"   # Slightly faster, clearer
        else:
            profile.is_quiet = False
            profile.is_noisy = False
            profile.environment_hint = "moderate"
            profile.volume_adjust = "normal"
            profile.rate_adjust = "0%"

        # 3. Music detection via spectral flatness
        try:
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            mean_flatness = float(np.mean(flatness))

            # Tonal audio (music) has LOW spectral flatness
            # Noise has HIGH spectral flatness
            # Speech is in between
            if mean_flatness < 0.01 and not profile.is_quiet:
                profile.has_music = True
                profile.environment_hint = "music"
        except Exception:
            pass

        self.ambient_profile = profile
        return profile

    def set_speaking_state(self, is_speaking: bool):
        """Update whether Lin Xia is currently outputting audio."""
        self.is_linxia_speaking = is_speaking

    def get_interruption_response(self, vad_result: VADResult) -> str:
        """Get an appropriate response when the user interrupts."""
        if not vad_result.is_interruption:
            return ""

        if vad_result.loudness_db > -15:
            # Loud interruption — urgent
            return "啊，你说，你说！我听着呢。"
        elif vad_result.loudness_db > -25:
            # Normal interruption
            return "嗯？怎么了，你要说什么？"
        else:
            # Soft interruption
            return "...嗯，你说。"


def demo():
    """Test VAD and ambient analysis with synthetic audio."""
    import soundfile as sf

    print("=" * 60)
    print("VAD Engine — Full-Duplex + Ambient Sensing Demo")
    print("=" * 60)

    engine = VADEngine()
    sr = 16000

    # Test 1: Speech detection
    print("\n[1] Speech detection test")
    t = np.linspace(0, 2.0, sr * 2)
    speech_sim = 0.5 * np.sin(2 * np.pi * 250 * t) * (1 + 0.3 * np.random.randn(len(t)))
    sf.write("/tmp/vad_speech.wav", speech_sim, sr)
    result = engine.detect_speech("/tmp/vad_speech.wav")
    print(f"  Has speech: {result.has_speech} (ratio: {result.speech_ratio:.2f})")
    print(f"  Segments: {result.speech_segments[:3]}")
    print(f"  Loudness: {result.loudness_db:.1f}dB")

    # Test 2: Quiet room
    print("\n[2] Quiet room test")
    quiet = 0.001 * np.random.randn(sr * 2)
    sf.write("/tmp/vad_quiet.wav", quiet, sr)
    profile = engine.analyze_ambient("/tmp/vad_quiet.wav")
    print(f"  Noise floor: {profile.noise_level_db:.1f}dB")
    print(f"  Environment: {profile.environment_hint}")
    print(f"  Voice adjust: volume={profile.volume_adjust}, rate={profile.rate_adjust}")
    print(f"  Context: {profile.to_context_string()}")

    # Test 3: Noisy environment
    print("\n[3] Noisy environment test")
    noise = 0.3 * np.random.randn(sr * 2)
    sf.write("/tmp/vad_noisy.wav", noise, sr)
    profile = engine.analyze_ambient("/tmp/vad_noisy.wav")
    print(f"  Noise floor: {profile.noise_level_db:.1f}dB")
    print(f"  Environment: {profile.environment_hint}")
    print(f"  Voice adjust: volume={profile.volume_adjust}, rate={profile.rate_adjust}")
    print(f"  Context: {profile.to_context_string()}")

    # Test 4: Interruption
    print("\n[4] Interruption detection test")
    engine.set_speaking_state(True)  # Lin Xia is talking
    result = engine.detect_speech("/tmp/vad_speech.wav")
    print(f"  Lin Xia speaking: True")
    print(f"  User spoke: {result.has_speech}")
    print(f"  Is interruption: {result.is_interruption}")
    if result.is_interruption:
        print(f"  Response: {engine.get_interruption_response(result)}")

    # Cleanup
    for f in ["/tmp/vad_speech.wav", "/tmp/vad_quiet.wav", "/tmp/vad_noisy.wav"]:
        os.remove(f)


if __name__ == "__main__":
    demo()
