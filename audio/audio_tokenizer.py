"""
Audio Tokenizer — Discrete Audio Representation for Emotional AI
Converts speech into analyzable features without the text bottleneck.

Two modes:
  1. Full codec (EnCodec/SNAC): Discrete tokens for future audio-to-audio LLM
  2. Feature mode (librosa): Extract prosody features directly from waveform

The key insight: audio contains emotional information that ASR destroys.
A hesitant "嗯...其实我..." and a confident "嗯，其实我" produce identical
text but carry very different emotional weight. We preserve this.
"""

import os
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[AUDIO] librosa not installed. Run: pip install librosa soundfile")

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProsodyFeatures:
    """Paralinguistic features extracted from audio — the information ASR throws away."""
    # Pitch (F0)
    pitch_mean: float = 0.0        # Hz — voice register (low=sad, high=excited)
    pitch_std: float = 0.0         # Hz — pitch variation (monotone vs. expressive)
    pitch_range: float = 0.0       # Hz — full range of pitch
    pitch_contour: str = "flat"    # "rising", "falling", "flat", "varied"

    # Energy
    energy_mean: float = 0.0       # dB — overall loudness
    energy_std: float = 0.0        # dB — loudness variation
    energy_peak: float = 0.0       # dB — max loudness (emphasis moments)

    # Tempo
    speech_rate: float = 0.0       # syllables/sec estimate
    voiced_ratio: float = 0.0      # ratio of voiced frames (speech vs. silence)

    # Pauses & Hesitation
    pause_count: int = 0           # number of pauses > 300ms
    max_pause_ms: float = 0.0      # longest pause duration
    has_hesitation: bool = False    # detected "嗯" / filler patterns

    # Spectral (timbre)
    spectral_centroid: float = 0.0 # brightness of voice
    mfcc_summary: list = field(default_factory=list)  # first 5 MFCC means

    # Duration
    duration_sec: float = 0.0

    # Derived emotion hint
    emotion_hint: str = "neutral"
    confidence: float = 0.0

    def to_context_string(self) -> str:
        """Format for LLM prompt injection."""
        parts = [f"[User Voice Tone: {self.emotion_hint}"]

        details = []
        if self.has_hesitation:
            details.append("hesitant")
        if self.speech_rate > 5.0:
            details.append("fast speech")
        elif self.speech_rate < 2.5 and self.speech_rate > 0:
            details.append("slow speech")
        if self.energy_mean > -20:
            details.append("loud")
        elif self.energy_mean < -35:
            details.append("quiet/whispering")
        if self.pitch_std > 50:
            details.append("very expressive pitch")
        elif self.pitch_std < 15 and self.pitch_std > 0:
            details.append("monotone")
        if self.pause_count > 2:
            details.append(f"{self.pause_count} pauses")

        if details:
            parts.append(f" — {', '.join(details)}")
        parts.append("]")
        return "".join(parts)


class AudioTokenizer:
    """
    Extract discrete audio tokens and prosody features from speech.
    This is the component that lets LinXia "hear" HOW things are said.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.enabled = HAS_LIBROSA

        if not self.enabled:
            print("[AUDIO TOKENIZER] Disabled — librosa not available")

    def load_audio(self, audio_path: str) -> tuple:
        """Load audio file and resample to target rate."""
        if not os.path.exists(audio_path):
            return None, 0

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return y, sr

    def extract_prosody(self, audio_path: str) -> ProsodyFeatures:
        """
        Extract comprehensive prosody features from an audio file.
        This is the core function — it hears what ASR cannot.
        """
        if not self.enabled:
            return ProsodyFeatures()

        t0 = time.time()
        y, sr = self.load_audio(audio_path)
        if y is None:
            return ProsodyFeatures()

        features = ProsodyFeatures()
        features.duration_sec = len(y) / sr

        # 1. Pitch (F0) analysis
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                features.pitch_mean = float(np.mean(voiced_f0))
                features.pitch_std = float(np.std(voiced_f0))
                features.pitch_range = float(np.max(voiced_f0) - np.min(voiced_f0))

                # Pitch contour direction
                if len(voiced_f0) > 10:
                    first_quarter = np.mean(voiced_f0[:len(voiced_f0)//4])
                    last_quarter = np.mean(voiced_f0[-len(voiced_f0)//4:])
                    diff = last_quarter - first_quarter
                    if diff > 20:
                        features.pitch_contour = "rising"
                    elif diff < -20:
                        features.pitch_contour = "falling"
                    elif features.pitch_std > 40:
                        features.pitch_contour = "varied"
                    else:
                        features.pitch_contour = "flat"

                features.voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0))
        except Exception:
            pass

        # 2. Energy analysis
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10)
            features.energy_mean = float(np.mean(rms_db))
            features.energy_std = float(np.std(rms_db))
            features.energy_peak = float(np.max(rms_db))
        except Exception:
            pass

        # 3. Pause detection
        try:
            # Find silent intervals
            intervals = librosa.effects.split(y, top_db=30)
            if len(intervals) > 1:
                pauses = []
                for i in range(1, len(intervals)):
                    gap_start = intervals[i - 1][1]
                    gap_end = intervals[i][0]
                    gap_ms = (gap_end - gap_start) / sr * 1000
                    if gap_ms > 300:  # Significant pause
                        pauses.append(gap_ms)

                features.pause_count = len(pauses)
                features.max_pause_ms = max(pauses) if pauses else 0
                features.has_hesitation = features.pause_count > 0 and features.max_pause_ms > 500
        except Exception:
            pass

        # 4. Speech rate estimation
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            # Rough syllable rate: tempo / 60 * 2 (approximate for Chinese)
            features.speech_rate = float(tempo[0] / 60 * 2) if len(tempo) > 0 else 0
        except Exception:
            pass

        # 5. Spectral features (timbre)
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.spectral_centroid = float(np.mean(centroid))

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.mfcc_summary = [float(np.mean(mfcc[i])) for i in range(min(5, mfcc.shape[0]))]
        except Exception:
            pass

        # 6. Derive emotion hint from features
        features.emotion_hint, features.confidence = self._infer_emotion(features)

        return features

    def _infer_emotion(self, f: ProsodyFeatures) -> tuple:
        """
        Infer emotion from prosody features.
        This replaces what would normally require a dedicated emotion recognition model.
        """
        scores = {
            "angry": 0.0,
            "sad": 0.0,
            "excited": 0.0,
            "calm": 0.0,
            "anxious": 0.0,
            "hesitant": 0.0,
        }

        # High pitch + high energy + fast → angry or excited
        if f.pitch_mean > 250:
            scores["excited"] += 0.3
            scores["angry"] += 0.2
        if f.energy_mean > -20:
            scores["angry"] += 0.3
            scores["excited"] += 0.2
        if f.speech_rate > 5:
            scores["excited"] += 0.2
            scores["anxious"] += 0.2

        # Low pitch + low energy + slow → sad
        if f.pitch_mean > 0 and f.pitch_mean < 180:
            scores["sad"] += 0.3
        if f.energy_mean < -35:
            scores["sad"] += 0.3
            scores["calm"] += 0.1
        if f.speech_rate > 0 and f.speech_rate < 3:
            scores["sad"] += 0.2
            scores["calm"] += 0.2

        # Hesitation patterns
        if f.has_hesitation:
            scores["hesitant"] += 0.5
            scores["anxious"] += 0.2
        if f.pause_count > 3:
            scores["hesitant"] += 0.3

        # Monotone + moderate → calm
        if f.pitch_std > 0 and f.pitch_std < 20:
            scores["calm"] += 0.3

        # High variation → emotionally expressive
        if f.pitch_std > 50:
            scores["excited"] += 0.2
            scores["angry"] += 0.1

        # Find winner
        if max(scores.values()) < 0.2:
            return "neutral", 0.5

        emotion = max(scores, key=scores.get)
        confidence = min(scores[emotion], 1.0)
        return emotion, round(confidence, 2)

    def tokenize_to_codes(self, audio_path: str) -> dict:
        """
        Convert audio to discrete feature codes (lightweight alternative to EnCodec).
        Uses MFCC quantization as a codec-free discrete representation.
        """
        if not self.enabled:
            return {"codes": [], "n_frames": 0}

        y, sr = self.load_audio(audio_path)
        if y is None:
            return {"codes": [], "n_frames": 0}

        # Extract MFCCs as our "codec"
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=320)

        # Quantize to discrete codes (simple k-means style)
        # Each frame → 13-dim vector → quantize each dim to 0-255
        mfcc_min = mfcc.min(axis=1, keepdims=True)
        mfcc_max = mfcc.max(axis=1, keepdims=True)
        mfcc_range = mfcc_max - mfcc_min + 1e-10
        codes = ((mfcc - mfcc_min) / mfcc_range * 255).astype(int)

        return {
            "codes": codes.tolist(),
            "n_frames": codes.shape[1],
            "n_coeffs": codes.shape[0],
            "frame_rate": sr / 320,  # frames per second
            "duration": len(y) / sr,
        }


def demo():
    """Generate a synthetic test audio and analyze it."""
    print("=" * 60)
    print("Audio Tokenizer — Prosody Extraction Demo")
    print("=" * 60)

    tokenizer = AudioTokenizer()

    # Create test audio with librosa (a tone with varying pitch)
    sr = 16000
    duration = 3.0

    # Simulate hesitant speech: tone → pause → tone
    t1 = np.linspace(0, 1.0, int(sr * 1.0))
    tone1 = 0.3 * np.sin(2 * np.pi * 200 * t1)  # 200Hz, low pitch

    pause = np.zeros(int(sr * 0.8))  # 800ms pause (hesitation)

    t2 = np.linspace(0, 1.2, int(sr * 1.2))
    tone2 = 0.5 * np.sin(2 * np.pi * 280 * t2)  # 280Hz, higher pitch

    audio = np.concatenate([tone1, pause, tone2])
    test_path = "/tmp/test_hesitant.wav"
    sf.write(test_path, audio, sr)

    # Extract prosody
    features = tokenizer.extract_prosody(test_path)
    print(f"\nProsody Features:")
    print(f"  Pitch: mean={features.pitch_mean:.0f}Hz, std={features.pitch_std:.0f}Hz, contour={features.pitch_contour}")
    print(f"  Energy: mean={features.energy_mean:.1f}dB, peak={features.energy_peak:.1f}dB")
    print(f"  Speech rate: {features.speech_rate:.1f} syl/s")
    print(f"  Pauses: {features.pause_count} (max {features.max_pause_ms:.0f}ms)")
    print(f"  Hesitation: {'YES' if features.has_hesitation else 'no'}")
    print(f"  Voiced ratio: {features.voiced_ratio:.1%}")
    print(f"  Duration: {features.duration_sec:.1f}s")
    print(f"\n  → Emotion hint: {features.emotion_hint} (confidence {features.confidence})")
    print(f"  → LLM context: {features.to_context_string()}")

    # Tokenize to codes
    codes = tokenizer.tokenize_to_codes(test_path)
    print(f"\n  Audio tokens: {codes['n_frames']} frames × {codes['n_coeffs']} coefficients")
    print(f"  Frame rate: {codes['frame_rate']:.0f} fps")

    os.remove(test_path)

    # Test with generated speech if we have voice_output
    voice_files = list(Path("voice_output").glob("*.mp3"))
    if voice_files:
        print(f"\n{'='*60}")
        print(f"Analyzing real Lin Xia voice output: {voice_files[0].name}")
        print(f"{'='*60}")
        features = tokenizer.extract_prosody(str(voice_files[0]))
        print(f"  Pitch: {features.pitch_mean:.0f}Hz ± {features.pitch_std:.0f}")
        print(f"  Energy: {features.energy_mean:.1f}dB")
        print(f"  Pauses: {features.pause_count}")
        print(f"  → Emotion: {features.emotion_hint} ({features.confidence})")
        print(f"  → Context: {features.to_context_string()}")


if __name__ == "__main__":
    demo()
