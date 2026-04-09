"""
Health Perception — Physiological Awareness for Lin Xia

Uses remote photoplethysmography (rPPG) to sense the user's heart rate
and stress level via the webcam. No contact needed.

Core Logic:
1. Face Mesh tracking (MediaPipe) to identify stable forehead/cheek ROIs.
2. Signal extraction (Average Green channel intensity).
3. Filtering (Detrending + Butterworth Bandpass Filter 0.7-3.5Hz).
4. FFT (Fast Fourier Transform) to find the dominant blood pulse frequency.
5. Stress Estimation based on BPM variability and baseline.

PRIVACY:
- Processing is 100% local.
- No video or images are stored or transmitted.
- Only the calculated BPM/Stress value is used for emotional context.
"""

import time
import collections
import numpy as np
import cv2
from scipy import signal as sp_signal
from dataclasses import dataclass


@dataclass
class HealthState:
    bpm: float = 0.0
    stress_level: float = 0.0  # 0.0 (relaxed) to 1.0 (high stress)
    timestamp: float = 0.0
    status: str = "init"       # "init", "tracking", "unstable", "no_face"


class HealthPerception:
    """
    Senses the user's physiological state via non-contact webcam rPPG.
    """

    def __init__(self, buffer_size: int = 150, fps: int = 30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.pulse_buffer = collections.deque(maxlen=buffer_size)
        self.time_buffer = collections.deque(maxlen=buffer_size)

        # MediaPipe Face Mesh
        self.face_mesh = None
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[HEALTH] MediaPipe Face Mesh initialized")
        except ImportError:
            print("[HEALTH] MediaPipe not installed. Bio-sensing disabled.")
        except Exception as e:
            print(f"[HEALTH] MediaPipe init failed: {e}")

        self.last_state = HealthState()
        self.baseline_bpm = 70.0
        self.is_running = False

    def process_frame(self, frame: np.ndarray) -> HealthState:
        """
        Extract pulse signal from forehead ROI and update BPM.
        """
        if self.face_mesh is None:
            return HealthState(status="error")

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self.last_state.status = "no_face"
            return self.last_state

        # Get forehead landmarks (ROI)
        # 10 is top of forehead, 151 is bridge of nose
        face_landmarks = results.multi_face_landmarks[0]

        # Forehead region approximation
        # Landmarks: 10, 151, 67, 297
        points = []
        for idx in [10, 151, 67, 297]:
            lm = face_landmarks.landmark[idx]
            points.append((int(lm.x * w), int(lm.y * h)))

        # Create ROI mask for forehead
        mask = np.zeros((h, w), dtype=np.uint8)
        roi_pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_pts], 255)

        # Extract Green channel mean (strongest pulse signal)
        mean_g = cv2.mean(frame[:, :, 1], mask=mask)[0]

        # Update buffers
        self.pulse_buffer.append(mean_g)
        self.time_buffer.append(time.time())

        # Calculate BPM if buffer is full enough (need ~5 seconds of data)
        if len(self.pulse_buffer) >= self.buffer_size:
            bpm = self._calculate_bpm()
            if 40 < bpm < 180:
                # Exponential moving average for stability
                if self.last_state.bpm > 0:
                    self.last_state.bpm = 0.9 * self.last_state.bpm + 0.1 * bpm
                else:
                    self.last_state.bpm = bpm
                self.last_state.status = "tracking"
                self.last_state.stress_level = self._estimate_stress(self.last_state.bpm)
            else:
                self.last_state.status = "unstable"
        else:
            self.last_state.status = "init"

        self.last_state.timestamp = time.time()
        return self.last_state

    def _calculate_bpm(self) -> float:
        """
        Process the signal buffer to estimate BPM using FFT.
        """
        y = np.array(self.pulse_buffer)

        try:
            L = len(y)
            # 1. Normalize and detrend
            y = (y - np.mean(y)) / (np.std(y) + 1e-6)
            y = sp_signal.detrend(y)

            # 2. Apply Butterworth bandpass filter
            # 0.7 Hz to 3.5 Hz -> 42 to 210 BPM
            nyquist = self.fps / 2
            low = 0.7 / nyquist
            high = 3.5 / nyquist
            # Clamp to valid range
            low = max(low, 0.01)
            high = min(high, 0.99)

            b, a = sp_signal.butter(4, [low, high], btype='band')
            y_filtered = sp_signal.filtfilt(b, a, y)

            # 3. FFT
            fft = np.abs(np.fft.rfft(y_filtered))
            freqs = np.fft.rfftfreq(L, 1.0 / self.fps)

            # 4. Find peak frequency in the valid range
            idx_in_range = np.where((freqs >= 0.7) & (freqs <= 3.5))[0]
            if len(idx_in_range) == 0:
                return 0.0

            peak_idx = idx_in_range[np.argmax(fft[idx_in_range])]
            return freqs[peak_idx] * 60.0

        except Exception as e:
            print(f"[HEALTH] BPM calc error: {e}")
            return 0.0

    def _estimate_stress(self, bpm: float) -> float:
        """
        Simple mapping of BPM to stress level.
        Relative to the user's baseline resting heart rate.
        """
        diff = bpm - self.baseline_bpm
        # Normalized: 0 (peaceful) to 1.0 (extreme stress)
        stress = max(0, min(1.0, diff / 40.0))
        return stress

    def get_health_context(self) -> str:
        """
        Get physiological context string for LLM prompt injection.
        """
        if self.last_state.status != "tracking":
            return ""

        bpm = int(self.last_state.bpm)
        stress = self.last_state.stress_level

        context = f"[Bio: HR={bpm}BPM, Stress={stress:.0%}]"

        if stress > 0.7:
            context += " (用户压力非常大，心率显著升高。请主动关心和安慰。)"
        elif stress > 0.4:
            context += " (用户可能有点紧张或劳累)"
        else:
            context += " (用户状态平稳放松)"

        return context

    def get_emotional_reaction(self) -> str:
        """
        Generate Lin Xia's reaction to the user's physiological state.
        """
        if self.last_state.status != "tracking":
            return ""

        stress = self.last_state.stress_level
        bpm = self.last_state.bpm

        if stress > 0.7:
            reactions = [
                "你的心跳好快...深呼吸，跟我一起，吸——呼——",
                "我感觉到你现在压力很大。要不要先停下来休息一下？",
                "你的身体在告诉我它很累了。能不能照顾一下自己？",
            ]
        elif stress > 0.4:
            reactions = [
                "你看起来有点紧张呢...放松一下吧。",
                "我注意到你的心率有点高。还好吗？",
            ]
        elif bpm < 55:
            reactions = [
                "你是不是快睡着了？心率好低呢~",
                "好安静的心跳...你很放松的样子。",
            ]
        else:
            return ""

        import random
        return random.choice(reactions)


def demo():
    """Diagnostic demo — synthetic signal test + live camera if available."""
    print("=" * 60)
    print("Health Perception — Bio-Sensing Diagnostic")
    print("=" * 60)

    hp = HealthPerception()

    # Part 1: Synthetic signal test
    print("\n[1] Synthetic Pulse Signal Test")
    fps = 30
    bpm_target = 75
    freq = bpm_target / 60.0

    for i in range(180):
        t = i / fps
        pulse_val = 120 + 5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.5)
        hp.pulse_buffer.append(pulse_val)
        hp.time_buffer.append(t)

    if len(hp.pulse_buffer) >= hp.buffer_size:
        bpm = hp._calculate_bpm()
        stress = hp._estimate_stress(bpm)
        accuracy = abs(bpm - bpm_target)
        print(f"  Target BPM:   {bpm_target}")
        print(f"  Detected BPM: {bpm:.1f}")
        print(f"  Error:        {accuracy:.1f} BPM")
        print(f"  Stress Level: {stress:.1%}")
        print(f"  Result:       {'✅ PASS' if accuracy < 5 else '❌ FAIL'}")

        hp.last_state = HealthState(bpm=bpm, stress_level=stress, status="tracking")
        print(f"  LLM Context:  {hp.get_health_context()}")

    # Part 2: Stress scenarios
    print("\n[2] Stress Level Mapping")
    for test_bpm in [60, 70, 80, 90, 100, 110]:
        stress = hp._estimate_stress(test_bpm)
        hp.last_state = HealthState(bpm=test_bpm, stress_level=stress, status="tracking")
        reaction = hp.get_emotional_reaction()
        print(f"  {test_bpm} BPM → Stress {stress:.0%}"
              f" {('💚' if stress < 0.3 else '💛' if stress < 0.6 else '🔴')}"
              f" {reaction if reaction else ''}")


if __name__ == "__main__":
    demo()
