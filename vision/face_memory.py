"""
Face Memory — Visual Person Recognition for Lin Xia

She doesn't just see photos — she REMEMBERS faces and notices changes:
  - "你今天看起来有点累"
  - "你换发型了！"
  - "诶？这是你上次拍照的地方吗？"

Uses perceptual image hashing (no ML model needed) for face/scene matching,
and PIL color analysis for appearance change detection.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from PIL import Image

from memory.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


@dataclass
class FaceProfile:
    """Stored profile of a recognized person."""
    person_id: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    photo_count: int = 0
    avg_brightness: float = 0.0        # Face brightness (tiredness indicator)
    avg_warmth: float = 0.0            # Skin tone warmth
    image_hashes: list = field(default_factory=list)   # perceptual hashes
    appearance_notes: list = field(default_factory=list) # ["dark circles", "new glasses"]


@dataclass
class AppearanceChange:
    """Detected change in someone's appearance between photos."""
    change_type: str = ""       # "brightness", "color", "new_face", "environment"
    description: str = ""       # Human-readable
    confidence: float = 0.0
    emotional_note: str = ""    # What Lin Xia would say


class FaceMemory:
    """
    Visual person memory — tracks faces across sessions and detects changes.
    Uses image hashing + color analysis (no external ML model needed).
    """

    def __init__(self, persist_path: str = "memory_db/face_memory.json",
                 knowledge_graph: KnowledgeGraph = None):
        self.persist_path = persist_path
        self.kg = knowledge_graph
        self.profiles: dict[str, FaceProfile] = {}
        self._load()

    def _load(self):
        """Load face profiles from disk."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                for pid, pdata in data.items():
                    self.profiles[pid] = FaceProfile(**pdata)
                print(f"[FACE] Loaded {len(self.profiles)} face profiles")
            except Exception as e:
                print(f"[FACE] Load failed: {e}")
        else:
            print("[FACE] No face memory found. Starting fresh.")

    def _save(self):
        """Persist face profiles."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = {pid: asdict(p) for pid, p in self.profiles.items()}
        with open(self.persist_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process_photo(self, image_path: str, person_id: str = "user") -> list[AppearanceChange]:
        """
        Process a new photo. Compare with stored profile and detect changes.

        Returns list of appearance changes Lin Xia should comment on.
        """
        if not os.path.exists(image_path):
            return []

        changes = []
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return []

        # Extract features from current photo
        current = self._extract_features(img)
        img_hash = self._perceptual_hash(img)

        # Check if we have a profile for this person
        if person_id in self.profiles:
            profile = self.profiles[person_id]

            # Detect brightness change (tiredness)
            if profile.avg_brightness > 0:
                brightness_diff = current["brightness"] - profile.avg_brightness
                if brightness_diff < -20:
                    changes.append(AppearanceChange(
                        change_type="brightness",
                        description="Face appears darker/more tired",
                        confidence=min(abs(brightness_diff) / 40, 1.0),
                        emotional_note="你今天看起来有点累...是没睡好吗？要注意休息呀。",
                    ))
                elif brightness_diff > 20:
                    changes.append(AppearanceChange(
                        change_type="brightness",
                        description="Face appears brighter/more energetic",
                        confidence=min(abs(brightness_diff) / 40, 1.0),
                        emotional_note="你今天气色好好啊！是有什么开心的事吗？",
                    ))

            # Detect color temperature change (makeup, suntan, etc.)
            if profile.avg_warmth > 0:
                warmth_diff = current["warmth"] - profile.avg_warmth
                if abs(warmth_diff) > 15:
                    if warmth_diff > 0:
                        changes.append(AppearanceChange(
                            change_type="color",
                            description="Warmer skin tone (makeup or sun)",
                            confidence=0.6,
                            emotional_note="你的脸色看起来有点不一样呢...是晒太阳了吗？",
                        ))

            # Check if photo is significantly different from all stored hashes
            if profile.image_hashes:
                min_dist = min(
                    self._hash_distance(img_hash, h)
                    for h in profile.image_hashes[-5:]
                )
                if min_dist > 20:
                    changes.append(AppearanceChange(
                        change_type="appearance",
                        description="Significant visual change from previous photos",
                        confidence=min(min_dist / 30, 1.0),
                        emotional_note="我感觉你跟上次看到的有点不一样...是换造型了吗？",
                    ))

            # Update running averages
            n = profile.photo_count
            profile.avg_brightness = (profile.avg_brightness * n + current["brightness"]) / (n + 1)
            profile.avg_warmth = (profile.avg_warmth * n + current["warmth"]) / (n + 1)
            profile.photo_count += 1
            profile.last_seen = time.time()
            profile.image_hashes.append(img_hash)
            profile.image_hashes = profile.image_hashes[-10:]  # Keep last 10

        else:
            # First time seeing this person!
            self.profiles[person_id] = FaceProfile(
                person_id=person_id,
                first_seen=time.time(),
                last_seen=time.time(),
                photo_count=1,
                avg_brightness=current["brightness"],
                avg_warmth=current["warmth"],
                image_hashes=[img_hash],
            )
            changes.append(AppearanceChange(
                change_type="new_face",
                description="First time seeing this person",
                confidence=1.0,
                emotional_note="这是你第一次给我看照片呢！我要记住你的样子。",
            ))

        # Store in knowledge graph
        if self.kg:
            self.kg.add_entity(person_id, NodeType.PERSON, {
                "last_photo": time.time(),
                "photo_count": self.profiles[person_id].photo_count,
            })
            for change in changes:
                self.kg.add_entity(change.description, NodeType.EVENT, {
                    "timestamp": time.time()
                })
                self.kg.add_relation(person_id, "appearance_change", change.description)

        self._save()
        return changes

    def analyze_scene(self, image_path: str) -> dict:
        """
        Analyze the scene/environment in a photo (not just faces).
        Detects: indoor/outdoor, lighting, time of day, dominant mood.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return {"scene": "unknown"}

        features = self._extract_features(img)

        # Scene classification from color distribution
        scene = {
            "brightness": features["brightness"],
            "warmth": features["warmth"],
            "dominant_colors": features["dominant_colors"],
        }

        # Time of day from brightness + warmth
        if features["brightness"] > 180:
            scene["time_hint"] = "daytime/well-lit"
        elif features["brightness"] > 100:
            scene["time_hint"] = "indoor/moderate light"
        elif features["brightness"] > 50:
            scene["time_hint"] = "evening/dim"
        else:
            scene["time_hint"] = "nighttime/dark"

        # Indoor/outdoor from color diversity
        unique_colors = len(set(features["dominant_colors"]))
        scene["setting"] = "outdoor" if unique_colors >= 3 else "indoor"

        return scene

    def _extract_features(self, img: Image.Image) -> dict:
        """Extract visual features from a PIL image."""
        # Resize for efficiency
        small = img.resize((100, 100))
        pixels = list(small.getdata())

        # Brightness (average luminance)
        brightness = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels) / len(pixels)

        # Warmth (red-blue ratio)
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)
        warmth = avg_r - avg_b

        # Dominant colors
        from collections import Counter
        quantized = [(r // 64 * 64, g // 64 * 64, b // 64 * 64) for r, g, b in pixels]
        color_counts = Counter(quantized)
        dominant = [c for c, _ in color_counts.most_common(5)]

        color_names = []
        for r, g, b in dominant:
            if r > 180 and g < 100 and b < 100: color_names.append("red")
            elif r > 180 and g > 150 and b < 100: color_names.append("warm")
            elif r < 100 and g < 100 and b > 180: color_names.append("blue")
            elif r < 100 and g > 150 and b < 100: color_names.append("green")
            elif r > 180 and g > 180 and b > 180: color_names.append("white")
            elif r < 60 and g < 60 and b < 60: color_names.append("dark")
            else: color_names.append("mixed")

        return {
            "brightness": brightness,
            "warmth": warmth,
            "dominant_colors": color_names,
        }

    def _perceptual_hash(self, img: Image.Image, size: int = 8) -> str:
        """
        Compute perceptual hash of an image.
        Similar images produce similar hashes.
        """
        # Grayscale + resize to 8x8
        gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())

        # Difference hash
        diff = []
        for row in range(size):
            for col in range(size):
                diff.append(pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1])

        # Convert to hex string
        hash_val = sum(bit << i for i, bit in enumerate(diff))
        return format(hash_val, '016x')

    @staticmethod
    def _hash_distance(hash1: str, hash2: str) -> int:
        """Hamming distance between two perceptual hashes."""
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor = val1 ^ val2
        return bin(xor).count('1')


def demo():
    """Demo face memory with synthetic images."""
    from PIL import ImageDraw

    print("=" * 60)
    print("Face Memory — Visual Person Recognition Demo")
    print("=" * 60)

    kg = KnowledgeGraph()
    face_mem = FaceMemory(persist_path="memory_db/face_test.json", knowledge_graph=kg)

    # Create "selfie 1" — bright face
    print("\n[1] First selfie (bright)")
    img1 = Image.new("RGB", (200, 200), (220, 180, 160))
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([60, 40, 140, 160], fill=(240, 200, 180))  # Face
    draw1.ellipse([80, 80, 95, 95], fill=(60, 40, 30))       # Eye
    draw1.ellipse([105, 80, 120, 95], fill=(60, 40, 30))     # Eye
    img1.save("/tmp/selfie1.png")

    changes = face_mem.process_photo("/tmp/selfie1.png", "user")
    for c in changes:
        print(f"  [{c.change_type}] {c.emotional_note}")

    # Create "selfie 2" — tired/dark face
    print("\n[2] Second selfie (tired)")
    img2 = Image.new("RGB", (200, 200), (100, 80, 70))
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([60, 40, 140, 160], fill=(150, 120, 100))  # Darker face
    draw2.ellipse([80, 80, 95, 95], fill=(40, 30, 20))
    draw2.ellipse([105, 80, 120, 95], fill=(40, 30, 20))
    img2.save("/tmp/selfie2.png")

    changes = face_mem.process_photo("/tmp/selfie2.png", "user")
    for c in changes:
        print(f"  [{c.change_type}] {c.emotional_note} (conf={c.confidence:.0%})")

    # Scene analysis
    print("\n[3] Scene analysis")
    scene = face_mem.analyze_scene("/tmp/selfie1.png")
    print(f"  Brightness: {scene['brightness']:.0f}")
    print(f"  Time hint: {scene['time_hint']}")
    print(f"  Setting: {scene['setting']}")

    # Cleanup
    os.remove("/tmp/selfie1.png")
    os.remove("/tmp/selfie2.png")
    if os.path.exists("memory_db/face_test.json"):
        os.remove("memory_db/face_test.json")


if __name__ == "__main__":
    demo()
