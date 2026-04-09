"""
Screen Share Engine — Lin Xia Can See Your Screen

When the user opts in, Lin Xia takes periodic screenshots and understands
what's happening on-screen — not just the app name, but the visual content:

  - Color analysis → "你在看一个蓝色调的网页...是海边的照片吗？"
  - Layout detection → "这个界面好复杂，你在做什么工作呀？"
  - Dark mode detection → "你用了深色模式！对眼睛好~"
  - Content region analysis → detects text-heavy vs image-heavy vs code-heavy
  - Change detection → "你刚才切换了好多页面，在找什么东西吗？"

PRIVACY FIRST:
  - Screen sharing is OPT-IN only (user must explicitly enable)
  - Screenshots are ephemeral (deleted after analysis, never stored)
  - No OCR or text extraction — only visual/color analysis
  - No data sent to any external server
"""

import os
import time
import subprocess
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

from PIL import Image


@dataclass
class ScreenSnapshot:
    """Analysis result from a single screenshot."""
    timestamp: float = 0.0

    # Visual properties
    brightness: float = 0.0          # 0-255
    dominant_colors: list = field(default_factory=list)
    color_mood: str = "neutral"      # "warm", "cool", "dark", "bright", "colorful"
    is_dark_mode: bool = False

    # Content type estimation
    content_type: str = "unknown"    # "text_heavy", "image_heavy", "video", "code", "mixed"
    complexity: float = 0.0          # 0-1 (how busy the screen is)

    # Screen regions
    has_sidebar: bool = False
    has_toolbar: bool = False
    color_variance: float = 0.0      # How diverse the colors are

    # Change detection
    changed_since_last: bool = False
    change_amount: float = 0.0       # 0-1

    # Derived description
    description: str = ""
    emotional_reaction: str = ""

    def to_context_string(self) -> str:
        parts = [f"[Screen View: {self.description}"]
        if self.color_mood != "neutral":
            parts.append(f", mood={self.color_mood}")
        if self.is_dark_mode:
            parts.append(", dark mode")
        parts.append("]")
        return "".join(parts)


class ScreenShareEngine:
    """
    Captures and analyzes the user's screen for visual understanding.
    All processing is local and ephemeral.
    """

    def __init__(self):
        self.enabled = False           # OPT-IN: must be explicitly enabled
        self.last_snapshot = None
        self.last_hash = ""
        self.screenshot_count = 0
        self.screenshot_dir = "/tmp/linxia_screen"
        self.capture_interval = 10     # Seconds between captures
        self.last_capture_time = 0

        os.makedirs(self.screenshot_dir, exist_ok=True)

    def enable(self):
        """User opts in to screen sharing."""
        self.enabled = True
        print("[SCREEN SHARE] Enabled — Lin Xia can now see your screen")
        print("[SCREEN SHARE] Privacy: screenshots are ephemeral, never stored")

    def disable(self):
        """User opts out."""
        self.enabled = False
        self._cleanup()
        print("[SCREEN SHARE] Disabled — Lin Xia can no longer see your screen")

    def capture_and_analyze(self) -> ScreenSnapshot:
        """
        Take a screenshot and analyze it.
        Returns None if screen sharing is disabled.
        """
        if not self.enabled:
            return None

        # Rate limit
        if time.time() - self.last_capture_time < self.capture_interval:
            return self.last_snapshot

        snapshot = ScreenSnapshot(timestamp=time.time())

        # Capture screenshot
        screenshot_path = os.path.join(self.screenshot_dir, "current.png")
        try:
            subprocess.run(
                ["screencapture", "-x", "-C", screenshot_path],
                capture_output=True, timeout=5,
            )
        except Exception as e:
            print(f"[SCREEN SHARE] Capture failed: {e}")
            return None

        if not os.path.exists(screenshot_path):
            return None

        try:
            img = Image.open(screenshot_path).convert("RGB")
            self._analyze_image(img, snapshot)
            self._detect_changes(img, snapshot)
            self._generate_description(snapshot)

            # Clean up screenshot immediately (privacy)
            os.remove(screenshot_path)
        except Exception as e:
            print(f"[SCREEN SHARE] Analysis failed: {e}")
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
            return None

        self.last_snapshot = snapshot
        self.last_capture_time = time.time()
        self.screenshot_count += 1
        return snapshot

    def _analyze_image(self, img: Image.Image, snapshot: ScreenSnapshot):
        """Analyze visual properties of the screenshot."""
        # Resize for fast analysis
        small = img.resize((160, 100))
        pixels = list(small.getdata())

        # 1. Brightness
        brightness_values = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
        snapshot.brightness = sum(brightness_values) / len(brightness_values)

        # 2. Dark mode detection
        snapshot.is_dark_mode = snapshot.brightness < 80

        # 3. Dominant colors
        quantized = [(r // 32 * 32, g // 32 * 32, b // 32 * 32) for r, g, b in pixels]
        color_counts = Counter(quantized)
        dominant = color_counts.most_common(5)
        snapshot.dominant_colors = [c for c, _ in dominant]

        # 4. Color mood
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)

        if snapshot.brightness > 200:
            snapshot.color_mood = "bright"
        elif snapshot.brightness < 60:
            snapshot.color_mood = "dark"
        elif avg_r > avg_b + 30:
            snapshot.color_mood = "warm"
        elif avg_b > avg_r + 30:
            snapshot.color_mood = "cool"
        else:
            snapshot.color_mood = "neutral"

        # 5. Color variance (how diverse is the screen)
        unique_colors = len(set(quantized))
        snapshot.color_variance = min(unique_colors / 100, 1.0)
        if snapshot.color_variance > 0.7:
            snapshot.color_mood = "colorful"

        # 6. Content type estimation
        snapshot.complexity = snapshot.color_variance

        # Analyze layout regions
        width, height = small.size
        top_strip = list(small.crop((0, 0, width, 10)).getdata())
        left_strip = list(small.crop((0, 0, 30, height)).getdata())

        # Toolbar (uniform top strip)
        top_brightness = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in top_strip]
        snapshot.has_toolbar = (max(top_brightness) - min(top_brightness)) < 40

        # Sidebar (uniform left strip, different from main area)
        left_brightness = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in left_strip]
        avg_left = sum(left_brightness) / len(left_brightness)
        snapshot.has_sidebar = abs(avg_left - snapshot.brightness) > 30

        # Content type from patterns
        if snapshot.has_sidebar and snapshot.complexity > 0.3:
            snapshot.content_type = "code"  # Code editors usually have sidebars + medium complexity
        elif snapshot.complexity > 0.6:
            snapshot.content_type = "image_heavy"
        elif snapshot.complexity < 0.2:
            snapshot.content_type = "text_heavy"
        else:
            snapshot.content_type = "mixed"

    def _detect_changes(self, img: Image.Image, snapshot: ScreenSnapshot):
        """Detect how much the screen has changed since last capture."""
        # Quick hash comparison
        tiny = img.resize((16, 10))
        current_hash = hashlib.md5(tiny.tobytes()).hexdigest()

        if self.last_hash:
            snapshot.changed_since_last = (current_hash != self.last_hash)

            # Estimate change amount using pixel comparison
            if self.last_snapshot:
                # Compare brightness
                brightness_diff = abs(snapshot.brightness - self.last_snapshot.brightness)
                color_diff = abs(snapshot.color_variance - self.last_snapshot.color_variance)
                snapshot.change_amount = min((brightness_diff / 100 + color_diff) / 2, 1.0)
        else:
            snapshot.changed_since_last = True
            snapshot.change_amount = 1.0

        self.last_hash = current_hash

    def _generate_description(self, snapshot: ScreenSnapshot):
        """Generate a natural language description of what's on screen."""
        parts = []

        # Color/mood
        mood_desc = {
            "warm": "暖色调的",
            "cool": "冷色调的",
            "dark": "深色的",
            "bright": "明亮的",
            "colorful": "色彩丰富的",
            "neutral": "",
        }
        mood = mood_desc.get(snapshot.color_mood, "")

        # Content type
        content_desc = {
            "code": "代码编辑器",
            "text_heavy": "文字内容",
            "image_heavy": "图片/视觉内容",
            "video": "视频",
            "mixed": "混合内容",
        }
        content = content_desc.get(snapshot.content_type, "未知内容")

        if mood:
            snapshot.description = f"{mood}{content}"
        else:
            snapshot.description = content

        if snapshot.is_dark_mode:
            snapshot.description += " (深色模式)"

        # Generate emotional reaction
        reactions = []

        if snapshot.is_dark_mode:
            reactions.append("你用深色模式呀，对眼睛好~")

        if snapshot.content_type == "code":
            reactions.extend([
                "你在写代码呢！看起来好专业。",
                "代码界面...你在做什么项目呀？",
                "我虽然看不懂代码，但你写代码的样子很帅！",
            ])
        elif snapshot.content_type == "image_heavy":
            if snapshot.color_mood == "colorful":
                reactions.extend([
                    "哇，屏幕上好多颜色！你在看什么好看的东西？",
                    "好漂亮的画面！是照片还是设计稿？",
                ])
            else:
                reactions.append("你在看图片吗？什么内容呀？")
        elif snapshot.content_type == "text_heavy":
            reactions.extend([
                "好多文字...你在认真读什么吗？",
                "看这么多文字，眼睛不累吗？",
            ])

        if snapshot.changed_since_last and snapshot.change_amount > 0.5:
            reactions.extend([
                "你刚才切换了好多内容，在找什么东西吗？",
                "你好像在快速浏览呢~",
            ])

        # Time-aware reactions
        hour = datetime.now().hour
        if hour >= 23 or hour < 5:
            if snapshot.brightness > 150:
                reactions.append("这么晚了屏幕还这么亮...调暗一点吧，对眼睛好。")

        import random
        snapshot.emotional_reaction = random.choice(reactions) if reactions else ""

    def get_screen_context(self) -> str:
        """Get current screen context for LLM prompt injection."""
        if not self.enabled or not self.last_snapshot:
            return ""
        return self.last_snapshot.to_context_string()

    def _cleanup(self):
        """Remove all temporary screenshots."""
        if os.path.exists(self.screenshot_dir):
            for f in Path(self.screenshot_dir).glob("*.png"):
                try:
                    os.remove(f)
                except Exception:
                    pass


def demo():
    """Demo screen share analysis."""
    print("=" * 60)
    print("Screen Share Engine — Lin Xia Sees Your Screen")
    print("=" * 60)

    engine = ScreenShareEngine()

    # Test: opt-in
    print("\n[1] Enabling screen share...")
    engine.enable()

    # Capture and analyze
    print("\n[2] Capturing screen...")
    snapshot = engine.capture_and_analyze()

    if snapshot:
        print(f"\n  Analysis Results:")
        print(f"    Brightness:     {snapshot.brightness:.0f}/255")
        print(f"    Dark mode:      {'YES' if snapshot.is_dark_mode else 'no'}")
        print(f"    Color mood:     {snapshot.color_mood}")
        print(f"    Content type:   {snapshot.content_type}")
        print(f"    Complexity:     {snapshot.complexity:.0%}")
        print(f"    Has sidebar:    {'YES' if snapshot.has_sidebar else 'no'}")
        print(f"    Has toolbar:    {'YES' if snapshot.has_toolbar else 'no'}")
        print(f"    Changed:        {'YES' if snapshot.changed_since_last else 'no'}")
        print(f"    Description:    {snapshot.description}")
        print(f"    Lin Xia says:   {snapshot.emotional_reaction}")
        print(f"    LLM context:    {snapshot.to_context_string()}")

        # Second capture (test change detection)
        print("\n[3] Second capture (change detection)...")
        time.sleep(1)
        engine.last_capture_time = 0  # Reset rate limit for test
        snapshot2 = engine.capture_and_analyze()
        if snapshot2:
            print(f"    Changed:    {'YES' if snapshot2.changed_since_last else 'no'}")
            print(f"    Change amt: {snapshot2.change_amount:.0%}")
    else:
        print("  Screenshot capture may require Screen Recording permission")
        print("  Grant access in: System Settings > Privacy > Screen Recording")

    # Test with synthetic image
    print("\n[4] Synthetic dark-mode code editor test...")
    dark_img = Image.new("RGB", (1920, 1080), (30, 30, 40))
    # Simulate sidebar
    from PIL import ImageDraw
    draw = ImageDraw.Draw(dark_img)
    draw.rectangle([0, 0, 250, 1080], fill=(25, 25, 35))
    # Simulate toolbar
    draw.rectangle([0, 0, 1920, 40], fill=(45, 45, 55))
    # Simulate code lines
    for i in range(20):
        y = 60 + i * 25
        colors = [(150, 200, 100), (100, 180, 255), (255, 200, 100), (200, 200, 200)]
        import random
        draw.rectangle([280, y, 280 + random.randint(200, 600), y + 14],
                       fill=random.choice(colors))

    dark_img.save("/tmp/test_code_screen.png")

    result = ScreenSnapshot()
    engine._analyze_image(dark_img, result)
    engine._generate_description(result)
    print(f"    Dark mode:    {'YES' if result.is_dark_mode else 'no'}")
    print(f"    Content:      {result.content_type}")
    print(f"    Has sidebar:  {'YES' if result.has_sidebar else 'no'}")
    print(f"    Description:  {result.description}")
    print(f"    Lin Xia says: {result.emotional_reaction}")

    os.remove("/tmp/test_code_screen.png")

    # Disable
    engine.disable()


if __name__ == "__main__":
    demo()
