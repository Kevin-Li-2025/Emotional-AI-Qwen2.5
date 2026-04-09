"""
Desktop App — Lin Xia Lives on Your Desktop

A transparent, frameless, always-on-top window that floats on your Mac desktop.
Lin Xia's avatar lives here — she breathes, reacts, and speaks through a
tiny companion window you can drag anywhere on screen.

Features:
  - Transparent background (only the avatar is visible)
  - Always on top of other windows
  - Draggable by clicking on the avatar
  - Real-time emotion updates via Python ↔ JS bridge
  - Speech bubble for messages
  - Minimizes to a tiny dot when inactive

Usage:
  python3 -m avatar.desktop_app
"""

import os
import sys
import json
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatar.avatar_engine import AvatarEngine, EXPRESSION_MAP


def generate_standalone_html(emotion: str = "calm", message: str = "") -> str:
    """
    Generate a full HTML page for the desktop floating window.
    Includes transparent background, drag support, speech bubble,
    and JS API for real-time updates from Python.
    """
    engine = AvatarEngine()
    avatar_svg = engine.render(emotion=emotion, width=200, height=240)

    # Escape message for JS
    safe_message = message.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    -webkit-app-region: drag;
    cursor: grab;
    user-select: none;
  }}
  body:active {{ cursor: grabbing; }}

  #container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    height: 100%;
    padding-bottom: 10px;
  }}

  /* Speech bubble */
  #bubble {{
    background: rgba(30, 30, 50, 0.85);
    color: #f0e6ff;
    font-family: 'Hiragino Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 220px;
    text-align: center;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    margin-bottom: 8px;
    opacity: 0;
    transform: translateY(5px);
    transition: opacity 0.4s ease, transform 0.4s ease;
    -webkit-app-region: no-drag;
    pointer-events: auto;
  }}
  #bubble.visible {{
    opacity: 1;
    transform: translateY(0);
  }}
  /* Bubble tail */
  #bubble::after {{
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-top: 8px solid rgba(30, 30, 50, 0.85);
  }}

  #avatar-wrapper {{
    position: relative;
  }}

  /* Emotion indicator dot */
  #mood-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    position: absolute;
    bottom: 5px;
    right: 5px;
    border: 2px solid rgba(255,255,255,0.3);
    transition: background 0.5s ease;
  }}

  /* Health indicator */
  #health-bar {{
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: rgba(255,255,255,0.5);
    font-family: monospace;
    opacity: 0;
    transition: opacity 0.3s;
  }}
  #health-bar.visible {{ opacity: 1; }}

  /* Subtle glow behind avatar */
  #glow {{
    position: absolute;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(180,130,255,0.15) 0%, transparent 70%);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    animation: pulse-glow 3s ease-in-out infinite;
  }}
  @keyframes pulse-glow {{
    0%, 100% {{ opacity: 0.5; transform: translate(-50%, -50%) scale(1); }}
    50% {{ opacity: 1; transform: translate(-50%, -50%) scale(1.15); }}
  }}
</style>
</head>
<body>
<div id="container">
  <div id="bubble" class="{('visible' if message else '')}">
    {safe_message if message else ''}
  </div>
  <div id="avatar-wrapper">
    <div id="glow"></div>
    {avatar_svg}
    <div id="mood-dot" style="background: #8BC34A;"></div>
    <div id="health-bar"></div>
  </div>
</div>

<script>
  // Mood colors
  const MOOD_COLORS = {{
    calm: '#8BC34A', happy: '#FFD700', excited: '#FF9800',
    sad: '#5C6BC0', angry: '#F44336', shy: '#F48FB1',
    hurt: '#7986CB', playful: '#AB47BC', gentle: '#81C784',
    cold: '#607D8B', neutral: '#9E9E9E'
  }};

  // Update emotion from Python bridge
  function updateEmotion(emotion) {{
    const dot = document.getElementById('mood-dot');
    dot.style.background = MOOD_COLORS[emotion] || MOOD_COLORS.neutral;
  }}

  // Show speech bubble
  function showMessage(text) {{
    const bubble = document.getElementById('bubble');
    bubble.textContent = text;
    bubble.classList.add('visible');
    // Auto-hide after text length * 100ms (min 3s, max 8s)
    const duration = Math.min(Math.max(text.length * 100, 3000), 8000);
    setTimeout(() => bubble.classList.remove('visible'), duration);
  }}

  // Update health display
  function updateHealth(bpm, stress) {{
    const bar = document.getElementById('health-bar');
    if (bpm > 0) {{
      bar.textContent = `♥ ${{bpm}} BPM`;
      bar.classList.add('visible');
      // Color based on stress
      bar.style.color = stress > 0.5
        ? 'rgba(255,100,100,0.7)'
        : 'rgba(150,255,150,0.6)';
    }} else {{
      bar.classList.remove('visible');
    }}
  }}

  // Initial message
  {'showMessage("' + safe_message + '");' if message else ''}

  // Expose to pywebview
  window.linxia = {{
    updateEmotion: updateEmotion,
    showMessage: showMessage,
    updateHealth: updateHealth,
  }};
</script>
</body>
</html>'''
    return html


class DesktopLinXia:
    """
    The desktop companion app.
    Creates a floating transparent window with Lin Xia's avatar.
    """

    def __init__(self, width: int = 250, height: int = 360):
        self.width = width
        self.height = height
        self.window = None
        self._running = False

    def launch(self, initial_emotion: str = "calm", initial_message: str = ""):
        """Launch the desktop window."""
        try:
            import webview
        except ImportError:
            print("[DESKTOP] pywebview not installed. Run: pip3 install pywebview")
            return

        html = generate_standalone_html(
            emotion=initial_emotion,
            message=initial_message or "嗨~我在这里哦。",
        )

        self.window = webview.create_window(
            title="Lin Xia",
            html=html,
            width=self.width,
            height=self.height,
            resizable=False,
            frameless=True,
            transparent=True,
            on_top=True,
            x=100,
            y=100,
        )

        self._running = True
        print("[DESKTOP] Lin Xia is now on your desktop!")
        print("[DESKTOP] Drag her anywhere. She'll float on top of everything.")

        webview.start(debug=False)
        self._running = False

    def update_emotion(self, emotion: str):
        """Update the avatar's emotion in real-time."""
        if self.window:
            self.window.evaluate_js(f"window.linxia.updateEmotion('{emotion}')")

    def say(self, message: str):
        """Show a speech bubble with a message."""
        if self.window:
            safe = message.replace("'", "\\'").replace('"', '\\"')
            self.window.evaluate_js(f"window.linxia.showMessage('{safe}')")

    def update_health(self, bpm: float, stress: float):
        """Update the health indicator."""
        if self.window:
            self.window.evaluate_js(
                f"window.linxia.updateHealth({bpm:.0f}, {stress:.2f})"
            )

    def reload_avatar(self, emotion: str, message: str = ""):
        """Fully reload the avatar with a new emotion."""
        if self.window:
            html = generate_standalone_html(emotion=emotion, message=message)
            self.window.load_html(html)


def demo():
    """Demo the desktop companion."""
    print("=" * 60)
    print("Desktop App — Lin Xia Lives on Your Desktop")
    print("=" * 60)

    desktop = DesktopLinXia()

    # Schedule emotion changes in background
    def emotion_cycle():
        time.sleep(3)
        emotions = [
            ("happy", "你来找我了！开心~"),
            ("shy", "...你一直看着我干嘛"),
            ("playful", "嘿嘿，你今天心情好吗？"),
            ("gentle", "我会一直在这里陪着你的。"),
            ("sad", "你是不是太忙了...都不理我。"),
            ("excited", "我们聊点什么吧！"),
            ("calm", "安静地待在一起也很好。"),
        ]
        for emo, msg in emotions:
            if not desktop._running:
                break
            time.sleep(5)
            try:
                desktop.reload_avatar(emo, msg)
            except Exception:
                pass

    t = threading.Thread(target=emotion_cycle, daemon=True)
    t.start()

    desktop.launch(
        initial_emotion="happy",
        initial_message="嗨~我在这里哦！把我拖到你喜欢的位置吧。",
    )


if __name__ == "__main__":
    demo()
