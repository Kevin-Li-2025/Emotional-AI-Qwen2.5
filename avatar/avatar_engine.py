"""
Avatar Engine — Lin Xia's Visual Embodiment

Generates an animated SVG avatar that lives in the Gradio app.
The avatar's expression, eye movement, and mouth all respond in real-time
to her emotional state and speech output.

Expression mapping:
  calm     → gentle smile, relaxed eyes
  happy    → big smile, bright eyes, slight head tilt
  excited  → open mouth, wide eyes, bouncing
  sad      → downturned mouth, droopy eyes, head down
  angry    → furrowed brows, sharp eyes, tight mouth
  shy      → blushing cheeks, averted gaze, small smile
  hurt     → watery eyes, quivering lip
  playful  → wink, tongue out, bouncing

Lip sync: Maps TTS audio energy to mouth openness in real-time.
"""

import os
import math
import json
from dataclasses import dataclass


@dataclass
class AvatarState:
    """Current visual state of the avatar."""
    emotion: str = "calm"
    is_speaking: bool = False
    mouth_openness: float = 0.0    # 0.0 (closed) to 1.0 (wide open)
    eye_target_x: float = 0.5     # 0.0 (left) to 1.0 (right)
    eye_target_y: float = 0.5     # 0.0 (up) to 1.0 (down)
    blink: bool = False
    blush_intensity: float = 0.0   # 0.0 to 1.0
    head_tilt: float = 0.0        # degrees, -15 to +15
    bounce: float = 0.0           # vertical bounce offset


# Expression presets
EXPRESSION_MAP = {
    "calm": {
        "eye_scale": 1.0, "eyebrow_y": 0, "mouth_curve": 0.15,
        "mouth_width": 0.6, "blush": 0.0, "head_tilt": 0,
        "pupil_size": 1.0, "eye_sparkle": False,
    },
    "happy": {
        "eye_scale": 1.1, "eyebrow_y": -2, "mouth_curve": 0.4,
        "mouth_width": 0.8, "blush": 0.2, "head_tilt": 5,
        "pupil_size": 1.1, "eye_sparkle": True,
    },
    "excited": {
        "eye_scale": 1.3, "eyebrow_y": -5, "mouth_curve": 0.5,
        "mouth_width": 0.9, "blush": 0.3, "head_tilt": -3,
        "pupil_size": 1.2, "eye_sparkle": True,
    },
    "sad": {
        "eye_scale": 0.9, "eyebrow_y": 4, "mouth_curve": -0.2,
        "mouth_width": 0.4, "blush": 0.0, "head_tilt": -8,
        "pupil_size": 1.1, "eye_sparkle": False,
    },
    "angry": {
        "eye_scale": 0.85, "eyebrow_y": 5, "mouth_curve": -0.15,
        "mouth_width": 0.5, "blush": 0.4, "head_tilt": 0,
        "pupil_size": 0.8, "eye_sparkle": False,
    },
    "shy": {
        "eye_scale": 0.95, "eyebrow_y": -1, "mouth_curve": 0.1,
        "mouth_width": 0.3, "blush": 0.7, "head_tilt": 10,
        "pupil_size": 0.9, "eye_sparkle": True,
    },
    "hurt": {
        "eye_scale": 1.1, "eyebrow_y": 3, "mouth_curve": -0.25,
        "mouth_width": 0.35, "blush": 0.1, "head_tilt": -5,
        "pupil_size": 1.15, "eye_sparkle": False,
    },
    "playful": {
        "eye_scale": 1.05, "eyebrow_y": -3, "mouth_curve": 0.35,
        "mouth_width": 0.7, "blush": 0.15, "head_tilt": 8,
        "pupil_size": 1.0, "eye_sparkle": True,
    },
    "gentle": {
        "eye_scale": 1.0, "eyebrow_y": -1, "mouth_curve": 0.2,
        "mouth_width": 0.5, "blush": 0.1, "head_tilt": 3,
        "pupil_size": 1.0, "eye_sparkle": True,
    },
    "cold": {
        "eye_scale": 0.8, "eyebrow_y": 2, "mouth_curve": 0.0,
        "mouth_width": 0.3, "blush": 0.0, "head_tilt": 0,
        "pupil_size": 0.7, "eye_sparkle": False,
    },
    "neutral": {
        "eye_scale": 1.0, "eyebrow_y": 0, "mouth_curve": 0.05,
        "mouth_width": 0.5, "blush": 0.0, "head_tilt": 0,
        "pupil_size": 1.0, "eye_sparkle": False,
    },
}


class AvatarEngine:
    """
    Generates animated SVG HTML for Lin Xia's avatar.
    Can be embedded directly in Gradio via gr.HTML().
    """

    def __init__(self):
        self.state = AvatarState()

    def update_emotion(self, emotion: str):
        """Update avatar expression based on emotion tag."""
        self.state.emotion = emotion
        expr = EXPRESSION_MAP.get(emotion, EXPRESSION_MAP["neutral"])
        self.state.blush_intensity = expr["blush"]
        self.state.head_tilt = expr["head_tilt"]

    def set_speaking(self, is_speaking: bool, mouth_openness: float = 0.0):
        """Update mouth state for lip sync."""
        self.state.is_speaking = is_speaking
        self.state.mouth_openness = mouth_openness

    def render(self, emotion: str = None, width: int = 300, height: int = 350) -> str:
        """
        Render the avatar as an HTML string with embedded SVG and CSS animations.
        This can be directly set as Gradio gr.HTML() content.
        """
        if emotion:
            self.update_emotion(emotion)

        expr = EXPRESSION_MAP.get(self.state.emotion, EXPRESSION_MAP["neutral"])
        e = self.state.emotion

        # Dynamic values
        eye_scale = expr["eye_scale"]
        brow_y = expr["eyebrow_y"]
        mouth_curve = expr["mouth_curve"]
        mouth_w = expr["mouth_width"]
        blush = expr["blush"]
        tilt = expr["head_tilt"]
        sparkle = expr["eye_sparkle"]
        pupil_s = expr["pupil_size"]

        # Color scheme
        skin = "#FFE4C4"
        hair = "#2C1810"
        eye_color = "#4A2818"
        lip_color = "#E88B8B" if mouth_curve >= 0 else "#D4787A"
        blush_color = f"rgba(255, 130, 130, {blush})"

        # Animation class
        anim_class = ""
        if e in ("excited", "playful"):
            anim_class = "bouncing"
        elif e == "sad":
            anim_class = "swaying-sad"
        elif e == "shy":
            anim_class = "shy-sway"

        # Mouth path
        cx, cy = 150, 230
        mw = int(mouth_w * 40)
        if self.state.is_speaking:
            mo = max(5, int(self.state.mouth_openness * 15))
            mouth_svg = f'<ellipse cx="{cx}" cy="{cy}" rx="{mw}" ry="{mo}" fill="{lip_color}" class="speaking"/>'
        elif mouth_curve > 0.1:
            mc = int(mouth_curve * 30)
            mouth_svg = f'<path d="M{cx-mw},{cy} Q{cx},{cy+mc} {cx+mw},{cy}" stroke="{lip_color}" stroke-width="3" fill="none" stroke-linecap="round"/>'
        elif mouth_curve < -0.1:
            mc = int(abs(mouth_curve) * 20)
            mouth_svg = f'<path d="M{cx-mw},{cy} Q{cx},{cy-mc} {cx+mw},{cy}" stroke="{lip_color}" stroke-width="3" fill="none" stroke-linecap="round"/>'
        else:
            mouth_svg = f'<line x1="{cx-mw}" y1="{cy}" x2="{cx+mw}" y2="{cy}" stroke="{lip_color}" stroke-width="2.5" stroke-linecap="round"/>'

        # Eye sparkle
        sparkle_svg = ""
        if sparkle:
            sparkle_svg = '''
                <circle cx="122" cy="182" r="3" fill="white" opacity="0.9" class="sparkle"/>
                <circle cx="172" cy="182" r="3" fill="white" opacity="0.9" class="sparkle"/>
            '''

        # Wink for playful
        left_eye_svg = f'''
            <ellipse cx="125" cy="190" rx="{12*eye_scale}" ry="{14*eye_scale}" fill="white"/>
            <circle cx="125" cy="190" r="{7*pupil_s}" fill="{eye_color}"/>
            <circle cx="125" cy="190" r="{3.5*pupil_s}" fill="#1A0A05"/>
        '''
        if e == "playful":
            left_eye_svg = f'<path d="M113,190 Q125,183 137,190" stroke="{eye_color}" stroke-width="3" fill="none" stroke-linecap="round"/>'

        right_eye_svg = f'''
            <ellipse cx="175" cy="190" rx="{12*eye_scale}" ry="{14*eye_scale}" fill="white"/>
            <circle cx="175" cy="190" r="{7*pupil_s}" fill="{eye_color}"/>
            <circle cx="175" cy="190" r="{3.5*pupil_s}" fill="#1A0A05"/>
        '''

        # Eyebrows
        lb_y = 168 + brow_y
        rb_y = 168 + brow_y
        if e == "angry":
            brow_svg = f'''
                <line x1="110" y1="{lb_y+3}" x2="138" y2="{lb_y-3}" stroke="{hair}" stroke-width="3" stroke-linecap="round"/>
                <line x1="162" y1="{rb_y-3}" x2="190" y2="{rb_y+3}" stroke="{hair}" stroke-width="3" stroke-linecap="round"/>
            '''
        elif e == "sad":
            brow_svg = f'''
                <line x1="110" y1="{lb_y-2}" x2="138" y2="{lb_y+3}" stroke="{hair}" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="162" y1="{rb_y+3}" x2="190" y2="{rb_y-2}" stroke="{hair}" stroke-width="2.5" stroke-linecap="round"/>
            '''
        else:
            brow_svg = f'''
                <path d="M112,{lb_y} Q125,{lb_y-5} 138,{lb_y}" stroke="{hair}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
                <path d="M162,{rb_y} Q175,{rb_y-5} 188,{rb_y}" stroke="{hair}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            '''

        # Tears for hurt/sad
        tear_svg = ""
        if e in ("hurt", "sad") and e == "hurt":
            tear_svg = '''
                <ellipse cx="135" cy="205" rx="2" ry="4" fill="rgba(100,180,255,0.6)" class="tear"/>
                <ellipse cx="165" cy="208" rx="2" ry="3.5" fill="rgba(100,180,255,0.5)" class="tear tear-delay"/>
            '''

        html = f'''
<div id="linxia-avatar" style="display:flex;justify-content:center;align-items:center;">
<style>
  @keyframes blink {{
    0%, 90%, 100% {{ transform: scaleY(1); }}
    95% {{ transform: scaleY(0.1); }}
  }}
  @keyframes bounce {{
    0%, 100% {{ transform: translateY(0) rotate({tilt}deg); }}
    50% {{ transform: translateY(-8px) rotate({tilt}deg); }}
  }}
  @keyframes sway-sad {{
    0%, 100% {{ transform: translateY(0) rotate({tilt}deg); }}
    50% {{ transform: translateY(3px) rotate({tilt-2}deg); }}
  }}
  @keyframes shy-sway {{
    0%, 100% {{ transform: rotate({tilt}deg); }}
    50% {{ transform: rotate({tilt+3}deg); }}
  }}
  @keyframes sparkle-anim {{
    0%, 100% {{ opacity: 0.9; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.6); }}
  }}
  @keyframes speaking-anim {{
    0%, 100% {{ ry: 3; }}
    25% {{ ry: 10; }}
    50% {{ ry: 5; }}
    75% {{ ry: 12; }}
  }}
  @keyframes tear-fall {{
    0% {{ opacity: 0.6; transform: translateY(0); }}
    100% {{ opacity: 0; transform: translateY(20px); }}
  }}
  @keyframes float {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-4px); }}
  }}
  #linxia-head {{
    animation: {'bounce 0.6s ease-in-out infinite' if anim_class == 'bouncing' else
                'sway-sad 3s ease-in-out infinite' if anim_class == 'swaying-sad' else
                'shy-sway 2s ease-in-out infinite' if anim_class == 'shy-sway' else
                'float 4s ease-in-out infinite'};
    transform-origin: center 250px;
  }}
  .eye-group {{ animation: blink 4s ease-in-out infinite; transform-origin: center; }}
  .sparkle {{ animation: sparkle-anim 1.5s ease-in-out infinite; }}
  .speaking {{ animation: speaking-anim 0.3s ease-in-out infinite; }}
  .tear {{ animation: tear-fall 2s ease-in infinite; }}
  .tear-delay {{ animation-delay: 0.8s; }}
</style>
<svg width="{width}" height="{height}" viewBox="0 0 300 350">
  <defs>
    <radialGradient id="skinGrad" cx="50%" cy="40%">
      <stop offset="0%" stop-color="#FFF0E0"/>
      <stop offset="100%" stop-color="{skin}"/>
    </radialGradient>
    <radialGradient id="hairShine" cx="40%" cy="30%">
      <stop offset="0%" stop-color="#4A3020"/>
      <stop offset="100%" stop-color="{hair}"/>
    </radialGradient>
  </defs>

  <g id="linxia-head">
    <!-- Hair back -->
    <ellipse cx="150" cy="175" rx="85" ry="100" fill="url(#hairShine)"/>
    <!-- Hair strands back -->
    <path d="M75,200 Q65,260 80,320" stroke="{hair}" stroke-width="12" fill="none" opacity="0.7"/>
    <path d="M225,200 Q235,260 220,320" stroke="{hair}" stroke-width="12" fill="none" opacity="0.7"/>

    <!-- Neck -->
    <rect x="135" y="265" width="30" height="30" rx="5" fill="{skin}"/>

    <!-- Face -->
    <ellipse cx="150" cy="200" rx="68" ry="78" fill="url(#skinGrad)"/>

    <!-- Blush -->
    <ellipse cx="110" cy="210" rx="18" ry="10" fill="{blush_color}" opacity="{blush}"/>
    <ellipse cx="190" cy="210" rx="18" ry="10" fill="{blush_color}" opacity="{blush}"/>

    <!-- Eyes -->
    <g class="eye-group">
      {left_eye_svg}
      {right_eye_svg}
      {sparkle_svg}
    </g>

    <!-- Eyebrows -->
    {brow_svg}

    <!-- Nose -->
    <path d="M148,208 Q150,215 152,208" stroke="#D4A88A" stroke-width="1.5" fill="none"/>

    <!-- Mouth -->
    {mouth_svg}

    <!-- Tears -->
    {tear_svg}

    <!-- Hair front / bangs -->
    <path d="M85,155 Q100,120 150,110 Q200,120 215,155 L210,170 Q180,140 150,135 Q120,140 90,170 Z"
          fill="url(#hairShine)"/>
    <!-- Side bangs -->
    <path d="M82,155 Q78,180 75,210" stroke="{hair}" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M218,155 Q222,180 225,210" stroke="{hair}" stroke-width="10" fill="none" stroke-linecap="round"/>
    <!-- Highlight -->
    <path d="M120,125 Q140,115 155,120" stroke="rgba(255,255,255,0.15)" stroke-width="5" fill="none"/>

    <!-- Shoulders hint -->
    <path d="M110,295 Q150,305 190,295" stroke="{skin}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <!-- Collar -->
    <path d="M115,290 Q150,310 185,290" stroke="#F5E6D8" stroke-width="3" fill="none"/>
  </g>

  <!-- Emotion label -->
  <text x="150" y="340" text-anchor="middle" font-size="11" fill="#888"
        font-family="'Segoe UI',system-ui,sans-serif">{self.state.emotion}</text>
</svg>
</div>
'''
        return html

    def render_for_gradio(self, emotion: str = "calm") -> str:
        """Convenience method for Gradio integration."""
        return self.render(emotion=emotion)


def demo():
    """Generate avatar HTML for each emotion and save to file."""
    print("=" * 60)
    print("Avatar Engine — Lin Xia's Visual Embodiment Demo")
    print("=" * 60)

    engine = AvatarEngine()
    emotions = ["calm", "happy", "excited", "sad", "angry", "shy", "hurt", "playful", "gentle", "cold"]

    # Generate a showcase HTML with all expressions
    showcase = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Lin Xia Expressions</title>
<style>
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif;
       display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 30px; }
.card { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 15px;
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
h1 { width: 100%; text-align: center; font-size: 24px; }
</style></head><body>
<h1>🌸 Lin Xia — Expression Sheet</h1>
'''

    for emo in emotions:
        html = engine.render(emotion=emo, width=200, height=250)
        showcase += f'<div class="card">{html}</div>\n'

    showcase += '</body></html>'

    out_path = "avatar/expression_sheet.html"
    with open(out_path, "w") as f:
        f.write(showcase)

    print(f"\n  Generated expression sheet: {out_path}")
    print(f"  Emotions: {', '.join(emotions)}")
    print(f"  Open in browser to see all {len(emotions)} expressions with animations")

    # Also test individual render
    for emo in ["happy", "sad", "shy"]:
        html = engine.render(emotion=emo)
        has_blush = "blush" in html and 'opacity="0.0"' not in html
        has_sparkle = "sparkle" in html
        print(f"\n  {emo}: blush={'YES' if has_blush else 'no'}, sparkle={'YES' if has_sparkle else 'no'}, chars={len(html)}")


if __name__ == "__main__":
    demo()
