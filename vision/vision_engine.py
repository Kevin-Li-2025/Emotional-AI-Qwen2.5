"""
Vision Engine — Multimodal Image Understanding for Lin Xia
Gives Lin Xia "eyes" — she can see images and react emotionally.

Strategy:
  1. Primary: Use llama-cpp-python with Qwen2.5-VL GGUF (local, private)
  2. Fallback: Use transformers + BLIP-2 for caption generation (local, lighter)
  3. Last resort: Use PIL to extract basic image metadata (always works)

The key innovation is NOT just describing the image, but generating
an EMOTIONAL reaction that fits Lin Xia's personality.
"""

import os
import base64
from pathlib import Path
from PIL import Image

# Try to import vision models
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAS_BLIP = True
except (ImportError, AttributeError, Exception):
    HAS_BLIP = False

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    HAS_LLAVA = True
except ImportError:
    HAS_LLAVA = False


class VisionEngine:
    """
    Multi-strategy image understanding engine.
    Extracts visual information and generates emotional reactions.
    """

    def __init__(self, strategy: str = "metadata"):
        """
        Args:
            strategy: "vlm" (Qwen2-VL GGUF), "blip" (BLIP-2), or "metadata" (PIL only)
        """
        self.strategy = strategy
        self.vlm = None
        self.blip_model = None
        self.blip_processor = None

        if strategy == "vlm" and HAS_LLAVA:
            self._init_vlm()
        elif strategy == "blip" and HAS_BLIP:
            self._init_blip()
        else:
            self.strategy = "metadata"
            print("[VISION] Using metadata-only mode (PIL)")

    def _init_vlm(self):
        """Initialize vision-language model (Qwen2-VL GGUF)."""
        # Look for VLM model in standard locations
        vlm_paths = [
            "emotional-model-output/qwen2-vl-2b-q4.gguf",
            "models/qwen2-vl-2b-q4.gguf",
        ]
        mmproj_paths = [
            "emotional-model-output/qwen2-vl-2b-mmproj.gguf",
            "models/qwen2-vl-2b-mmproj.gguf",
        ]

        model_path = next((p for p in vlm_paths if os.path.exists(p)), None)
        mmproj_path = next((p for p in mmproj_paths if os.path.exists(p)), None)

        if model_path and mmproj_path:
            chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            self.vlm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False,
            )
            print(f"[VISION] VLM loaded: {model_path}")
        else:
            print("[VISION] VLM model not found, falling back to metadata mode")
            self.strategy = "metadata"

    def _init_blip(self):
        """Initialize BLIP-2 for image captioning."""
        try:
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            print("[VISION] BLIP-2 loaded")
        except Exception as e:
            print(f"[VISION] BLIP-2 failed: {e}, falling back to metadata")
            self.strategy = "metadata"

    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze an image and return structured information.

        Returns:
            {
                "description": str,    # What's in the image
                "colors": list[str],   # Dominant colors
                "mood_hint": str,      # Suggested mood from visual cues
                "dimensions": tuple,   # (width, height)
                "source": str,         # Which strategy was used
            }
        """
        if not os.path.exists(image_path):
            return {"description": "无法打开图片", "source": "error"}

        result = {
            "source": self.strategy,
            "file_name": os.path.basename(image_path),
        }

        # Always extract metadata
        try:
            img = Image.open(image_path)
            result["dimensions"] = img.size
            result["format"] = img.format or "unknown"

            # Extract dominant colors
            small = img.resize((50, 50)).convert("RGB")
            pixels = list(small.getdata())
            from collections import Counter
            color_counts = Counter(pixels)
            top_colors = color_counts.most_common(5)

            color_names = []
            for rgb, _ in top_colors:
                name = self._rgb_to_name(rgb)
                if name not in color_names:
                    color_names.append(name)
            result["colors"] = color_names[:3]

            # Mood hint from colors
            result["mood_hint"] = self._colors_to_mood(color_names)

        except Exception as e:
            result["dimensions"] = (0, 0)
            result["colors"] = []
            result["mood_hint"] = "neutral"

        # VLM description
        if self.strategy == "vlm" and self.vlm:
            try:
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                response = self.vlm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": "用中文简短描述这张图片中的内容和氛围。"},
                        ]
                    }],
                    max_tokens=100,
                )
                result["description"] = response["choices"][0]["message"]["content"]
            except Exception as e:
                result["description"] = f"图片分析失败: {e}"

        # BLIP description
        elif self.strategy == "blip" and self.blip_model:
            try:
                img = Image.open(image_path).convert("RGB")
                inputs = self.blip_processor(img, return_tensors="pt")
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                result["description"] = caption
            except Exception as e:
                result["description"] = f"图片分析失败: {e}"

        # Metadata-only description
        else:
            w, h = result.get("dimensions", (0, 0))
            colors = ", ".join(result.get("colors", []))
            orientation = "横" if w > h else "竖" if h > w else "方"
            result["description"] = (
                f"一张{orientation}版的{result.get('format', '图片')}照片 "
                f"({w}×{h})，主色调为{colors}。"
                f"整体氛围{result['mood_hint']}。"
            )

        return result

    def generate_emotional_reaction(self, image_info: dict, llm=None) -> str:
        """
        Given image analysis results, generate Lin Xia's emotional reaction.

        If an LLM is provided, use it; otherwise, use template-based reactions.
        """
        desc = image_info.get("description", "一张照片")
        mood = image_info.get("mood_hint", "neutral")
        colors = image_info.get("colors", [])

        if llm:
            # Use the emotional AI model for a personalized reaction
            prompt = (
                f"<|im_start|>system\n"
                f"你是林夏，看到用户发来的一张图片。图片内容：{desc}。"
                f"主色调：{', '.join(colors)}。氛围：{mood}。"
                f"请用林夏的性格（真实、有情绪、有个性的女生）简短回复对这张图片的感受。"
                f"<|im_end|>\n"
                f"<|im_start|>user\n你看看这张照片。<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            output = llm(prompt, max_tokens=120, stop=["<|im_end|>"],
                        temperature=0.8, repeat_penalty=1.15)
            return output["choices"][0]["text"].strip()

        # Template-based fallback
        reactions = {
            "warm": [
                f"这张照片好温暖...{colors[0] if colors else '柔和'}的色调让人觉得很舒服。",
                f"看到这张照片心情突然变好了呢。",
            ],
            "cool": [
                f"好清冷的感觉...{colors[0] if colors else '蓝'}色调给人一种安静的感觉。",
                f"这种色调让我想起下雨天窝在家里的感觉。",
            ],
            "vibrant": [
                f"哇！颜色好鲜艳！看着就很有活力。",
                f"这照片拍得好好看，色彩很丰富呢。",
            ],
            "dark": [
                f"这张有点暗暗的...是晚上拍的吗？",
                f"氛围有点神秘呢，你在哪里拍的？",
            ],
            "neutral": [
                f"我看到了。你想跟我分享这个吗？",
                f"嗯，这张照片...你拍的吗？",
            ],
        }

        import random
        options = reactions.get(mood, reactions["neutral"])
        return random.choice(options)

    @staticmethod
    def _rgb_to_name(rgb: tuple) -> str:
        """Map RGB tuple to approximate color name in Chinese."""
        r, g, b = rgb
        if r > 200 and g < 100 and b < 100: return "红色"
        if r > 200 and g > 150 and b < 100: return "橙色"
        if r > 200 and g > 200 and b < 100: return "黄色"
        if r < 100 and g > 150 and b < 100: return "绿色"
        if r < 100 and g < 100 and b > 200: return "蓝色"
        if r > 150 and g < 100 and b > 150: return "紫色"
        if r > 200 and g > 100 and b > 150: return "粉色"
        if r > 200 and g > 200 and b > 200: return "白色"
        if r < 50 and g < 50 and b < 50: return "黑色"
        if abs(r - g) < 30 and abs(g - b) < 30:
            if r > 180: return "浅灰"
            if r > 100: return "灰色"
            return "深灰"
        return "混合色"

    @staticmethod
    def _colors_to_mood(colors: list) -> str:
        """Infer mood from dominant colors."""
        warm = {"红色", "橙色", "黄色", "粉色"}
        cool = {"蓝色", "紫色", "绿色"}
        dark = {"黑色", "深灰"}

        warm_count = sum(1 for c in colors if c in warm)
        cool_count = sum(1 for c in colors if c in cool)
        dark_count = sum(1 for c in colors if c in dark)

        if warm_count >= 2: return "warm"
        if cool_count >= 2: return "cool"
        if dark_count >= 2: return "dark"
        if len(set(colors)) >= 3: return "vibrant"
        return "neutral"


if __name__ == "__main__":
    # Generate a test image and analyze it
    print("=" * 60)
    print("Vision Engine Demo (Metadata Mode)")
    print("=" * 60)

    engine = VisionEngine(strategy="metadata")

    # Create a simple test image
    test_path = "/tmp/test_sunset.png"
    img = Image.new("RGB", (800, 600))
    pixels = img.load()
    for y in range(600):
        for x in range(800):
            r = int(255 * (1 - y / 600))
            g = int(100 * (1 - y / 600))
            b = int(50 + 150 * (y / 600))
            pixels[x, y] = (r, g, b)
    img.save(test_path)

    # Analyze
    info = engine.analyze_image(test_path)
    print(f"\nImage analysis:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    reaction = engine.generate_emotional_reaction(info)
    print(f"\nLin Xia's reaction: {reaction}")

    os.remove(test_path)
