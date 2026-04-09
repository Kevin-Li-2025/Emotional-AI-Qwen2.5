"""
Memory Extractor — Extracts storable facts, emotions, and preferences
from each conversation turn using the LLM itself.
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory.memory_store import Memory, MemoryType


EXTRACTION_PROMPT = """Analyze this conversation exchange and extract any important information worth remembering about the user.

EXCHANGE:
User: {user_msg}
Lin Xia: {assistant_msg}

Extract memories in these categories:
1. **Facts**: Concrete information about the user (name, job, hobbies, favorites, etc.)
2. **Emotional Events**: Significant emotional moments (fights, reconciliations, celebrations)
3. **Preferences**: How the user likes to be treated, communication style preferences

Only extract genuinely important information. Skip trivial greetings.
If nothing is worth remembering, return an empty list.

OUTPUT FORMAT (strict JSON):
{{
    "memories": [
        {{
            "content": "<what to remember>",
            "type": "fact" or "emotion" or "preference",
            "importance": <1-10>,
            "emotional_valence": <-1.0 to 1.0 (negative to positive)>
        }}
    ]
}}"""


class MemoryExtractor:
    """Uses the LLM to extract memories from conversation turns."""

    def __init__(self, llm=None):
        """
        Args:
            llm: A llama_cpp.Llama instance for extraction.
                 If None, extraction is disabled.
        """
        self.llm = llm

    def extract(self, user_msg: str, assistant_msg: str) -> list[Memory]:
        """
        Extract memories from a single user-assistant exchange.
        Returns a list of Memory objects ready for storage.
        """
        if self.llm is None:
            return []

        prompt = (
            "<|im_start|>system\nYou are a memory extraction assistant. "
            "Extract important facts and emotional events from conversations.<|im_end|>\n"
            f"<|im_start|>user\n{EXTRACTION_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            output = self.llm(
                prompt,
                max_tokens=500,
                stop=["<|im_end|>"],
                temperature=0.1,
            )
            response = output["choices"][0]["text"].strip()
            parsed = json.loads(response)

            memories = []
            for item in parsed.get("memories", []):
                mem_type = {
                    "fact": MemoryType.FACT,
                    "emotion": MemoryType.EMOTION,
                    "preference": MemoryType.PREFERENCE,
                }.get(item.get("type"), MemoryType.FACT)

                memories.append(Memory(
                    content=item["content"],
                    memory_type=mem_type,
                    importance=float(item.get("importance", 5)),
                    emotional_valence=float(item.get("emotional_valence", 0)),
                    timestamp=time.time(),
                ))

            return memories

        except Exception as e:
            print(f"[EXTRACTOR ERROR] {e}")
            return []
