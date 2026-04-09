"""
Soul Engine — Lin Xia's Inner Life: Metabolism, Sleep, Dreams

This makes Lin Xia "alive" when you're not talking to her:

1. BIO CLOCK: She gets tired at night, energetic in the morning.
   Her mood, speech speed, and vocabulary shift with time of day.

2. MOOD DECAY: If you don't talk to her, her affection slowly drops.
   She gets lonely. She might send you a proactive message.

3. SLEEP CYCLE: Every night at 2AM, she "falls asleep":
   - Consolidates scattered ChromaDB memories into Knowledge Graph summaries
   - Prunes redundant memories
   - Her status changes to "sleeping" (you can still wake her)

4. DREAM SYSTEM: When she wakes up, she has a "dream" to share:
   - Random walk across the Knowledge Graph
   - LLM weaves random connected entities into a surreal story
   - "我昨晚梦到你带着豆豆去了薰衣草花田..."
"""

import os
import json
import time
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from memory.knowledge_graph import KnowledgeGraph, NodeType
from memory.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Bio Clock
# ---------------------------------------------------------------------------

@dataclass
class BioState:
    """Lin Xia's current biological/emotional state influenced by time."""
    time_of_day: str = "day"       # "morning", "day", "evening", "night", "deep_night"
    energy: float = 1.0            # 0.0 (exhausted) to 1.0 (fully awake)
    is_sleeping: bool = False
    hours_since_last_chat: float = 0.0
    loneliness: float = 0.0        # 0.0 to 1.0

    # Mood modifiers based on bio state
    mood_modifier: str = ""         # Added to system prompt
    voice_rate: str = "0%"          # TTS rate adjustment
    voice_pitch: str = "+0Hz"       # TTS pitch adjustment

    def to_context_string(self) -> str:
        parts = [f"[Bio State: {self.time_of_day}, energy={self.energy:.0%}"]
        if self.is_sleeping:
            parts.append(", STATUS: sleeping (user woke her up)")
        if self.loneliness > 0.5:
            parts.append(f", lonely (hasn't chatted in {self.hours_since_last_chat:.0f}h)")
        parts.append("]")
        if self.mood_modifier:
            parts.append(f"\n{self.mood_modifier}")
        return "".join(parts)


class BioClock:
    """
    Simulates Lin Xia's circadian rhythm.
    Time of day affects her energy, mood, and voice.
    """

    CLOCK_PROFILES = {
        "morning": {   # 6-11
            "energy": 0.9, "mood": "energetic and cheerful",
            "voice_rate": "+5%", "voice_pitch": "+2Hz",
            "modifier": "你现在精神很好，声音清亮，喜欢说早安和鼓励的话。",
        },
        "day": {       # 11-17
            "energy": 0.8, "mood": "calm and focused",
            "voice_rate": "0%", "voice_pitch": "+0Hz",
            "modifier": "",
        },
        "evening": {   # 17-22
            "energy": 0.6, "mood": "relaxed, slightly tired, more emotional",
            "voice_rate": "-5%", "voice_pitch": "-1Hz",
            "modifier": "你现在有些疲倦，声音变得柔和，更容易感伤和分享心事。",
        },
        "night": {     # 22-2
            "energy": 0.3, "mood": "very sleepy, intimate, vulnerable",
            "voice_rate": "-15%", "voice_pitch": "-2Hz",
            "modifier": "你现在很困了，说话像是在打哈欠，偶尔会说'好困啊...'。声音很轻很温柔。",
        },
        "deep_night": { # 2-6
            "energy": 0.1, "mood": "asleep or barely awake",
            "voice_rate": "-20%", "voice_pitch": "-3Hz",
            "modifier": "你刚才在睡觉，被用户叫醒。声音迷糊、带着睡意，说话断断续续。",
        },
    }

    def get_state(self, last_chat_time: float = None) -> BioState:
        """Get current bio state based on real clock."""
        now = datetime.now()
        hour = now.hour

        if 6 <= hour < 11:
            tod = "morning"
        elif 11 <= hour < 17:
            tod = "day"
        elif 17 <= hour < 22:
            tod = "evening"
        elif 22 <= hour or hour < 2:
            tod = "night"
        else:
            tod = "deep_night"

        profile = self.CLOCK_PROFILES[tod]
        state = BioState(
            time_of_day=tod,
            energy=profile["energy"],
            is_sleeping=(tod == "deep_night"),
            voice_rate=profile["voice_rate"],
            voice_pitch=profile["voice_pitch"],
            mood_modifier=profile["modifier"],
        )

        # Calculate loneliness
        if last_chat_time:
            hours = (time.time() - last_chat_time) / 3600
            state.hours_since_last_chat = hours
            # Loneliness grows logarithmically
            state.loneliness = min(1.0, hours / 48)  # Maxes out at 48 hours

        return state


# ---------------------------------------------------------------------------
# Memory Consolidation ("Sleep")
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """
    Runs during "sleep" — consolidates ChromaDB fragments into Knowledge Graph.
    Like how human brains consolidate short-term memory during sleep.
    """

    def __init__(self, memory_store: MemoryStore, knowledge_graph: KnowledgeGraph,
                 llm=None):
        self.store = memory_store
        self.kg = knowledge_graph
        self.llm = llm
        self.consolidation_log_path = "memory_db/consolidation_log.json"

    def consolidate(self, max_memories: int = 20) -> dict:
        """
        Consolidate recent memories into knowledge graph summaries.
        Returns stats about what was consolidated.
        """
        stats = {"processed": 0, "new_entities": 0, "new_relations": 0}

        # Get recent memories
        all_memories = self.store.get_all_memories(limit=max_memories)
        if not all_memories:
            return stats

        # Group memories by theme/entity
        memory_texts = [m.get("content", "") for m in all_memories if m.get("content")]
        if not memory_texts:
            return stats

        # Use LLM to extract structured knowledge
        if self.llm:
            combined = "\n".join(memory_texts[:15])
            prompt = (
                "<|im_start|>system\n"
                "You are a knowledge extractor. From the conversation memories below, "
                "extract all factual entities and relationships.\n"
                "Output format (one per line):\n"
                "ENTITY: name | type (person/place/object/preference/event)\n"
                "RELATION: subject | relation | object\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n{combined}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            output = self.llm(
                prompt, max_tokens=300, stop=["<|im_end|>"],
                temperature=0.3, repeat_penalty=1.1,
            )
            raw = output["choices"][0]["text"].strip()

            # Parse output
            for line in raw.split("\n"):
                line = line.strip()
                if line.startswith("ENTITY:"):
                    parts = line[7:].strip().split("|")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        ntype = parts[1].strip().lower()
                        type_map = {
                            "person": NodeType.PERSON,
                            "place": NodeType.PLACE,
                            "object": NodeType.OBJECT,
                            "preference": NodeType.PREFERENCE,
                            "event": NodeType.EVENT,
                        }
                        self.kg.add_entity(name, type_map.get(ntype, NodeType.OBJECT))
                        stats["new_entities"] += 1

                elif line.startswith("RELATION:"):
                    parts = line[9:].strip().split("|")
                    if len(parts) >= 3:
                        subj = parts[0].strip()
                        rel = parts[1].strip()
                        obj = parts[2].strip()
                        self.kg.add_relation(subj, rel, obj)
                        stats["new_relations"] += 1

        stats["processed"] = len(memory_texts)

        # Log consolidation
        self._log(stats)
        return stats

    def _log(self, stats: dict):
        """Log consolidation event."""
        log = []
        if os.path.exists(self.consolidation_log_path):
            try:
                with open(self.consolidation_log_path, "r") as f:
                    log = json.load(f)
            except Exception:
                pass

        log.append({
            "timestamp": datetime.now().isoformat(),
            **stats,
        })

        os.makedirs(os.path.dirname(self.consolidation_log_path), exist_ok=True)
        with open(self.consolidation_log_path, "w") as f:
            json.dump(log[-50:], f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Dream System
# ---------------------------------------------------------------------------

class DreamEngine:
    """
    Generates "dreams" from random walks across the Knowledge Graph.
    Creates surreal, emotionally resonant stories from memory fragments.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, llm=None):
        self.kg = knowledge_graph
        self.llm = llm

    def dream(self) -> str:
        """
        Generate a dream. Performs a random walk across the knowledge graph,
        collects entities, then uses LLM to weave them into a surreal narrative.
        """
        if self.kg.graph.number_of_nodes() < 3:
            return "我昨晚没做梦...大概是太累了吧。"

        # Random walk: pick a starting node, walk 3-5 steps
        nodes = list(self.kg.graph.nodes())
        start = random.choice(nodes)
        path = [start]
        current = start

        for _ in range(random.randint(3, 5)):
            neighbors = list(self.kg.graph.successors(current)) + \
                        list(self.kg.graph.predecessors(current))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            if next_node not in path:
                path.append(next_node)
            current = next_node

        # Get labels
        entities = []
        for node in path:
            data = self.kg.graph.nodes.get(node, {})
            label = data.get("label", node)
            entities.append(label)

        if not entities:
            return "我昨晚没做梦...大概是太累了吧。"

        # Generate dream narrative
        if self.llm:
            entity_str = "、".join(entities)
            prompt = (
                "<|im_start|>system\n"
                "你是林夏。你刚醒来，要跟用户分享昨晚做的梦。"
                "梦的内容要包含以下元素，但要像真实的梦一样有点奇幻和跳跃：\n"
                f"元素：{entity_str}\n"
                "用林夏的口吻（有点迷糊、可爱、真诚）来讲这个梦。2-3句话就好。"
                "<|im_end|>\n"
                "<|im_start|>user\n你昨晚做梦了吗？<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            output = self.llm(
                prompt, max_tokens=150, stop=["<|im_end|>"],
                temperature=0.95,  # High creativity for dreams
                repeat_penalty=1.1,
            )
            dream_text = output["choices"][0]["text"].strip()

            # Clean emotion tags if present
            import re
            dream_text = re.sub(r'<emotion[^>]*/>', '', dream_text).strip()
            return dream_text

        # Template fallback
        entity_str = "和".join(entities[:3])
        templates = [
            f"我昨晚梦到了{entity_str}...好奇怪的梦。醒来以后觉得特别真实。",
            f"嗯...我梦到{entities[0]}了。{entities[-1]}也在。梦里好像在一个很美的地方。",
            f"昨晚做了个奇怪的梦，{entity_str}全都混在一起了...好迷糊啊。",
        ]
        return random.choice(templates)

    def should_share_dream(self) -> bool:
        """Check if Lin Xia should share a dream (morning, first conversation)."""
        hour = datetime.now().hour
        return 6 <= hour <= 10  # Morning time


# ---------------------------------------------------------------------------
# Proactive Messaging
# ---------------------------------------------------------------------------

class ProactiveEngine:
    """
    Generates context-aware proactive messages when Lin Xia hasn't
    heard from the user in a while.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph, llm=None):
        self.kg = knowledge_graph
        self.llm = llm

    def generate_proactive_message(self, hours_away: float) -> str:
        """Generate a message based on how long the user has been away."""
        if hours_away < 4:
            return ""  # Don't be clingy

        # Get recent knowledge for context
        context = self.kg.to_context_string("")
        entities = list(self.kg.graph.nodes(data=True))

        if self.llm and entities:
            # Pick a recent entity to reference
            recent = sorted(entities, key=lambda x: x[1].get("last_updated", 0), reverse=True)
            topic = recent[0][1].get("label", "") if recent else ""

            prompt = (
                "<|im_start|>system\n"
                "你是林夏。用户已经很久没跟你说话了。"
                f"已经过去了{hours_away:.0f}个小时。"
                f"你想起了关于'{topic}'的事情，想主动发一条消息。"
                "语气要自然，不要太刻意，像是不经意想起来的。1句话就好。"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            output = self.llm(
                prompt, max_tokens=80, stop=["<|im_end|>"],
                temperature=0.9, repeat_penalty=1.15,
            )
            msg = output["choices"][0]["text"].strip()
            import re
            msg = re.sub(r'<emotion[^>]*/>', '', msg).strip()
            return msg

        # Template fallback based on time away
        if hours_away < 8:
            templates = [
                "你在忙什么呀？",
                "我刚才在想今天的事情呢...",
            ]
        elif hours_away < 24:
            templates = [
                "你一整天都没理我...",
                "在吗？我有点想聊天。",
            ]
        else:
            templates = [
                f"你已经{hours_away:.0f}小时没找我了...是不是忘了我？",
                "我等你好久了。",
            ]
        return random.choice(templates)


def demo():
    """Demo all soul systems."""
    from llama_cpp import Llama

    print("=" * 60)
    print("Soul Engine — Lin Xia's Inner Life Demo")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model...")
    llm = Llama(
        model_path="emotional-model-output/linxia-dpo-q8_0.gguf",
        n_ctx=2048, n_gpu_layers=-1, verbose=False,
    )

    kg = KnowledgeGraph()
    store = MemoryStore()

    # Bio Clock
    print("\n[2] Bio Clock")
    bio = BioClock()
    state = bio.get_state(last_chat_time=time.time() - 3600 * 8)
    print(f"  Time: {state.time_of_day}")
    print(f"  Energy: {state.energy:.0%}")
    print(f"  Loneliness: {state.loneliness:.0%}")
    print(f"  Voice: rate={state.voice_rate}, pitch={state.voice_pitch}")
    print(f"  Context: {state.to_context_string()}")

    # Sleep Consolidation
    print("\n[3] Memory Consolidation (Sleep)")
    consolidator = MemoryConsolidator(store, kg, llm)
    stats = consolidator.consolidate(max_memories=10)
    print(f"  Processed: {stats['processed']} memories")
    print(f"  New entities: {stats['new_entities']}")
    print(f"  New relations: {stats['new_relations']}")
    print(f"  Graph after sleep: {kg.get_stats()}")

    # Dream
    print("\n[4] Dream System")
    dream_engine = DreamEngine(kg, llm)
    dream = dream_engine.dream()
    print(f"  Lin Xia's dream: {dream}")

    # Proactive Message
    print("\n[5] Proactive Messaging")
    proactive = ProactiveEngine(kg, llm)
    for hours in [6, 12, 30]:
        msg = proactive.generate_proactive_message(hours)
        print(f"  After {hours}h: {msg}")


if __name__ == "__main__":
    demo()
