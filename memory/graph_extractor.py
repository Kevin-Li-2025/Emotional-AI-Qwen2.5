"""
Graph Extractor — Extract entities and relationships from conversations.
Uses rule-based + pattern matching to identify structured knowledge
from dialogue turns, then stores them in the KnowledgeGraph.

Runs after each conversation turn alongside existing memory extraction.
"""

import re
import time
from memory.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


# Pattern-based entity extraction rules
EXTRACT_PATTERNS = [
    # Preferences: "我喜欢X" / "我最爱X"
    {
        "pattern": r"我(?:最)?(?:喜欢|爱|爱吃|爱喝|爱看|想要|想吃|想买)(.{1,15}?)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.LIKES,
        "entity_type": NodeType.PREFERENCE,
    },
    # Dislikes: "我不喜欢X" / "我讨厌X"
    {
        "pattern": r"我(?:不喜欢|讨厌|不爱|不想|受不了)(.{1,15}?)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.DISLIKES,
        "entity_type": NodeType.PREFERENCE,
    },
    # Has: "我有X" / "我养了X"
    {
        "pattern": r"我(?:有|养了?|买了?|带了?)(?:一[只个条件])?(.{1,10}?)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.HAS,
        "entity_type": NodeType.OBJECT,
    },
    # Lives/Works: "我住在X" / "我在X工作"
    {
        "pattern": r"我(?:住在|在)(.{1,10}?)(?:工作|上班|上学|读书|住)",
        "relation": EdgeType.LIVES_IN,
        "entity_type": NodeType.PLACE,
    },
    # Names: "我叫X" / "我是X"
    {
        "pattern": r"我(?:叫|是|名字是)(.{1,8}?)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.IS_A,
        "entity_type": NodeType.PERSON,
    },
    # Pets with names: "X叫Y"
    {
        "pattern": r"(?:它|他|她|宠物|狗|猫)(?:叫|名字是)(.{1,8}?)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.IS_A,
        "entity_type": NodeType.OBJECT,
    },
    # Events: "今天X了" / "我刚X"
    {
        "pattern": r"(?:今天|昨天|刚才|最近|上周)(?:我)?(.{2,15}?)(?:了|过)(?:[，。！？,!?\s]|$)",
        "relation": EdgeType.RELATED_TO,
        "entity_type": NodeType.EVENT,
    },
    # Family: "我妈/爸/奶奶/爷爷"
    {
        "pattern": r"我(?:的)?(妈妈?|爸爸?|奶奶|爷爷|姐姐?|哥哥?|弟弟?|妹妹?|女朋友|男朋友|老婆|老公)",
        "relation": EdgeType.HAS,
        "entity_type": NodeType.PERSON,
    },
]

# Emotional event patterns (for Lin Xia's perspective)
EMOTION_PATTERNS = [
    (r"开心|高兴|太好了|哇|恭喜", "happy"),
    (r"难过|伤心|委屈|哭", "sad"),
    (r"生气|过分|闭嘴|烦|不礼貌", "angry"),
    (r"害怕|担心|怎么办|紧张", "anxious"),
    (r"想你|想念|思念", "longing"),
]


class GraphExtractor:
    """
    Extract entities and relationships from conversations
    and store them in the KnowledgeGraph.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def extract_from_turn(self, user_msg: str, assistant_msg: str) -> list[dict]:
        """
        Extract knowledge from a single conversation turn.

        Returns list of extracted triples:
            [{"subject": str, "relation": str, "object": str}]
        """
        extracted = []

        # 1. Extract from user message (facts about the user)
        for rule in EXTRACT_PATTERNS:
            matches = re.findall(rule["pattern"], user_msg)
            for match in matches:
                match = match.strip()
                if len(match) < 1 or match in ("我", "你", "的", "了"):
                    continue

                self.kg.add_entity("用户", NodeType.PERSON)
                self.kg.add_entity(match, rule["entity_type"])
                self.kg.add_relation("用户", rule["relation"], match)
                extracted.append({
                    "subject": "用户",
                    "relation": rule["relation"],
                    "object": match,
                    "source": "user_msg",
                })

        # 2. Extract emotional events from assistant message
        for pattern, emotion_label in EMOTION_PATTERNS:
            if re.search(pattern, user_msg):
                event_desc = user_msg[:20]
                self.kg.add_entity("林夏", NodeType.PERSON)
                self.kg.add_entity(emotion_label, NodeType.EMOTION)
                self.kg.add_relation("林夏", EdgeType.FELT, emotion_label, {
                    "trigger": event_desc,
                    "timestamp": time.time(),
                })
                extracted.append({
                    "subject": "林夏",
                    "relation": EdgeType.FELT,
                    "object": emotion_label,
                    "source": "emotion_detection",
                })

        return extracted

    def extract_from_image(self, image_description: str) -> list[dict]:
        """
        Extract entities from a vision-generated image description.
        """
        extracted = []

        # Simple entity extraction from image descriptions
        items_pattern = r"(?:图片|照片|画面)(?:中|里)?(?:有|是|showing)(.+?)(?:[。，]|$)"
        matches = re.findall(items_pattern, image_description)

        for match in matches:
            self.kg.add_entity("用户", NodeType.PERSON)
            self.kg.add_entity(match.strip(), NodeType.OBJECT, {"source": "image"})
            self.kg.add_relation("用户", "shared_image_of", match.strip())
            extracted.append({
                "subject": "用户",
                "relation": "shared_image_of",
                "object": match.strip(),
                "source": "vision",
            })

        return extracted


if __name__ == "__main__":
    # Demo
    kg = KnowledgeGraph(persist_path="./memory_db/kg_test.json")
    extractor = GraphExtractor(kg)

    test_turns = [
        ("我最喜欢紫色的薰衣草，因为小时候奶奶家后院种了一大片。", "真的吗？薰衣草确实很美。"),
        ("我养了一只柯基犬，它叫豆豆。", "豆豆好可爱的名字！它多大了？"),
        ("我今天升职了！", "哇！太好了！恭喜你！"),
        ("你就是个人工智能程序而已。", "你说话能不能尊重一点？"),
        ("我住在北京朝阳区工作", "朝阳区啊，离我也不远呢。"),
    ]

    print("=" * 60)
    print("Graph Extractor Demo")
    print("=" * 60)

    for user_msg, assistant_msg in test_turns:
        results = extractor.extract_from_turn(user_msg, assistant_msg)
        print(f"\nUser: {user_msg}")
        for r in results:
            print(f"  → ({r['subject']}) --[{r['relation']}]--> ({r['object']})")

    print(f"\nGraph stats: {kg.get_stats()}")
    print(f"\nContext for '用户':")
    print(kg.to_context_string("用户"))

    # Cleanup test file
    import os
    os.remove("./memory_db/kg_test.json")
