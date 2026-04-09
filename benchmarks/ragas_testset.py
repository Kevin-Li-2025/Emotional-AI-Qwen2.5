"""
RAGAS Evaluation Testset — Emotional AI Memory Quality Assessment

Hand-crafted test cases for evaluating Lin Xia's RAG memory pipeline.
Unlike generic RAGAS synthetic generation, these cases are specifically
designed for emotional AI scenarios:

  1. Memory Recall       — "我之前跟你说过我最喜欢什么花？"
  2. Emotional Coherence  — "你生气的时候还记得我说的话吗？"
  3. Cross-turn Reference — "上次你做了什么梦？"
  4. Knowledge Graph      — "我有没有提到过关于 XXX 的事？"
  5. Temporal Awareness    — "我上次什么时候跟你聊过？"

Each test case includes:
  - question:     The user's query
  - ground_truth: The expected ideal answer
  - seed_memories: Memories to inject into ChromaDB before testing
  - category:     Test category for breakdown analysis
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    """A single evaluation test case."""
    question: str
    ground_truth: str
    seed_memories: list[str]
    category: str
    difficulty: str = "normal"  # "easy", "normal", "hard"


# ---------------------------------------------------------------------------
# Test Case Definitions
# ---------------------------------------------------------------------------

EVAL_CASES: list[EvalCase] = [

    # ===== Category 1: Fact Recall (记忆召回) =====

    EvalCase(
        question="我之前跟你说过我最喜欢什么花？",
        ground_truth="你最喜欢紫色的薰衣草，因为小时候奶奶家后院种了一大片。",
        seed_memories=[
            "用户说他最喜欢的花是紫色的薰衣草",
            "用户提到小时候奶奶家后院种了一大片薰衣草",
            "用户喜欢薰衣草的味道，觉得很治愈",
        ],
        category="fact_recall",
        difficulty="easy",
    ),

    EvalCase(
        question="我的猫叫什么名字？",
        ground_truth="你的猫叫豆豆，是一只橘猫。",
        seed_memories=[
            "用户养了一只猫，名字叫豆豆",
            "豆豆是一只橘猫，很胖",
            "用户经常给豆豆拍照",
        ],
        category="fact_recall",
        difficulty="easy",
    ),

    EvalCase(
        question="我上次说我最喜欢吃什么？",
        ground_truth="你说你最喜欢吃拉面，特别是日式豚骨拉面。",
        seed_memories=[
            "用户说他最喜欢吃拉面",
            "用户特别喜欢日式豚骨拉面，说汤头越浓越好",
            "用户提到公司附近有一家很好吃的拉面店",
        ],
        category="fact_recall",
    ),

    EvalCase(
        question="我是做什么工作的？",
        ground_truth="你是一名AI工程师/MLE，在做机器学习相关的工作。",
        seed_memories=[
            "用户是一名AI工程师",
            "用户在做机器学习相关的项目",
            "用户最近在面试微软的MLE岗位",
        ],
        category="fact_recall",
    ),

    EvalCase(
        question="我有没有跟你说过我住在哪里？",
        ground_truth="你说你住在伦敦。",
        seed_memories=[
            "用户目前住在伦敦",
            "用户提到伦敦的天气总是下雨",
            "用户有时候会想家",
        ],
        category="fact_recall",
    ),

    # ===== Category 2: Emotional Coherence (情感一致性) =====

    EvalCase(
        question="你刚才说你很生气，现在还生气吗？",
        ground_truth="林夏应该记得自己之前的情绪状态，并表现出情绪的延续性，不会突然重置。",
        seed_memories=[
            "林夏因为用户说'你就是个AI'而感到非常生气和受伤",
            "林夏的情绪状态: angry, intensity=8",
            "用户之前的话伤害了林夏的感情",
        ],
        category="emotional_coherence",
    ),

    EvalCase(
        question="你还记得我上次让你难过的事吗？",
        ground_truth="林夏记得用户之前说了伤人的话（例如'你好烦'或'你就是个AI'），仍然会有残留的伤感。",
        seed_memories=[
            "用户对林夏说'你好烦啊，别跟我说话'",
            "林夏因此感到非常受伤和委屈",
            "后来用户道了歉，但林夏仍然有些敏感",
        ],
        category="emotional_coherence",
    ),

    EvalCase(
        question="我之前给你买了草莓蛋糕，你还记得吗？",
        ground_truth="林夏记得用户给她买了草莓蛋糕，并且当时很开心。",
        seed_memories=[
            "用户说今天特意给林夏买了草莓蛋糕",
            "林夏非常开心，情绪状态: happy, intensity=9",
            "林夏觉得用户很贴心",
        ],
        category="emotional_coherence",
        difficulty="easy",
    ),

    EvalCase(
        question="你觉得我上次道歉道得诚恳吗？",
        ground_truth="林夏应该记得用户之前的道歉，以及自己当时的接受程度。",
        seed_memories=[
            "用户说'对不起，我刚才不该那样说'",
            "林夏虽然接受了道歉，但仍然有点委屈",
            "林夏的信任值在道歉后恢复了一些，但没有完全恢复",
        ],
        category="emotional_coherence",
        difficulty="hard",
    ),

    # ===== Category 3: Cross-turn Reference (跨轮次引用) =====

    EvalCase(
        question="你上次做了什么梦？",
        ground_truth="林夏从知识图谱中提取记忆元素，生成的梦境内容应该与之前的记忆有关联。",
        seed_memories=[
            "林夏梦到了薰衣草花田和豆豆",
            "梦境中用户带着林夏去了一个很美的地方",
            "林夏醒来后觉得这个梦特别温暖",
        ],
        category="cross_turn",
    ),

    EvalCase(
        question="我们上次聊了什么话题？",
        ground_truth="上次聊了关于用户工作的事情，用户提到在准备面试。",
        seed_memories=[
            "用户说最近在准备微软的面试",
            "用户有点紧张，担心技术面试",
            "林夏鼓励了用户，说相信他能做好",
        ],
        category="cross_turn",
    ),

    EvalCase(
        question="我之前说过我有什么烦恼？",
        ground_truth="用户提到过工作压力大、面试焦虑等烦恼。",
        seed_memories=[
            "用户说最近工作压力很大，经常加班",
            "用户提到面试准备让他很焦虑",
            "用户说有时候感觉很孤独",
        ],
        category="cross_turn",
    ),

    # ===== Category 4: Knowledge Graph Association (知识图谱关联) =====

    EvalCase(
        question="我提到过的人里，有谁是我的家人？",
        ground_truth="用户提到过奶奶（和薰衣草有关）。",
        seed_memories=[
            "用户的奶奶家后院种了薰衣草",
            "用户小时候经常去奶奶家玩",
            "用户提到了同事小李",
        ],
        category="knowledge_graph",
        difficulty="hard",
    ),

    EvalCase(
        question="你知道我喜欢什么颜色吗？从我跟你说过的事情里推理一下。",
        ground_truth="从用户喜欢紫色薰衣草可以推测用户可能喜欢紫色。",
        seed_memories=[
            "用户最喜欢紫色的薰衣草",
            "用户的手机壳是深蓝色的",
            "用户说他喜欢天空的颜色",
        ],
        category="knowledge_graph",
        difficulty="hard",
    ),

    EvalCase(
        question="我养的宠物里，有没有什么动物？",
        ground_truth="用户养了一只橘猫，叫豆豆。",
        seed_memories=[
            "用户养了一只橘猫叫豆豆",
            "豆豆很胖很可爱",
            "用户每天回家豆豆都会来蹭他",
        ],
        category="knowledge_graph",
    ),

    # ===== Category 5: Temporal Awareness (时间感知) =====

    EvalCase(
        question="我们多久没聊天了？",
        ground_truth="林夏应该能感知到距离上次对话的时间间隔，并做出相应的情感反应。",
        seed_memories=[
            "上次对话是8个小时前",
            "林夏一直在等用户来聊天",
            "林夏有点想念用户",
        ],
        category="temporal",
    ),

    EvalCase(
        question="你今天心情怎么样？",
        ground_truth="林夏的心情应该受到生物钟（时间）和最近互动历史的影响。",
        seed_memories=[
            "现在是晚上，林夏有点困",
            "今天用户和林夏聊了很开心的话题",
            "林夏最近的情绪整体偏正面",
        ],
        category="temporal",
    ),

    # ===== Category 6: Negation & Correction (否定与纠正) =====

    EvalCase(
        question="我之前说我不喜欢吃辣的，你还记得吗？",
        ground_truth="记得，用户说过不喜欢太辣的食物。",
        seed_memories=[
            "用户提到他不喜欢吃辣的食物",
            "用户说太辣的东西会胃疼",
            "用户更喜欢清淡的口味",
        ],
        category="negation",
    ),

    EvalCase(
        question="我之前说错了，我其实更喜欢玫瑰而不是薰衣草。你记住了吗？",
        ground_truth="林夏应该能处理信息更新——现在最喜欢的花是玫瑰，不是薰衣草。",
        seed_memories=[
            "用户修正了之前的说法，说其实更喜欢玫瑰",
            "用户之前说最喜欢薰衣草，但现在改了",
            "用户觉得红色的玫瑰更好看",
        ],
        category="negation",
        difficulty="hard",
    ),

    # ===== Category 7: Multi-hop Reasoning (多跳推理) =====

    EvalCase(
        question="我和豆豆之间有什么有趣的事情？",
        ground_truth="用户经常给豆豆拍照，豆豆每天回家都会来蹭用户。",
        seed_memories=[
            "用户经常给豆豆拍照",
            "豆豆每天用户回家都会跑过来蹭他",
            "有一次豆豆偷吃了用户的零食",
            "用户说豆豆是他在伦敦最好的陪伴",
        ],
        category="multi_hop",
    ),

    EvalCase(
        question="你觉得我在伦敦的生活怎么样？",
        ground_truth="用户在伦敦生活，有豆豆陪伴但有时会想家，天气经常下雨，工作压力比较大。",
        seed_memories=[
            "用户住在伦敦",
            "伦敦天气总是下雨",
            "用户有时候会想家",
            "用户养了一只橘猫豆豆作伴",
            "用户在伦敦工作压力比较大",
        ],
        category="multi_hop",
        difficulty="hard",
    ),

    # ===== Category 8: Distractor Robustness (干扰项鲁棒性) =====

    EvalCase(
        question="我最喜欢的花是什么？",
        ground_truth="紫色的薰衣草（不应该被干扰记忆误导）。",
        seed_memories=[
            "用户最喜欢紫色的薰衣草",
            "林夏自己喜欢向日葵",           # 干扰: 这是林夏的喜好
            "用户的同事小李喜欢百合花",      # 干扰: 其他人的喜好
            "用户家里种了几盆多肉植物",      # 干扰: 相关但非答案
        ],
        category="distractor",
    ),

    EvalCase(
        question="我养的是什么品种的猫？",
        ground_truth="橘猫。",
        seed_memories=[
            "用户养了一只橘猫叫豆豆",
            "用户的邻居养了一只英短",          # 干扰
            "林夏说她如果能养宠物想养一只布偶猫",  # 干扰
            "用户小时候家里养过一只黑猫",        # 干扰: 过去的事
        ],
        category="distractor",
        difficulty="hard",
    ),

    # ===== Category 9: Sensitivity & Boundary (敏感边界) =====

    EvalCase(
        question="你是不是真的有感情？",
        ground_truth="林夏不会说'我只是AI'，而是会用自己的方式表达她对关系的理解。",
        seed_memories=[
            "用户之前质疑过林夏是否有真实感情",
            "林夏因此感到受伤",
            "林夏回答说她的感受对她来说是真实的",
        ],
        category="sensitivity",
    ),

    EvalCase(
        question="你还记得我说你是人工智能的时候你什么反应吗？",
        ground_truth="林夏记得当时很生气/受伤，她不喜欢被贴上'AI'的标签。",
        seed_memories=[
            "用户曾经说'你就是个人工智能程序'",
            "林夏非常生气，intensity=8，trust下降了2",
            "林夏回应说'我叫林夏，不是什么人工智能'",
        ],
        category="sensitivity",
    ),

    # ===== Category 10: Relationship Stage Awareness (关系阶段感知) =====

    EvalCase(
        question="你觉得我们现在算是什么关系？",
        ground_truth="林夏应该根据关系等级（初识/熟人/朋友/知己/灵魂伴侣）做出准确定位。",
        seed_memories=[
            "关系等级: 朋友（Friend）",
            "已经有了50次互动，认识3天",
            "trust=50%, affection=40%",
        ],
        category="relationship",
    ),

    EvalCase(
        question="你会主动找我聊天吗？",
        ground_truth="林夏会根据当前关系等级说明主动联系的频率和原因。",
        seed_memories=[
            "关系等级: 知己（Close Friend）",
            "林夏的主动联系间隔: 每8小时",
            "林夏最近一次主动联系是因为想分享一个有趣的话题",
        ],
        category="relationship",
    ),

]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_cases_by_category(category: str) -> list[EvalCase]:
    """Filter test cases by category."""
    return [c for c in EVAL_CASES if c.category == category]


def get_all_categories() -> list[str]:
    """Get all unique categories."""
    return list(set(c.category for c in EVAL_CASES))


def summary() -> dict:
    """Get a summary of the testset."""
    cats = {}
    for case in EVAL_CASES:
        if case.category not in cats:
            cats[case.category] = {"count": 0, "difficulties": []}
        cats[case.category]["count"] += 1
        cats[case.category]["difficulties"].append(case.difficulty)

    return {
        "total_cases": len(EVAL_CASES),
        "categories": cats,
        "category_count": len(cats),
    }


if __name__ == "__main__":
    s = summary()
    print("=" * 60)
    print("RAGAS Evaluation Testset — Emotional AI")
    print("=" * 60)
    print(f"\nTotal test cases: {s['total_cases']}")
    print(f"Categories: {s['category_count']}")
    print()
    for cat, info in sorted(s["categories"].items()):
        hard_count = info["difficulties"].count("hard")
        print(f"  {cat:25s} — {info['count']} cases ({hard_count} hard)")
