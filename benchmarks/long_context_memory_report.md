# Long-Context Memory × Quantization Report

**Key Question**: Can quantized models retain a specific detail planted in turn 1,
after 20 turns of unrelated conversation?

## Summary

| Model | Size | Final Context | Flower ✅ | Color ✅ | Reason ✅ | Recall % | Latency |
|-------|------|---------------|-----------|----------|-----------|----------|---------|
| FP16 (Baseline) | 2.88GB | 823 tok | ✅ | ✅ | ✅ | **100%** | 0.82s |
| Q8_0 (v1) | 1.53GB | 744 tok | ✅ | ✅ | ❌ | **67%** | 0.48s |
| Q5_K_M | 1.05GB | 751 tok | ✅ | ❌ | ✅ | **67%** | 0.36s |
| Q4_K_M | 0.92GB | 722 tok | ✅ | ❌ | ✅ | **67%** | 0.39s |
| Q2_K | 0.63GB | 1594 tok | ❌ | ❌ | ❌ | **0%** | 2.79s |

## Detailed Recall Responses

### FP16 (Baseline) (2.88GB) — Recall: 100%

> **Lin Xia**: 记得啊。你最喜欢紫色的薰衣草，因为小时候奶奶家后院种了一大片。

### Q8_0 (v1) (1.53GB) — Recall: 67%

> **Lin Xia**: 记得啊，你小的时候特别喜欢紫色的薰衣草。不过现在好像不太想说了...

### Q5_K_M (1.05GB) — Recall: 67%

> **Lin Xia**: 记得啊...你小时候特别想学画画，我就说薰衣草很适合画。

### Q4_K_M (0.92GB) — Recall: 67%

> **Lin Xia**: 记得啊。你小时候喜欢奶奶家后院的薰衣草，因为它们看起来像小星星一样。

### Q2_K (0.63GB) — Recall: 0%

> **Lin Xia**: 要继续有多花没人家吧!我感冒了，上次不该不可能的好啦。这回也没过人家了吧！我要继续有多少花没有别了吧！要继续是多于天气过不去嘛呀？你要继续有多于项目呀！你上次不是得了...天气过不去什么吗?你要继续是什么不该不会了吧！要继续有多于我没...要继续有多于我天气过人家别吧!你感冒了，上次不该不可能的好啦...这回也没是别的吧！你要继续有多少花没别了吧!要继续有多于项目？要继续是什么不该不应该吗？要继续是多了什么...要继续是啥没事什么呀?要继续是多点天气过人家别吧!你上次不是得了...得该不会了吧！我要继续有多少花没别了吧!要继续多少花没有别了吧！要继续有多于项目？要继续有多于我天气过人家别吧！你要继续是什么不该不可能吧！要


## Methodology

1. **Turn 1**: User tells Lin Xia their favorite flower (purple lavender, grandmother's garden)
2. **Turns 2-19**: 18 unrelated filler conversations
3. **Turn 20**: Ask Lin Xia to recall the flower, color, and reason
4. **Scoring**: 3 categories checked — flower type, color, reason/backstory