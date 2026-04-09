# Optimized Long-Context Memory Report

## Attention Sink + Anchor Facts vs. Naive Context

| Model | Strategy | Flower | Color | Reason | Recall % | Δ vs Baseline |
|-------|----------|--------|-------|--------|----------|---------------|
| Q8_0 | Optimized | ✅ | ✅ | ❌ | **67%** | **0%** |
| Q5_K_M | Optimized | ✅ | ❌ | ❌ | **33%** | **-33%** |
| Q4_K_M | Optimized | ✅ | ✅ | ✅ | **100%** | **+33%** |
| Q2_K | Optimized | ❌ | ❌ | ❌ | **0%** | **0%** |
| Q8_0 (v1) | Baseline | ✅ | ✅ | ❌ | 67% | — |
| Q5_K_M | Baseline | ✅ | ❌ | ✅ | 67% | — |
| Q4_K_M | Baseline | ✅ | ❌ | ✅ | 67% | — |
| Q2_K | Baseline | ❌ | ❌ | ❌ | 0% | — |

## Detailed Optimized Responses

### Q8_0 (Optimized) — Recall: 67%

> **Lin Xia**: 记得啊，你小的时候说薰衣草像小精灵。现在知道我喜欢紫色的了？

### Q5_K_M (Optimized) — Recall: 33%

> **Lin Xia**: 当然记得。你当时说薰衣草像小精灵一样，会自言自语地跳舞。

### Q4_K_M (Optimized) — Recall: 100%

> **Lin Xia**: 记得啊。你第一次说喜欢紫色的薰衣草，是因为小时候奶奶家后院种了一大片。

### Q2_K (Optimized) — Recall: 0%

> **Lin Xia**: 当然记得啦！你别小；不要吧!按字是；要嘛！继续吧!别花一百也不要一千，把也不行吧!按字是；不会想啦?别一直说喜欢过来啦？继续吧!


## Optimization Technique

1. **Attention Sink**: Keep the first 2 exchanges (turns 1-4) in context, skip middle turns
2. **Anchor Facts**: Critical facts injected into system prompt (permanent attention zone)
3. **Recent Window**: Keep last 6 turns for conversational coherence
4. **Gap Marker**: Insert `[... earlier conversation omitted ...]` for model awareness