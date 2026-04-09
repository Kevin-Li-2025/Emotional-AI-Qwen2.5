# Emotional AI — Lin Xia (林夏) v2.0

A full-stack emotional AI system built on **Qwen2.5-1.5B-Instruct**, fine-tuned to exhibit human-like emotional behavior. Lin Xia is a realistic, emotionally intelligent companion with persistent memory, context compression, and multi-modal output capabilities.

## ✨ Features

| Feature | Description | Status |
|:--------|:------------|:-------|
| **SFT Fine-tuning** | LoRA-based personality training on 1000+ emotional conversations | ✅ Trained |
| **DPO Alignment** | Direct Preference Optimization — reward accuracy 100%, margin 10.65 | ✅ Trained |
| **Model-Level Emotions** | Model outputs `<emotion>` tags with mood/trust/affection deltas | ✅ Trained |
| **Quantization Study** | FP16/Q8_0/Q5_K_M/Q4_K_M/Q2_K emotional fidelity + throughput | ✅ Benchmarked |
| **Long-Context Memory** | 20-turn recall test across quantization levels (Q4_K_M=67→100%) | ✅ Researched |
| **Attention Sink** | SmartContextBuilder — anchor facts + first-turn retention | ✅ Implemented |
| **Context Compression** | Sliding window + emotional state summary for infinite conversation | ✅ Implemented |
| **Persistent Memory** | ChromaDB RAG — she remembers you across sessions | ✅ Implemented |
| **Data Pipeline** | 15+ scenario generator, AI judge, offline DPO pair generation | ✅ Implemented |
| **TTS Integration** | Edge TTS with emotion-to-SSML prosody mapping (8 emotions) | ✅ Integrated |
| **Unified App** | CLI + Gradio web interface with real-time status display | ✅ Implemented |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                   User Input                      │
├──────────────┬───────────────────────────────────┤
│ Memory       │ Context Engine                     │
│ Retriever    │ ┌─────────────┐ ┌───────────────┐ │
│ (ChromaDB)   │ │ Sliding     │ │ Emotional     │ │
│     ↓        │ │ Summary     │ │ State Tracker │ │
│ Relevant     │ └─────────────┘ └───────────────┘ │
│ Memories     │           ↓                        │
├──────────────┴───────────────────────────────────┤
│              Optimized Prompt Builder              │
│  [System + Emotion State + Memories + Recent Turns]│
├──────────────────────────────────────────────────┤
│          Qwen2.5-1.5B (GGUF / Q8_0)              │
├──────────────────────────────────────────────────┤
│              Response + Emotion Tag               │
├──────────┬────────────────────┬──────────────────┤
│ Memory   │ Context            │ TTS Engine       │
│ Extractor│ Update             │ (CosyVoice)      │
│ → Store  │ → Sliding Summary  │ → Voice Output   │
└──────────┴────────────────────┴──────────────────┘
```

## 📁 Project Structure

```
├── app.py                       # Unified chat app (CLI + Gradio)
├── train.py                     # SFT training pipeline
├── train_dpo.py                 # DPO alignment training
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── data_pipeline/               # Data engineering
│   ├── generate_diverse.py      # 15+ scenario data generation
│   ├── generate_dpo_pairs.py    # DPO preference pair creation (API)
│   ├── generate_dpo_offline.py  # DPO pairs from existing data (no API)
│   ├── ai_judge.py              # Automated quality scoring (5 dimensions)
│   └── visualize_data.py        # Interactive HTML distribution report
│
├── benchmarks/                  # Performance research
│   ├── quantization_benchmark.py      # Multi-level quant comparison
│   ├── kv_cache_benchmark.py          # KV cache strategy analysis
│   └── long_context_memory_benchmark.py # 20-turn recall × quant study
│
├── context_engine/              # Context compression + optimization
│   ├── context_manager.py       # Budget-based context allocation
│   ├── sliding_summary.py       # Sliding window + emotional summary
│   ├── emotional_state_model.py # Model-level emotion tag output
│   └── smart_context.py         # Attention sink + anchor optimization
│
├── memory/                      # Long-term persistent memory
│   ├── memory_store.py          # ChromaDB vector store
│   ├── memory_extractor.py      # LLM-based fact extraction
│   └── memory_retriever.py      # Recency-biased semantic retrieval
│
└── voice/                       # Text-to-speech
    └── tts_engine.py            # Edge TTS with emotion-to-SSML mapping
```

## 🚀 Quick Start

### Chat (CLI)
```bash
python3 app.py
```

### Chat (Web UI)
```bash
python3 app.py --ui
```

### Chat with TTS
```bash
python3 app.py --tts --tts-backend cosyvoice
```

### Run Quantization Benchmark
```bash
python3 -m benchmarks.quantization_benchmark
```

### Run KV Cache Benchmark
```bash
python3 -m benchmarks.kv_cache_benchmark
```

### Generate Training Data (v2)
```bash
python3 -m data_pipeline.generate_diverse
```

### Train with DPO
```bash
python3 train_dpo.py
```

## 🧠 Technical Highlights

### Context Compression
The `ContextManager` allocates the context window budget across 4 layers:
- **15%** — System prompt (character definition)
- **5%** — Emotional state vector (compressed mood/trust/affection)
- **20%** — Retrieved memories from ChromaDB
- **60%** — Recent conversation turns (sliding window)

When older turns overflow, they are summarized by the LLM into compact emotional state blocks, creating the illusion of **infinite context** while staying within the model's actual window.

### Memory Architecture
Memories are stored in three types:
- **Facts**: "User works in tech", "User's birthday is March 15"
- **Emotional Events**: "User insulted Lin Xia, she was hurt (severity: 8/10)"
- **Preferences**: "User prefers playful responses over formal ones"

Retrieval uses a **combined scoring** function: `(1-w) × relevance × importance + w × recency`, with a 7-day half-life decay for recency.

### Quantization × Memory Retention Study

| Model | Size | Avg tok/s | 20-Turn Recall | Optimized Recall |
|:------|:-----|:----------|:---------------|:-----------------|
| FP16 | 2.88GB | 25.7 | **100%** | — |
| Q8_0 | 1.53GB | 43.9 | 67% | 67% |
| Q5_K_M | 1.05GB | 53.9 | 67% | 33% |
| **Q4_K_M** | **0.92GB** | **61.2** | 67% | **100%** ✅ |
| Q2_K | 0.63GB | 63.7 | 0% | 0% |

**Key finding**: Q4_K_M + Attention Sink optimization achieves **100% recall** at 20 turns — matching FP16 at **1/3 the size** and **2.4× the speed**.

### DPO Alignment Results
- **Reward Accuracy**: 100% (model perfectly distinguishes chosen vs rejected)
- **Reward Margin**: 10.65 (strong preference signal)
- **Eval Accuracy**: 98.4%
- **Training Loss**: 0.0066

## 📊 Training History

| Stage | Method | Loss | Dataset | GPU |
|:------|:-------|:-----|:--------|:----|
| SFT v1 | LoRA (r=32, α=64) | 0.179 | 1,000 convos | L20 |
| SFT v2 (emotion-aware) | LoRA (r=32, α=64) | **0.164** | 1,000 + emotion tags | L20 |
| DPO | LoRA (r=16, α=32) | **0.007** | 3,682 preference pairs | L20 |

## 📄 License
MIT
