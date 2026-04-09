# Emotional AI — Lin Xia (林夏) v2.0

A full-stack emotional AI system built on **Qwen2.5-1.5B-Instruct**, fine-tuned to exhibit human-like emotional behavior. Lin Xia is a realistic, emotionally intelligent companion with persistent memory, context compression, and multi-modal output capabilities.

## ✨ Features

| Feature | Description | Status |
|:--------|:------------|:-------|
| **SFT Fine-tuning** | LoRA-based personality training on 1000+ emotional conversations | ✅ Complete |
| **DPO Alignment** | Direct Preference Optimization for sharper emotional responses | ✅ Built |
| **Quantization Study** | FP16 → Q2_K emotional fidelity benchmark with latency analysis | ✅ Built |
| **KV Cache Optimization** | Multi-turn throughput analysis with cache quantization | ✅ Built |
| **Context Compression** | Sliding window + emotional state summary for infinite conversation | ✅ Built |
| **Persistent Memory** | ChromaDB RAG — she remembers you across sessions | ✅ Built |
| **Data Pipeline** | 15+ scenario generator, AI judge scoring, distribution visualization | ✅ Built |
| **TTS Integration** | Emotion-tagged speech synthesis (CosyVoice / GPT-SoVITS) | ✅ Built |
| **Unified App** | CLI + Gradio web interface with real-time status display | ✅ Built |

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
│   ├── generate_dpo_pairs.py    # DPO preference pair creation
│   ├── ai_judge.py              # Automated quality scoring (5 dimensions)
│   └── visualize_data.py        # Interactive HTML distribution report
│
├── benchmarks/                  # Performance research
│   ├── quantization_benchmark.py # Multi-level quant comparison
│   └── kv_cache_benchmark.py    # KV cache strategy analysis
│
├── context_engine/              # Context compression
│   ├── context_manager.py       # Budget-based context allocation
│   └── sliding_summary.py       # Sliding window + emotional summary
│
├── memory/                      # Long-term persistent memory
│   ├── memory_store.py          # ChromaDB vector store
│   ├── memory_extractor.py      # LLM-based fact extraction
│   └── memory_retriever.py      # Recency-biased semantic retrieval
│
└── voice/                       # Text-to-speech
    └── tts_engine.py            # Emotion-aware TTS (CosyVoice/GPT-SoVITS)
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

### Quantization Study
Benchmarks across 5 quantization levels (FP16 → Q2_K) measure:
- Tokens per second (throughput)
- File size (deployment cost)
- Emotional fidelity (scored by AI judge on 15 standardized prompts)

## 📊 Training Results (v1.0)
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Method**: PEFT LoRA (r=32, α=64) on L20 GPU
- **Epochs**: 4 | **Final Loss**: 0.179
- **Dataset**: 1,000 conversations across 6 emotional scenarios
- **Quantized**: GGUF Q8_0 (1.6GB) for local Mac inference (Metal accelerated)

## 📄 License
MIT
