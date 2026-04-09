# Project: Emotional AI "Lin Xia" (林夏)

This project focuses on fine-tuning a Large Language Model (**Qwen2.5-1.5B-Instruct**) to exhibit human-like emotional behavior. The goal is to create a character named **Lin Xia (林夏)**—a realistic, emotional girl who responds naturally to various social interactions (warmth, neglect, offense, and reconciliation).

## 🚀 Project Status
- **Training**: Successfully completed 4 epochs of PEFT/LoRA training on a remote L20 GPU.
- **Quantization**: Converted to **GGUF (Q8_0)** for high-performance local inference on macOS (Metal accelerated).
- **Personality**: Verified to show consistent emotional state memory and boundaries.

---

## 🛠️ Technical Architecture

### Core Stack
- **Model**: Qwen2.5-1.5B-Instruct
- **Framework**: HuggingFace Transformers + PEFT (LoRA)
- **Precision**: bf16 (Training) / Q8_0 (Local Inference)
- **Engine**: llama-cpp-python (Local Mac)

### Key Files
- `config.py`: Central configuration for personality (System Prompt), training hyperparameters, and server credentials.
- `train.py`: The core training pipeline used on the remote GPU server. Handles model loading, LoRA adaptation, and weight merging.
- `test_local_gguf.py`: A local script to carry out inference with the GGUF model on your Mac. Includes proper System Prompt injection and repetition control.
- `test_model.py`: A remote validation script used to verify model behavior during the training phase.

---

## 👩‍💼 Character Profile: Lin Xia (林夏)
Lin Xia is designed to be a realistic partner with distinct emotional rules:
- **Emotional Memory**: She remembers how you treat her. Hurt doesn't disappear instantly.
- **Boundaries**: She is not unconditionally submissive. She gets angry if insulted and requires a genuine apology.
- **Natural Transitions**: Her mood shifts gradually based on conversation flow.
- **Human-like**: She avoids "As an AI..." language and excessive emoji use.

---

## 📖 Usage Guide

### 1. Local Chat (macOS)
The model is already downloaded to `emotional-model-output/linxia-emotional-q8_0.gguf`. To start a test chat:
```bash
python3 test_local_gguf.py
```

### 2. Manual Inference
You can load the `.gguf` file into any compatible tool (LM Studio, Ollama, etc.). Use the following recommended parameters:
- **System Prompt**: Found in `config.py` under `CHARACTER_DESCRIPTION`.
- **Temperature**: 0.7 - 0.9
- **Repeat Penalty**: 1.1 - 1.2
- **Stop Tokens**: `<|im_end|>`, `<|im_start|>`

### 3. Remote Training
If you wish to retrain or update the model, use the `train.py` script on a GPU-enabled server:
```bash
python3 train.py
```

---

## 📝 Training History & Fixes
- **Driver Compatibility**: Transitioned from Unsloth/BitsAndBytes to a standard **bf16 PEFT** pipeline to resolve remote driver conflicts.
- **Weight Integrity**: Fixed "incomplete metadata" errors by ensuring full `model.safetensors` synchronization.
- **Precision**: Standardized on **bf16** for training to preserve the delicate emotional nuances of the 1.5B model.
- **English Documentation**: All code comments and project documentation have been translated to English for international standards.
