"""
DPO Training Script — Direct Preference Optimization
Refines the SFT-trained Lin Xia model to prefer emotionally intelligent responses.
Uses trl.DPOTrainer with LoRA for memory-efficient alignment.
"""

import torch
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from config import *


def load_preference_data(path: str = "dpo_preference_data.json") -> Dataset:
    """Load and format DPO preference pairs into HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    for item in raw_data:
        formatted.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    return Dataset.from_list(formatted)


def train_dpo():
    """Main DPO training function. Requires SFT model to be already trained."""

    sft_model_path = f"{OUTPUT_DIR}/merged_model"

    print("=" * 60)
    print("DPO Alignment Training for Emotional AI")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU required for DPO training.")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine precision
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load the SFT-trained model as starting point
    print(f"\nLoading SFT model from: {sft_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA config for DPO (lighter than SFT — we're refining, not learning from scratch)
    peft_config = LoraConfig(
        r=16,                # Lower rank than SFT (32) for subtle refinement
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load preference dataset
    print("\nLoading preference data...")
    dataset = load_preference_data()
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Evaluation pairs: {len(eval_dataset)}")

    # DPO training configuration
    dpo_config = DPOConfig(
        output_dir=f"{OUTPUT_DIR}/dpo_model",
        beta=0.1,                    # KL-divergence weight: balance personality sharpness vs. coherence
        learning_rate=5e-6,          # Lower LR than SFT for subtle alignment
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_steps=20,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # Initialize DPO trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL creates implicit reference from initial weights
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO alignment training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print("\nSaving DPO-aligned model...")
    trainer.save_model(f"{OUTPUT_DIR}/dpo_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/dpo_lora")

    # Merge and save full model
    print("Merging DPO LoRA with base...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/dpo_merged_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/dpo_merged_model")

    print("\n" + "=" * 60)
    print("DPO training complete!")
    print(f"DPO LoRA saved to: {OUTPUT_DIR}/dpo_lora")
    print(f"Merged model saved to: {OUTPUT_DIR}/dpo_merged_model")
    print("=" * 60)


if __name__ == "__main__":
    train_dpo()
