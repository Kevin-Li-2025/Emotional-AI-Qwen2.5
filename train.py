"""Train emotional model using PEFT LoRA (no Unsloth dependency)"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from config import *


def format_conversation(example):
    """Format conversation for training"""
    messages = example["conversations"]

    # Convert to chat format
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return {"messages": formatted}


def formatting_prompts_func(examples, tokenizer):
    """Apply chat template to format conversations for training"""
    convos = examples["messages"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def train_model():
    """Main training function"""

    print("=" * 50)
    print("Emotional Model Training with PEFT LoRA")
    print("=" * 50)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. GPU is required for training.")
        return
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model in bf16 (The 1.5B model fits easily in the L20 GPU's 48GB VRAM)
    print(f"\nLoading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA config
    print("\nAdding LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_conversation)

    # Apply chat template to create text field
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save final LoRA adapter
    print("\nSaving final LoRA adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")

    # Merge LoRA weights with base model and save
    print("\nMerging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_model")

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}/final_lora")
    print(f"Merged model saved to: {OUTPUT_DIR}/merged_model")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
