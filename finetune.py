import argparse
import logging
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# Prepare model and tokenizer for training
def prepare_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # use EOS token as PAD

    device_map = "mps" if torch.backends.mps.is_available() else None
    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype
    )
    logger.info(f"Loaded model `{model_name}` on {device_map or 'auto'} with dtype {dtype}")
    return model, tokenizer

# Tokenize and format dataset into model inputs
def preprocess(batch, tokenizer, max_length: int = 512):
    # Prefix user/assistant roles
    formatted = [f"User: {p}\nAssistant: {r}" for p, r in zip(batch["prompt"], batch["response"])]
    tokens = tokenizer(
        formatted,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokens.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss
    tokens["labels"] = labels
    return tokens

# Load JSONL dataset, split, and tokenize
def load_and_process(data_path: str, tokenizer, test_size: float = 0.1):
    raw = load_dataset("json", data_files=data_path)
    ds = raw.get("train") or raw[list(raw.keys())[0]]

    splits = ds.train_test_split(test_size=test_size)
    # Apply preprocessing
    tokenized = DatasetDict({
        split: splits[split].map(
            lambda batch: preprocess(batch, tokenizer),
            batched=True,
            remove_columns=["prompt", "response"]
        )
        for split in splits
    })
    logger.info(f"Dataset loaded and tokenized: {data_path}")
    return tokenized["train"], tokenized["test"]

# Apply LoRA for parameter-efficient fine-tuning
def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    lora_model = get_peft_model(model, config)
    logger.info("LoRA model wrapping complete.")
    return lora_model

# Main training and evaluation routine
def train_and_save(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    seed: int
):
    set_seed(seed)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        optim="adamw_torch",
        fp16=False,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # Initialize Trainer with datasets and tokenizer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Start fine-tuning
    trainer.train()

    # Save model, LoRA adapters, and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training complete. Artifacts saved to {output_dir}")

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with LoRA")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_path", type=str, default="dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_tinyllama")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use as validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = prepare_model_and_tokenizer(args.model_name)
    train_ds, eval_ds = load_and_process(args.data_path, tokenizer, args.test_size)
    lora_model = apply_lora(model)
    train_and_save(
        lora_model,
        tokenizer,
        train_ds,
        eval_ds,
        args.output_dir,
        args.seed
    )
    print("✅ Fine-tuning complete! Model and tokenizer saved.")
