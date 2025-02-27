import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. Setup Model & Tokenizer ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

# Load model with MPS (Apple Silicon) and bfloat16 precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",  # Use MPS for M1/M2 chips
    torch_dtype=torch.bfloat16  # Preferred over float16 on Apple Silicon
)

# --- 2. Preprocess Dataset ---
def preprocess_function(examples):
    # Format: "User: {prompt}\nAssistant: {response}"
    inputs = [
        f"User: {prompt}\nAssistant: {response}"
        for prompt, response in zip(examples["prompt"], examples["response"])
    ]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    # Labels = input IDs (shifted right by 1)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
    return model_inputs

# Load and preprocess dataset
dataset = load_dataset("json", data_files="dataset.jsonl")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["prompt", "response"]  # Remove raw text columns
)

# --- 3. Apply LoRA (Memory-Efficient Fine-Tuning) ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Reduced rank for M1 memory
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Key layers for TinyLlama
    bias="none"
)
model = get_peft_model(model, peft_config)

# --- 4. Training Configuration ---
training_args = TrainingArguments(
    output_dir="./fine_tuned_tinyllama",
    per_device_train_batch_size=2,  # Adjust based on M1 memory
    gradient_accumulation_steps=4,  # Simulate larger batch size
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  # Lower LR for stability
    num_train_epochs=3,
    logging_steps=10,
    optim="adamw_torch",  # Use Torch AdamW for MPS
    fp16=False,  # Disable mixed precision (MPS works better with bfloat16)
    save_total_limit=2,
    remove_unused_columns=False  # Prevent errors with MPS
)

# --- 5. Split Dataset ---
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)

# --- 6. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"]
)

# --- 7. Train & Save ---
trainer.train()
model.save_pretrained("fine_tuned_tinyllama")  # Saves LoRA adapters
tokenizer.save_pretrained("fine_tuned_tinyllama")

print("✅ Fine-tuning complete! Model and tokenizer saved.")