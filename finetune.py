
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# Load dataset
model_name = "Qwen/Qwen2.5-3B-Instruct"
raw_data = load_dataset("json", data_files="dataset\oporichita.json")
print(raw_data["train"][0])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

def preprocess(sample):
    text = f"### Instruction: {sample['prompt']} ### Response: {sample['completion']} ###"
    tokenized = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["attention_mask"] == 0] = -100
    return {k: v.squeeze(0) for k, v in tokenized.items()}

# Apply preprocessing
data = raw_data.map(preprocess, remove_columns=["prompt", "completion"])

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Prepare model for training with LoRA
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"],
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    use_rslora=True  # Use rank-stabilized LoRA for stability
)

model = get_peft_model(model, lora_config)

# Ensure trainable parameters require gradients
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./bangla_qwen",
    num_train_epochs=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    warmup_steps=50,
    weight_decay=0.01,
    gradient_checkpointing=False  # Disabled to avoid gradient issues
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    processing_class=tokenizer  # Updated to avoid FutureWarning
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./bangla_qwen")
tokenizer.save_pretrained("./bangla_qwen")
