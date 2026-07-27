"""
QLoRA fine-tune of Llama 3.1 8B Instruct on the judge dataset. Built to run on a
single Colab GPU (T4 works for 8B in 4-bit).

Steps to use on Colab:
    !pip install -q transformers peft bitsandbytes accelerate datasets trl -U
    Upload finetuning/dataset/judge_train.jsonl (run prepare_dataset.py locally first)
    Log in with huggingface_hub.login() (needed for the gated Llama 3.1 base model)
    Run this script

Output: a LoRA adapter, saved locally then pushed to the HF Hub.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "judge_train.jsonl"  # flat path — matches Colab's upload location
OUTPUT_DIR = "checkpoints/llama3-8b-code-judge"
HUB_REPO = "jahnavi0803/llama3-8b-code-judge"  # your actual Hugging Face username

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def format_example(example):
    return {"text": f"{example['prompt']}\n\n{example['completion']}{tokenizer.eos_token}"}


dataset = dataset.map(format_example)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    dataset_text_field="text",
    max_length=1024,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}. Push to {HUB_REPO} when ready.")