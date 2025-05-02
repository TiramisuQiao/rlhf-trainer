from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        prompt = prompt.strip()
        chosen = chosen.strip()
        rejected = rejected.strip()
        
        new_examples["prompt"].append(prompt)
        new_examples["chosen"].append(chosen)
        new_examples["rejected"].append(rejected)

    return new_examples

dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
dataset = dataset.map(
    lambda x: {
        "prompt": [c.split("Assistant:")[0].replace("Human:", "").strip() for c in x["chosen"]],
        "chosen": [c.split("Assistant:")[-1].strip() for c in x["chosen"]],
        "rejected": [r.split("Assistant:")[-1].strip() for r in x["rejected"]]
    },
    batched=True,
    remove_columns=dataset["train"].column_names
)

training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    max_length=1024,
    max_prompt_length=512,
    gradient_accumulation_steps=4,
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False 
)
trainer = DPOTrainer(
    model=model,
    ref_model=None,  
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("dpo_final_model")