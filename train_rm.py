import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer
from transformers import DataCollatorWithPadding
from deepspeed import DeepSpeedConfig
import yaml
import argparse

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DeepSpeed
    ds_config = config['deepspeed']
    dschf = DeepSpeedConfig(ds_config)
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    dataset = load_dataset(config['data']['path'])
    
    # Tokenization function
    def tokenize_function(examples):
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        split = examples["split"] 
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        return {
            "text_chosen": chosen_texts,
            "text_rejected": rejected_texts,
            "split": split  
        }
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(tokenized_datasets.keys())  
    # Create reward model with custom head
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'], 
        num_labels=1,
        trust_remote_code=True
    )
    
    # Add 2-layer MLP head
    model.score = torch.nn.Sequential(
        torch.nn.Linear(4096, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1)
    )
    
    # Training arguments
    training_args = {
        "output_dir": config['training']['output_dir'],
        "per_device_train_batch_size": config['training']['batch_size'],
        "gradient_accumulation_steps": config['training']['grad_accum_steps'],
        "learning_rate": config['training']['lr'],
        "num_train_epochs": config['training']['epochs'],
        "save_steps": config['training']['save_steps'],
        "logging_steps": config['training']['log_steps'],
        "evaluation_strategy": "steps",
        "eval_steps": config['training']['eval_steps'],
        "deepspeed": ds_config,
        "bf16": config['training']['mixed_precision'],
        "report_to": config['training']['report_to']
    }
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        padding="max_length",
        return_tensors="pt"
    )
    def custom_data_collator(features):
        chosen_features = [{"input_ids": f["input_ids_chosen"], "attention_mask": f["attention_mask_chosen"]} for f in features]
        rejected_features = [{"input_ids": f["input_ids_rejected"], "attention_mask": f["attention_mask_rejected"]} for f in features]
    
        batch_chosen = data_collator(chosen_features)
        batch_rejected = data_collator(rejected_features)
    
        return {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
        }
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=custom_data_collator
    )
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rm_config.yaml")
    args = parser.parse_args()
    main(args)