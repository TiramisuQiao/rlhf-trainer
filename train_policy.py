import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer
from deepspeed import HfDeepSpeedConfig
import yaml
import argparse

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DeepSpeed
    ds_config = config['deepspeed']
    dschf = HfDeepSpeedConfig(ds_config)
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    dataset = load_dataset(config['data']['path'])
    
    # Tokenization function
    def tokenize_function(examples):
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        return {
            "prompt": prompts,
            "chosen": [p + c for p, c in zip(prompts, chosen)],
            "rejected": [p + r for p, r in zip(prompts, rejected)],
        }
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = {
        "output_dir": config['training']['output_dir'],
        "per_device_train_batch_size": config['training']['batch_size'],
        "gradient_accumulation_steps": config['training']['grad_accum_steps'],
        "learning_rate": config['training']['lr'],
        "num_train_epochs": config['training']['epochs'],
        "save_steps": config['training']['save_steps'],
        "logging_steps": config['training']['log_steps'],
        "deepspeed": ds_config,
        "bf16": config['training']['mixed_precision'],
        "report_to": config['training']['report_to']
    }
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=config['model']['name'],
        ref_model=config['model']['ref_model'],
        beta=config['training']['beta'],
        loss_type=config['training']['loss_type'],
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/policy_config.yaml")
    args = parser.parse_args()
    main(args)