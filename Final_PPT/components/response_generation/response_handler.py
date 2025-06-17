import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from pydantic_ai import Agent
from datasets import Dataset

class ResponseHandler:
    def __init__(self, model_config, main_config):
        self.model_config = model_config
        self.main_config = main_config
        self.device = main_config['default_device']
        self.model_type = model_config.get('type')
        self.setup_model()

    def setup_model(self):
        if self.model_type == "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config['name']).to(self.device)
    

    def preprocess(self, examples):
        # Format input based on whether intent is available
        if "intent" in examples:
            formatted_input = [f"[Intent: {i}] User Query: {q}" for i, q in zip(examples["intent"], examples["input"])]
        else:
            formatted_input = [f"User Query: {q}" for q in examples["input"]]
            
        inputs = self.tokenizer(formatted_input, padding="max_length", truncation=True, max_length=self.main_config["max_length"])
        labels = self.tokenizer(examples["output"], padding="max_length", truncation=True, max_length=self.main_config["max_length"])
        inputs["labels"] = labels["input_ids"]
        return inputs

    def format_input(self, query, intent=None):
        if intent:
            return f"[Intent: {intent}] User Query: {query}"
        return f"User Query: {query}"

    async def generate_response(self, query, intent=None):
        formatted_input = self.format_input(query, intent)
        
        if self.model_type == "t5":
            inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

    def train(self, train_data, eval_data=None):
        if self.model_type not in ["t5"]:
            raise ValueError(f"Model type {self.model_type} is not trainable")
        
        if self.model_type == "t5":
            print("Starting model training...")
            # Convert DataFrame to Dataset and preprocess
            train_dataset = Dataset.from_pandas(train_data).map(self.preprocess, batched=True)
            eval_dataset = None
            if eval_data is not None:
                eval_dataset = Dataset.from_pandas(eval_data).map(self.preprocess, batched=True)

            # Training configuration
            training_args = Seq2SeqTrainingArguments(
                output_dir="./results",  # Simple output directory like in baseline.py
                per_device_train_batch_size=self.main_config["batch_size"],
                num_train_epochs=self.main_config["epochs"],
                learning_rate=5e-5,
                weight_decay=self.main_config["weight_decay"],
                logging_steps=50,
                save_strategy="epoch",
                report_to="none"
            )

            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )

            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            # Start training
            trainer.train()
            print("Training completed successfully!") 