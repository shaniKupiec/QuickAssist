"""Response generation handler for different approaches."""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import os

class ResponseHandler:
    def __init__(self, model_config, device="cuda"):
        """Initialize response handler based on model configuration."""
        self.model_config = model_config
        self.device = device
        self.model_type = model_config.get('type')
        self.setup_model()

    def setup_model(self):
        """Setup the appropriate model based on configuration."""
        if self.model_type == "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config['output_dir'] if self.model_config['fine_tuned'] else self.model_config['name']
            ).to(self.device)
        elif self.model_type == "gpt-4":
            # Setup Groq model for response generation
            self.model = Agent[None, str](
                model=GroqModel(
                    self.model_config['model_name'],
                    provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
                ),
                system_prompt="""You are a helpful customer service assistant.
                Provide clear, concise, and helpful responses to customer queries."""
            )

    def format_input(self, query, intent=None):
        """Format input based on whether intent is provided."""
        if intent:
            return f"[Intent: {intent}] User Query: {query}"
        return f"User Query: {query}"

    async def generate_response(self, query, intent=None):
        """Generate response based on the model type."""
        formatted_input = self.format_input(query, intent)
        
        if self.model_type == "t5":
            # Use T5 for generation
            inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        elif self.model_type == "gpt-4":
            # Use GPT-4 for response generation
            result = await self.model.run(formatted_input)
            return result.output.strip()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, train_data, eval_data=None):
        """Train the response model if it's trainable."""
        if self.model_type not in ["t5"]:
            raise ValueError(f"Model type {self.model_type} is not trainable")
        
        if self.model_type == "t5":
            # Training configuration
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.model_config['output_dir'],
                per_device_train_batch_size=16,
                num_train_epochs=5,
                learning_rate=5e-5,
                weight_decay=0.01,
                logging_steps=50,
                save_strategy="epoch",
                evaluation_strategy="epoch" if eval_data is not None else "no",
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
                train_dataset=train_data,
                eval_dataset=eval_data,
                data_collator=data_collator,
            )

            # Start training
            trainer.train() 