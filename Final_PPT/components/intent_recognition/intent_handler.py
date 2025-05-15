"""Intent recognition handler for different approaches."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import os

class IntentHandler:
    def __init__(self, model_config, device="cuda"):
        """Initialize intent handler based on model configuration."""
        self.model_config = model_config
        self.device = device
        self.model_type = model_config.get('type')
        self.setup_model()

    def setup_model(self):
        """Setup the appropriate model based on configuration."""
        if self.model_type == "ground_truth":
            # No model needed for ground truth
            pass
        elif self.model_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config['output_dir'] if self.model_config['fine_tuned'] else self.model_config['name']
            ).to(self.device)
        elif self.model_type == "gpt-4":
            # Setup Groq model for intent recognition
            self.model = Agent[None, str](
                model=GroqModel(
                    self.model_config['model_name'],
                    provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
                ),
                system_prompt="""You are an intent classifier for customer service queries.
                Return only the intent label that best describes the query.
                Example intents: Account_Access, Technical_Support, Billing_Inquiry, etc."""
            )

    async def get_intent(self, query, ground_truth_intent=None):
        """Get intent based on the model type."""
        if self.model_type == "ground_truth":
            if ground_truth_intent is None:
                raise ValueError("Ground truth intent required but not provided")
            return ground_truth_intent
        
        elif self.model_type == "bert":
            # Use BERT for classification
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_intent = self.model.config.id2label[outputs.logits.argmax().item()]
            return predicted_intent
        
        elif self.model_type == "gpt-4":
            # Use GPT-4 for intent recognition
            result = await self.model.run(query)
            return result.output.strip()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, train_data, eval_data=None):
        """Train the intent model if it's trainable."""
        if self.model_type not in ["bert"]:
            raise ValueError(f"Model type {self.model_type} is not trainable")
        
        # Training logic for BERT model
        if self.model_type == "bert":
            # Implementation for BERT fine-tuning
            # This would include dataset preparation, training loop, etc.
            pass 