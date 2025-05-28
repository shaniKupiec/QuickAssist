"""Experiment runner for executing different experimental configurations."""

import os
import yaml
import asyncio
import torch
from typing import Dict, Any
import pandas as pd

# Update imports to use correct package paths
from components.data_loader import load_and_prepare_dataset
from components.intent_recognition import IntentHandler
from components.response_generation import ResponseHandler

from components.evaluation import Evaluator 

class ExperimentRunner:
    def __init__(self, config_dir="config"):
        """Initialize experiment runner with configuration directory."""
        self.config_dir = config_dir
        self.load_configs()
        
    def load_configs(self):
        """Load all configuration files."""
        # Load main configuration
        with open(os.path.join(self.config_dir, "config.yaml"), "r") as f:
            self.main_config = yaml.safe_load(f)
            self.main_config['default_device'] = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Load model configurations
        with open(os.path.join(self.config_dir, "models.yaml"), "r") as f:
            self.model_config = yaml.safe_load(f)
            
        # Load experiment configurations
        with open(os.path.join(self.config_dir, "experiments.yaml"), "r") as f:
            self.experiment_config = yaml.safe_load(f)
            
        # Load dataset configurations
        with open(os.path.join(self.config_dir, "datasets.yaml"), "r") as f:
            self.dataset_config = yaml.safe_load(f)

    async def run_experiment(self, experiment_name: str, dataset_name: str):
        """Run a single experiment with specified dataset."""
        print(f"\nRunning experiment: {experiment_name} with dataset: {dataset_name}")
        
        # Get experiment configuration
        experiment = next(exp for exp in self.experiment_config['experiments'] 
                        if exp['name'] == experiment_name)
        
        # Load dataset
        needs_intent = experiment['requirements']['needs_intent']
        train_data, test_data = load_and_prepare_dataset(
            dataset_name=dataset_name,
            needs_intent=needs_intent,
            dataset_config=self.dataset_config
        )

        # Setup models based on experiment type
        if experiment['type'] == "single_step":
            # Single-step: Only response generation
            response_config = self.model_config['response_models'][experiment['response_model']]
            response_handler = ResponseHandler(response_config, main_config=self.main_config)

            # If model needs fine-tuning, train it
            if response_config.get('fine_tuned', False):
                response_handler.train(train_data)
            
            # Generate responses for test set
            results = []
            for _, row in test_data.iterrows():
                response = await response_handler.generate_response(row['input'])
                results.append({
                    'query': row['input'],
                    'generated_response': response,
                    'reference_response': row['output']
                })
            
                
        else:  # Two-step
            # Step 1: Intent Recognition
            all_data = pd.concat([train_data, test_data])
            intents = {}
            
            if experiment['intent_model'] == 'ground_truth':
                # Skip intent model setup and training for ground truth
                print("Using ground truth intents from dataset")
                for _, row in all_data.iterrows():
                    intent = row.get('intent')
                    intents[row['input']] = intent
            else: 
                # Only setup and train intent model if not using ground truth
                intent_config = self.model_config['intent_models'][experiment['intent_model']]
                intent_handler = IntentHandler(intent_config, main_config=self.main_config)
                
                # Train intent model if needed
                if intent_config.get('fine_tuned', False):
                    intent_handler.train(train_data)
                
                # Get intents using the model
                for _, row in all_data.iterrows():
                    intent = await intent_handler.get_intent(row['input'])
                    intents[row['input']] = intent

            # Prepare training data with intents
            train_data_with_intents = []
            for _, row in train_data.iterrows():
                train_data_with_intents.append({
                    'input': row['input'],
                    'intent': intents[row['input']],
                    'output': row['output']
                })
            
            # Convert list of dictionaries to DataFrame
            train_data_with_intents = pd.DataFrame(train_data_with_intents)

            # Step 2: Response Generation
            response_config = self.model_config['response_models'][experiment['response_model']]
            response_handler = ResponseHandler(response_config, main_config=self.main_config)
            
            # Train response model if needed with intent information
            if response_config.get('fine_tuned', False):
                response_handler.train(train_data_with_intents)
            
            # Generate responses for test set
            results = []
            for _, row in test_data.iterrows():
                # Get stored intent
                intent = intents[row['input']]
                
                # Generate response using the intent
                response = await response_handler.generate_response(row['input'], intent)
                
                results.append({
                    'query': row['input'],
                    'intent': intent,
                    'generated_response': response,
                    'reference_response': row['output']
                })
        
        # Evaluate results
        evaluator = Evaluator(
            use_human_eval=True,
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=self.main_config["human_eval_model"],
            main_config=self.main_config
        )

        metrics = await evaluator.evaluate(results)
                
        return {
            'experiment': experiment_name,
            'dataset': dataset_name,
            'metrics': metrics,
            'results': results
        }

    async def run_all_experiments(self):
        """Run all experiments with all compatible datasets."""
        results = []
        for experiment in self.experiment_config['experiments']:
            for dataset in self.experiment_config['settings']['available_datasets']:
                try:
                    result = await self.run_experiment(experiment['name'], dataset)
                    results.append(result)
                except Exception as e:
                    print(f"Error running {experiment['name']} with {dataset}: {str(e)}")
        
        return results 
