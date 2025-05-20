"""Data loading and preprocessing utilities."""

import os
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def needs_intent_for_experiment(experiment_name, config_path):
    """Return whether the experiment requires the intent field."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    experiments = config.get("experiments", [])
    for exp in experiments:
        if exp.get("name") == experiment_name:
            return exp.get("requirements", {}).get("needs_intent", False)
    raise ValueError(f"Experiment '{experiment_name}' not found in config.")

def load_and_prepare_dataset(dataset_name, experiment_name,
                             dataset_config_path="config/datasets.yaml",
                             experiment_config_path="config/experiments.yaml"):
    """
    Load and prepare a dataset based on configuration and experiment intent needs.

    Args:
        dataset_name: Name of the dataset (e.g., "bitext", "bitod")
        experiment_name: Name of the experiment (e.g., "two_step_complete_ft")
        dataset_config_path: Path to the dataset config YAML
        experiment_config_path: Path to the experiment config YAML
    """
    # Determine if intent is required by the experiment
    needs_intent = needs_intent_for_experiment(experiment_name, experiment_config_path)

    # Load dataset config
    config = load_config(dataset_config_path)
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    dataset_config = config['datasets'][dataset_name]
    data = load_dataset(dataset_config['name'], split=dataset_config['split'])
    fields = dataset_config['fields']

    # Required fields
    columns = {
        'input': data[fields['input']],
        'output': data[fields['output']]
    }

    # Optional intent field
    if needs_intent:
        if 'intent' not in fields or fields['intent'] not in data.column_names:
            raise ValueError(f"'intent' required by experiment but not found in dataset '{dataset_name}'")
        columns['intent'] = data[fields['intent']]

    df = pd.DataFrame(columns)

    # Drop missing values
    df = df.dropna()

    # Preprocess
    df = preprocess_dataset(df, dataset_name)

    # Optional sampling
    if 'sample_frac' in dataset_config:
        df = df.sample(frac=dataset_config['sample_frac'], random_state=42)

    # Train/test split
    test_size = dataset_config.get('test_size', 0.2)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def preprocess_dataset(df, dataset_name):
    """Apply dataset-specific preprocessing logic if needed."""
    if dataset_name == "bitod":
        df['input'] = df['input'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    elif dataset_name == "bitext":
        df['input'] = df['input'].apply(lambda x: x.strip())
    return df

def get_available_datasets(config_path="config/datasets.yaml"):
    """Get list of available datasets from config."""
    config = load_config(config_path)
    return list(config['datasets'].keys())
