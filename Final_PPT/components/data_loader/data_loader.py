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

def preprocess_bitod(df: pd.DataFrame) -> pd.DataFrame:
    """Specific preprocessing for BiToD dataset.
    
    Args:
        df: DataFrame containing BiToD data
    Returns:
        Preprocessed DataFrame
    """
    # Convert to lowercase
    df['input'] = df['input'].str.lower()
    df['output'] = df['output'].str.lower()
    
    # Remove extra whitespace
    df['input'] = df['input'].str.strip()
    df['output'] = df['output'].str.strip()
    
    # Remove special characters but keep basic punctuation
    df['input'] = df['input'].apply(lambda x: re.sub(r'[^a-z0-9\s.,!?]', '', x))
    df['output'] = df['output'].apply(lambda x: re.sub(r'[^a-z0-9\s.,!?]', '', x))
    
    # Remove multiple spaces
    df['input'] = df['input'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['output'] = df['output'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def load_and_prepare_dataset(dataset_name, config_path="config/datasets.yaml"):
    """Load and prepare a dataset based on configuration.
    
    Args:
        dataset_name: Name of the dataset (e.g., "bitext", "bitod")
        config_path: Path to the datasets configuration file
    """
    # Load dataset configuration
    config = load_config(config_path)
    dataset_config = config['datasets'][dataset_name]
    
    # Load dataset using HuggingFace datasets
    data = load_dataset(dataset_config['name'], split=dataset_config['split'])
    
    # Convert to DataFrame and map fields according to config
    fields = dataset_config['fields']
    df = pd.DataFrame({
        'input': data[fields['input']],
        'output': data[fields['output']]
    })
    
    # Add intent if available
    if 'intent' in fields:
        df['intent'] = data[fields['intent']]
    
    # Drop any rows with missing values
    df = df.dropna()
    
    # Apply dataset-specific preprocessing
    if dataset_name == "bitod":
        df = preprocess_bitod(df)
    
    # Sample fraction if specified
    if 'sample_frac' in dataset_config:
        df = df.sample(frac=dataset_config['sample_frac'], random_state=42)
    
    # Split into train/test
    test_size = dataset_config.get('test_size', 0.2)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_available_datasets(config_path="config/datasets.yaml"):
    """Get list of available datasets from config."""
    config = load_config(config_path)
    return list(config['datasets'].keys()) 