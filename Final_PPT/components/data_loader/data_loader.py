"""Data loading and preprocessing utilities."""

import os
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re


def load_and_prepare_dataset(dataset_name, needs_intent, dataset_config):
    """
    Load and prepare a dataset based on configuration and experiment intent needs.

    Args:
        dataset_name (str): Name of the dataset (e.g., "bitext", "bitod")
        needs_intent (bool): Whether the experiment requires intent information in the dataset
        dataset_config (dict): Configuration dictionary containing dataset settings

    Returns:
        tuple: A tuple containing:
            - train_df (pandas.DataFrame): Training dataset with columns ['input', 'output'] and optionally ['intent']
            - test_df (pandas.DataFrame): Test dataset with columns ['input', 'output'] and optionally ['intent']

    Raises:
        ValueError: If dataset name is not found in config or if intent is required but not available
    """

    if dataset_name not in dataset_config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    dataset_config = dataset_config['datasets'][dataset_name]
    data = load_dataset(dataset_config['name'], split=dataset_config['split'])
    fields = dataset_config['fields']

    # Required fields
    columns = {
        'input': data[fields['input']],
        'output': data[fields['output']]
    }

    # Optional intent field
    if needs_intent:
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

