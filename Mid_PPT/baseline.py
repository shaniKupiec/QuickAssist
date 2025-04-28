"""# 📚 Imports"""

import os
import time
import torch
import pandas as pd
import nest_asyncio
import asyncio
import time
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from bert_score import score
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from dotenv import load_dotenv

"""# 🔑 API Key (can be set via environment variables)"""

# Load environment variables from the .env file
load_dotenv()

# Access environment variables using os.getenv() or os.environ
groq_api_key = os.getenv("GROQ_API_KEY", "")

"""# ✅ Load & preprocess dataset

"""

# Define the directory where the files will be saved
output_dir = "C:\\Shani\\baselineOutputs"
# Make the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def load_and_prepare_data():
    data = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(data)[['instruction', 'response']].dropna()

    # Reduce dataset size by sampling 20%
    df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


train_df, test_df = load_and_prepare_data()

"""# 🔧 Tokenization

"""

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(examples):
    inputs = tokenizer(examples["instruction"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = Dataset.from_pandas(train_df).map(preprocess, batched=True)

"""# 🏋️ Training config

"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

"""# ✅ Train model"""

trainer.train()

"""# 🧪 Response Generation"""

MAX_NEW_TOKENS = 128
MAX_EVAL_SAMPLES = 50

def generate_responses(df, tokenizer, model):
    sample = df.iloc[:MAX_EVAL_SAMPLES].copy()
    dataset = Dataset.from_pandas(sample)
    inputs = tokenizer(dataset["instruction"], return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    sample["generated_response"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return sample

sample_with_predictions = generate_responses(test_df, tokenizer, model)
references = sample_with_predictions["response"].tolist()
generated = sample_with_predictions["generated_response"].tolist()

"""# 📊 BERTScore Evaluation

"""

P, R, F1 = score(generated, references, lang="en", verbose=True)
print(f"\n📈 BERTScore (F1): {F1.mean().item():.4f}")

"""# 🧠 Human-like Eval (Groq)"""

nest_asyncio.apply()

"""Evaluation schema"""

class ChatEvaluation(BaseModel):
    Helpfulness: int
    Fluency: int
    Appropriateness: int

"""System prompt"""

system_instruction = """
You are evaluating a chatbot's reply to a customer.

Please rate the response on the following from 1 (poor) to 5 (excellent).
Return only a JSON object with the fields:
- Helpfulness
- Fluency
- Appropriateness
"""

"""Setup Groq agent"""

groq_model = GroqModel(
    "llama-3.3-70b-versatile",
    provider=GroqProvider(api_key=groq_api_key),
)

agent = Agent[None, ChatEvaluation](
    model=groq_model,
    system_prompt=system_instruction,
    result_type=ChatEvaluation,
)

"""Helper functions"""

# Prompt for each row
def create_user_prompt(query, response):
    return f"""Customer Query:\n{query}\n\nChatbot Response:\n{response}"""

# Retry logic
async def evaluate_single(index, instruction, response, max_retries=5, base_delay=2):
    prompt = create_user_prompt(instruction, response)

    for attempt in range(max_retries):
        try:
            print(f"Evaluating {index + 1}/{MAX_EVAL_SAMPLES}... (Attempt {attempt + 1})")
            result = await agent.run(prompt)
            return result.output

        except Exception as e:
            delay = base_delay * (2 ** attempt)
            print(f"⚠️ Rate limit hit at {index + 1}, retrying in {delay} seconds...")
            await asyncio.sleep(delay)

    print(f"🚫 Skipping {index + 1} after {max_retries} failed attempts.")

# Main loop for evaluation
async def evaluate_all(df):
    tasks = [
        evaluate_single(i, row["instruction"], row["generated_response"])
        for i, row in df.iloc[:MAX_EVAL_SAMPLES].iterrows()
    ]
    results = []
    for task in tasks:
        result = await task
        results.append(result)
        await asyncio.sleep(1.5)  # Small delay to ease pressure on API
    return results

"""Run Human Evaluation & Add Results to DataFrame"""

# After Human Evaluation & Adding Results to DataFrame

structured_scores = asyncio.run(evaluate_all(sample_with_predictions))

for metric in ["Helpfulness", "Fluency", "Appropriateness"]:
    sample_with_predictions.loc[:MAX_EVAL_SAMPLES - 1, metric] = [
        res[metric] if isinstance(res, dict) else getattr(res, metric)
        for res in structured_scores
    ]

sample_with_predictions["average_score"] = sample_with_predictions[["Helpfulness", "Fluency", "Appropriateness"]].mean(axis=1)

# Example of saving the whole DataFrame to a CSV file in the new directory
print("\n📊 Metric Averages:")
print(sample_with_predictions[["Helpfulness", "Fluency", "Appropriateness"]].mean())

# Save the metric averages to the new directory
metric_file_path = os.path.join(output_dir, "metric_averages.csv")
sample_with_predictions[["Helpfulness", "Fluency", "Appropriateness"]].mean().to_csv(metric_file_path, index=True)
print(f"✅ Metric averages saved to '{metric_file_path}'")

print("\n🧮 Example of Row-wise Averages:")
# Save row-wise averages to the new directory
row_wise_file_path = os.path.join(output_dir, "row_wise_averages.csv")
sample_with_predictions["average_score"].head().to_csv(row_wise_file_path, index=False)
print(f"✅ Row-wise averages saved to '{row_wise_file_path}'")

# Full results
full_results_file_path = os.path.join(output_dir, "full_results.csv")
sample_with_predictions.to_csv(full_results_file_path, index=False)
print(f"✅ Full results saved to '{full_results_file_path}'")
