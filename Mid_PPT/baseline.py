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

nest_asyncio.apply()

# 🔑 API Key (can be set via environment variables)

# Load environment variables from the .env file
load_dotenv()

# Access environment variables using os.getenv() or os.environ
groq_api_keys = os.getenv("GROQ_API_KEYS", "").split(",")

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
MAX_EVAL_SAMPLES = 500

def generate_responses(df, tokenizer, model, batch_size=8):
    sample = df.iloc[:MAX_EVAL_SAMPLES].copy()
    dataset = Dataset.from_pandas(sample)
    generated_responses = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        inputs = tokenizer(batch["instruction"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_responses.extend(decoded)

    sample["generated_response"] = generated_responses
    return sample


sample_with_predictions = generate_responses(test_df, tokenizer, model)
references = sample_with_predictions["response"].tolist()
generated = sample_with_predictions["generated_response"].tolist()

"""# 📊 BERTScore Evaluation

"""

P, R, F1 = score(generated, references, lang="en", verbose=True)
print(f"\n📈 BERTScore (F1): {F1.mean().item():.4f}")

"""# 🧠 Human-like Eval (Groq)"""


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

# Define your models here

groq_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

# Make sure the lengths of groq_api_keys and groq_models match
if len(groq_api_keys) != len(groq_models):
    raise ValueError("Number of API keys and models must match.")

groq_credentials = list(zip(groq_api_keys, groq_models))
current_credential_index = 0

def create_agent(api_key, model_name):
    return Agent[None, ChatEvaluation](
        model=GroqModel(model_name, provider=GroqProvider(api_key=api_key)),
        system_prompt=system_instruction,
        result_type=ChatEvaluation,
    )

# Initialize agent with the first set of credentials
agent = create_agent(*groq_credentials[current_credential_index])

"""Helper functions"""

evaluation_stopped = False  # Global flag to stop evaluations

# Prompt for each row
def create_user_prompt(query, response):
    return f"""Customer Query:\n{query}\n\nChatbot Response:\n{response}"""


# Retry logic
async def evaluate_single(index, instruction, response, max_retries=6, base_delay=2):
    global agent, current_credential_index, evaluation_stopped
    prompt = create_user_prompt(instruction, response)

    for attempt in range(max_retries):
        try:
            print(f"Evaluating {index + 1}/{MAX_EVAL_SAMPLES}... (Attempt {attempt + 1})")
            result = await agent.run(prompt)
            return result.output

        except Exception as e:
            delay = base_delay * (2 ** attempt)
            print(f"⚠️ Error: {str(e)} — retrying in {delay} seconds...")

            if attempt == max_retries - 1:
                current_credential_index += 1

                if current_credential_index >= len(groq_credentials):
                    print("❌ All API keys and models have been exhausted. Halting evaluation.")
                    evaluation_stopped = True
                    return None

                next_key, next_model = groq_credentials[current_credential_index]
                print(f"🔁 Switching to next API key and model: {next_model} (index {current_credential_index})")
                agent = create_agent(next_key, next_model)

            await asyncio.sleep(delay)

    print(f"🚫 Skipping {index + 1} after {max_retries} failed attempts.")
    return None



async def evaluate_all(df):
    results = []
    for i, row in df.iloc[:MAX_EVAL_SAMPLES].iterrows():
        if evaluation_stopped:
            print("🛑 Stopping evaluation early due to exhausted credentials.")
            break
        result = await evaluate_single(i, row["instruction"], row["generated_response"])
        results.append(result)
        await asyncio.sleep(1.5)
    return results



"""Run Human Evaluation & Add Results to DataFrame"""

# After Human Evaluation & Adding Results to DataFrame

structured_scores = asyncio.run(evaluate_all(sample_with_predictions))

# Drop remaining unevaluated rows
valid_scores = [res for res in structured_scores if res is not None]
valid_rows = sample_with_predictions.iloc[:len(valid_scores)].copy()

for metric in ["Helpfulness", "Fluency", "Appropriateness"]:
    valid_rows[metric] = [
        res[metric] if isinstance(res, dict) else getattr(res, metric)
        for res in valid_scores
    ]

valid_rows["average_score"] = valid_rows[["Helpfulness", "Fluency", "Appropriateness"]].mean(axis=1)

# Save outputs
print("\n📊 Metric Averages:")
print(valid_rows[["Helpfulness", "Fluency", "Appropriateness"]].mean())

metric_file_path = os.path.join(output_dir, "metric_averages.csv")
valid_rows[["Helpfulness", "Fluency", "Appropriateness"]].mean().to_csv(metric_file_path, index=True)
print(f"✅ Metric averages saved to '{metric_file_path}'")

row_wise_file_path = os.path.join(output_dir, "row_wise_averages.csv")
valid_rows["average_score"].to_csv(row_wise_file_path, index=False)
print(f"✅ Row-wise averages saved to '{row_wise_file_path}'")

full_results_file_path = os.path.join(output_dir, "full_results.csv")
valid_rows.to_csv(full_results_file_path, index=False)
print(f"✅ Full results saved to '{full_results_file_path}'")
