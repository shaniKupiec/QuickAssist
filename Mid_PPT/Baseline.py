# 🐍 Import Libraries
import os
import time
import torch
import pandas as pd
import asyncio
import httpx
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from bert_score import score
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# 🔑 API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# 🖥️ Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Load & preprocess dataset
def load_and_prepare_data():
    data = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(data)[['instruction', 'response']].dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

train_df, test_df = load_and_prepare_data()

# 🔧 Tokenization
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) 

def preprocess(examples):
    inputs = tokenizer(examples["instruction"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = Dataset.from_pandas(train_df).map(preprocess, batched=True)

# 🏋️ Training config
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    dataloader_pin_memory=torch.cuda.is_available()  # ✅ Pin memory only if GPU exists
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ✅ Train model
trainer.train()

# 🧪 Response Generation
MAX_NEW_TOKENS = 128
MAX_EVAL_SAMPLES = 50

async def generate_response_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GROQ_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

async def generate_all_groq(prompts):
    tasks = [generate_response_groq(p) for p in prompts]
    return await asyncio.gather(*tasks)

def generate_responses(df, tokenizer, model, use_groq=False):
    sample = df.iloc[:MAX_EVAL_SAMPLES].copy()
    if use_groq:
        prompts = sample["instruction"].tolist()
        responses = asyncio.run(generate_all_groq(prompts))
        sample["generated_response"] = responses
    else:
        dataset = Dataset.from_pandas(sample)
        inputs = tokenizer(dataset["instruction"], return_tensors="pt", padding=True, truncation=True).to(device)
        model = model.to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        sample["generated_response"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return sample

# 🔥 Choose Local or Groq
USE_GROQ = False  # set to True if you want to offload generation to Groq

sample_with_predictions = generate_responses(test_df, tokenizer, model, use_groq=USE_GROQ)
references = sample_with_predictions["response"].tolist()
generated = sample_with_predictions["generated_response"].tolist()

# 📊 BERTScore Evaluation
P, R, F1 = score(generated, references, lang="en", verbose=True)
print(f"\n📈 BERTScore (F1): {F1.mean().item():.4f}")
