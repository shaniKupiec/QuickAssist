# QuickAssist: Intent-Aware Chatbot for Customer Support

## 📌 Project Overview

**QuickAssist** is an NLP project aimed at enhancing customer support chatbots using **intent conditioning**. We explore how including an **explicit intent label** in the input affects the **quality of generated responses** from large language models (LLMs).

---

## 🎯 Objective

To compare chatbot response generation **with** and **without** intent information:

- **Variant A:** Input = Customer query  
- **Variant B:** Input = Customer query + intent label  

By analyzing both, we assess whether **intent-aware input** leads to more helpful, fluent, and contextually appropriate responses.

---

## 💡 Key Features

- Uses **pre-annotated customer support datasets** with labeled intents
- Employs **LLMs** for response generation
- Evaluates response quality through:
  - **Automatic metrics** (e.g., BERTScore)
  - **LLM-based human-like judgments** (GPT-4 scoring)

---

## 🧠 NLP Tasks Involved

- **Text Generation**
- **Intent Recognition**
- **Contextual Understanding**

---

## 🗃️ Datasets

We use two public datasets:

1. [**Customer-Service-for-LLM**](https://huggingface.co/datasets/pranav301102/customer-service-for-llm/viewer/default/train)
   - ~2,700 QA pairs
   - 27 intent types
   - 11 categories

2. [**Bitext Customer Support Dataset**](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/viewer/default/train?row=1&views%5B%5D=train)
   - ~26,872 QA pairs
   - 27 intent types
   - 10 categories
   - 30 entity/slot types

Each sample includes customer queries, intents, and slot information.

---

## ⚙️ Methodology

1. **Preprocess Data:** Extract and organize query–intent–response triples  
2. **Generate Responses:**
   - Variant A: Feed query alone
   - Variant B: Feed query + intent
3. **Evaluate Outputs:**
   - **BERTScore** for semantic similarity to reference
   - **GPT-4-based scoring** on Helpfulness, Fluency, and Appropriateness

---

## 📊 Evaluation

- **BERTScore:** Captures semantic similarity between generated and reference responses using contextual embeddings.
- **GPT-4 Evaluation:**
  - **Helpfulness:** Did the answer address the query?
  - **Fluency:** Is it clear and grammatically correct?
  - **Appropriateness:** Is the tone and content suitable?

---

## 🚀 Expected Outcomes

- Empirical comparison of chatbot performance **with vs. without intent labels**
- Insights into the importance of **intent conditioning** for improving LLM response quality in customer support scenarios

---

## 📎 Authors

- [Inbal Bolshinsky](https://github.com/InbalBolshinsky), [Shani Kupiec](https://github.com/shaniKupiec), [Almog Sasson](https://github.com/Almog-Sasson) and [Nadav Margaliot](https://github.com/NadavMargaliot)
- HIT: NLP course

