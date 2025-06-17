# QuickAssist: Intent-Aware Chatbot for Customer Support

## 🧠 Overview

**QuickAssist** is an NLP course project focused on improving chatbot responses in customer support scenarios by conditioning on *explicit intent labels*. We investigate whether models generate more *accurate*, *helpful*, and *appropriate* responses when given both the customer query and its intent.

> ❓ “Why was I charged again?”
> A chatbot that understands this is a *billing issue* may reply more appropriately than one that doesn't.

---

## 🔍 Graphical Abstract

![Graphical Abstract](https://github.com/user-attachments/assets/0d0d9f3c-bef1-4430-939f-4c65823a5468)

---

## 🎯 Project Goals

1. **Intent Conditioning**: Evaluate whether providing intent labels improves chatbot response quality.
2. **Pipeline Comparison**: Compare single-step (no intent) vs. two-step (intent-aware) models.
3. **Model Performance**: Test pretrained vs. fine-tuned T5/BERT models for both tasks.
4. **Dataset Generalization**: Compare model performance across two real-world datasets.
5. **Evaluation**: Use both automatic (BLEU, ROUGE, BERTScore) and LLM-based human-like evaluation (helpfulness, fluency, appropriateness).

---

## 📁 Datasets

We use two datasets with pre-annotated intents:

### 📌 Bitext

* **Size**: 26,872 QA pairs
* **Intents**: 27, grouped into 11 categories
* **Avg query length**: \~8.69 words
* **Most common intents**: `contact_customer_service`, `complaint`, `check_invoice`

### 📌 Customer-Service-for-LLM

* **Size**: 2,700 QA pairs
* **Same 27 intents & 11 categories**
* **Avg query length**: \~8.6 words
* **Most common intents**: `check_invoice`, `switch_account`, `edit_account`

### 🔹 Example Format

```
Input: instruction (user query): "I want to cancel my subscription"  
Output: response (agent reply): "Sure, I can help with that. Please provide your account ID."  
Intent: intent (pre-annotated intent label): cancel_service  
```

---

## 🧠 Model Architecture & Configurations

We implemented both **single-step** and **two-step** architectures to evaluate the effect of intent conditioning on response quality.

### 🧩 Two-Step Pipeline

Intent detection and response generation are decoupled:

#### 🔹 Intent Detection

* **Fine-tuned BERT** (`bert-finetuned`)
  Trained to classify the intent of a customer query.
* **Pretrained T5-base** (`t5-base`)
  Also tested as a zero-shot intent classifier (no fine-tuning).

#### 🔹 Response Generation

* **Pretrained T5-base** (`t5-base`)
  Used as-is to generate responses based on query + intent.
* **Fine-tuned T5** (`t5-finetuned`)
  Trained on our labeled data to generate more accurate, context-aware responses.

> ✅ Best-performing configuration: `two_step_complete_ft`
> Uses **fine-tuned BERT** for intent detection and **fine-tuned T5** for response generation.

---

### 🧩 Single-Step Pipeline

No explicit intent classification — the model generates responses directly from the query.

* **Pretrained T5-base** (`single_step_pretrained`)
  Zero-shot generation, no task-specific tuning.
* **Fine-tuned T5** (`single_step_ft`)
  Trained to map raw customer queries directly to responses.

---

### 🔧 Experiment Configurations

| Name                     | Intent Model   | Response Model | Intent Used? | Fine-tuned?      |
| ------------------------ | -------------- | -------------- | ------------ | ---------------- |
| `single_step_pretrained` | —              | T5-base        | ❌            | No               |
| `single_step_ft`         | —              | T5-finetuned   | ❌            | Yes (T5)         |
| `two_step_baseline`      | Ground-truth   | T5-base        | ✅            | No               |
| `two_step_pretrained`    | T5-base        | T5-base        | ✅            | No               |
| `two_step_partial_ft`    | BERT-finetuned | T5-base        | ✅            | Partial (BERT)   |
| `two_step_complete_ft`   | BERT-finetuned | T5-finetuned   | ✅            | Full (BERT + T5) |

---

## 🧪 Evaluation

We used a combination of automatic, human-like, and intent-level metrics:

* **Automatic**: BERTScore, ROUGE (1/2/L), BLEU
* **Human-like (LLM-based)**: Helpfulness, Fluency, Appropriateness
* **Intent Accuracy**: Calculated for two-step models where the intent is predicted by the model

> ✔ Evaluation results are saved to: `metrics.json`, `intent_accuracy.json`, `human_scores.csv`

---

## 📈 Key Results

| Configuration            | BERTScore (F1) | BLEU | ROUGE-L |
| ------------------------ | -------------- | ---- | ------- |
| `single_step_pretrained` | Baseline       | ↓    | ↓       |
| `two_step_complete_ft`   | ✅ Best         | ↑    | ↑       |

✔ Intent conditioning improves chatbot performance
✔ Fine-tuning significantly boosts BLEU and BERTScore
✔ Two-step (intent-aware) models consistently outperform single-step models

---

## 📂 Repository Structure

This repository includes all three project stages with both **presentations** and **code**:

```bash
QuickAssist/
├── Project_Proposal_Presentation.pdf      # Initial idea and proposal
├── Mid_PPT/
│   ├── requirements.txt                   # Mid-stage dependencies
│   ├── Mid_Project_Presentation.pdf       # Mid-project slides
│   └── README_Mid_PPT.md                  # Mid-stage documentation
├── Final_PPT/
│   ├── requirements.txt                   # Final-stage dependencies
│   ├── Final_Project_Presentation.pdf     # Final slides
│   └── README_Final_PPT.md                # Detailed model, results & evaluation
```

---

## 🧪 Running the Code

To run our code, see the instructions in:

* [Mid-PPT README](https://github.com/shaniKupiec/QuickAssist/blob/main/Mid_PPT/README_Mid_PPT.md)
* [Final-PPT README](https://github.com/shaniKupiec/QuickAssist/blob/main/Final_PPT/README_Final_PPT.md)

---

## 👥 Team

* [Inbal Bolshinsky](https://github.com/InbalBolshinsky)
* [Shani Kupiec](https://github.com/shaniKupiec)
* [Almog Sasson](https://github.com/Almog-Sasson)
* [Nadav Margaliot](https://github.com/NadavMargaliot)

📍 NLP Course, Holon Institute of Technology (HIT)

---

Would you like me to export this as a `README.md` file for GitHub upload?
