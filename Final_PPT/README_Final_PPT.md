# Final\_PPT Project

This folder contains the final version of our code and results for the **QuickAssist** project.

## 📂 Files Overview

| File                             | Description                                                |
| :------------------------------- | :--------------------------------------------------------- |
| `requirements.txt`               | Final dependency list                                      |
| `Final_Project_Presentation.pdf` | Final slide deck summarizing the project                   |
| `.env`                           | Environment variables file (you must create this manually) |
| `README_Final_PPT.md`            | This file — full guide for setup and execution             |

---

## 🛠️ Code Structure

```yaml
├── data_loader/               # Data loading and formatting
├── intent_recognition/        # Intent detection using BERT/T5
├── response_generation/       # T5-based response generator
├── evaluation/                # Metric calculations (auto + human)
├── experiment_runner/         # Runs all pipelines based on config
├── main.py                    # Entry point script
```

---

# 🚀 How to Set Up and Run the Final Code

## 1. Python Installation

```bash
python --version
```

✅ Use Python 3.9–3.11

---

## 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
```

---

## 3. (Optional) Check Your GPU

```bash
nvidia-smi
```

✅ Ensure CUDA is working

---

## 4. Install PyTorch with CUDA 12.1 Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Upgrade pip (Optional)

```bash
python -m pip install --upgrade pip
```

---

## 6. Move to the Project Directory

```bash
cd your_path_to_project/Final_PPT
```

---

## 🔑 7. Set Up Your GROQ API Key

This project requires a valid **GROQ API key**.

### How to get your key:

1. [Go to groq site](https://console.groq.com/keys)
2. Sign up or log in
3. Create an API key

### Create a `.env` file in the `Final_PPT` folder:

```env
GROQ_API_KEY=your_key_here
```

✅ Replace `your_key_here` with your actual key.
🚫 Never share your `.env` or upload it to GitHub!

---

## 8. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Choose Experiment and Dataset

Open `main.py` and set the experiment and dataset by changing the following variables:

```python
experiment_name = "two_step_complete_ft"
dataset_name = "bitext"
```

Available options:

* `experiment_name`: `"single_step_pretrained"`, `"single_step_ft"`, `"two_step_baseline"`, `"two_step_pretrained"`, `"two_step_partial_ft"`, `"two_step_complete_ft"`
* `dataset_name`: `"bitext"`, `"customer_service"`

---

## 10. Run the Pipeline

```bash
python main.py
```

✅ Runs the selected experiment with the specified dataset.

---

# 📜 Useful Commands Summary

```bash
python --version
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Edit experiment_name and dataset_name in main.py
python main.py
```

---

## 📤 Output Files

After running, you will find the results saved under the `Final_PPT` folder:

* `human_scores.csv` — LLM-based human-like evaluation results
* `metrics.json` — Automatic evaluation metrics
* `intent_accuracy.json` — Accuracy of predicted intents (for two-step models)
