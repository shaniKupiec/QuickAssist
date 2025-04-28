# Mid_PPT Project

This project contains baseline code and visualizations for QuickAssist tasks.

## 📂 Files Overview

| File | Description |
| :--- | :--- |
| `baseline.py` | Main Python script to generate baseline results. |
| `Visualizations_For_QuickAssist.ipynb` | Jupyter Notebook for analyzing and visualizing outputs. |
| `checkgpu.py` | Simple script to check GPU availability. |
| `.env` | Environment variables file (sensitive data, not uploaded). |
| `.gitignore` | Git ignore file (ignores `.env`, `.ipynb_checkpoints/`, etc.). |
| `requirements.txt` | Project dependencies list. |

---

# To run `Visualizations_For_QuickAssist.ipynb`:

1. Open Jupyter Notebook (or VSCode with Jupyter extension).
2. Open the notebook file.
3. Run the cells step-by-step.

---

# 🚀 How to Set Up and Run `baseline.py`

## 1. Check Python Installation

Open a terminal and type:

```bash
python --version
```

✅ Confirm that Python 3.9, 3.10, or 3.11 is installed.

---

## 2. Create and Activate a Virtual Environment (Recommended)

Create:

```bash
python -m venv myenv
```

Activate (Windows):

```bash
myenv\Scripts\activate
```

---

## 3. (Optional) Check GPU Availability

To check if your GPU is available and CUDA drivers are working, type:

```bash
nvidia-smi
```

✅ Note your CUDA version.

ℹ️ **Important**:  
PyTorch officially supports **up to CUDA 12.1** for now.  
**No problem!**  
CUDA drivers are **backward compatible**, so even if your `nvidia-smi` shows a higher version (like 12.3 or 12.4),  
you can still **safely use** the CUDA 12.1 build of PyTorch.

---

## 4. Install PyTorch (GPU Version)

Inside the virtual environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

✅ This installs the correct version of PyTorch that works with your GPU and drivers.

---

## 5. Upgrade pip (Optional but Recommended)


```bash
python -m pip install --upgrade pip
```

---

## 6. Move to the project path

```bash
cd ./your_path_to_to_the_project/Mid_PPT
```

✅ **Important:**
- Replace `your_path_to_to_the_project` with your real path.

---

## 🔑 7. Set Up Your GROQ API Key

This project **requires** a GROQ API key for access to the models.

You must manually create a `.env` file in the project folder and add your API key:

1. Create a file named `.env` in the root of your project (same level as your Python files).
2. Open the `.env` file and add:

```bash
GROQ_API_KEY=your_actual_api_key_here
```

✅ **Important:**
- Replace `your_actual_api_key_here` with your real API key.
- Make sure there are **no spaces** around the `=`.
- Keep the `.env` file private — **never upload it to GitHub** or share it publicly!

---

## 8. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---


## 9. Verify GPU from Python

Before running the main scripts, it’s recommended to **run the GPU check** script:

```bash
python checkgpu.py
```

✅ It will print whether the GPU is available and which device is being used.

---

## 10. Running the Scripts

### To run `baseline.py`:

```bash
python baseline.py
```

✅ This will generate and save outputs inside the directory:

```
C:\Shani\baselineOutputs
```
*(The folder will be created automatically if it does not exist.)*

---


# ⚙️ Notes

- All AI-generated outputs and evaluations are saved in the output directory (`Shani\baselineOutputs`).
- Environment variables (like API keys) should be stored inside the `.env` file.
- `.env` is already ignored by `.gitignore`.

---

# 📜 Useful Commands Summary

```bash
python --version
python -m venv myenv
myenv\Scripts\activate
nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python checkgpu.py
python baseline.py
```

---

# 🌟 You're Ready!

---