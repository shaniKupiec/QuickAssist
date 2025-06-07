# QuickAssist: Intent-Aware Chatbot for Customer Support

## 🧠 Overview

*QuickAssist* is an NLP course project focused on improving chatbot responses in customer support settings using *intent conditioning. Our work investigates whether explicitly providing the chatbot with an **intent label* improves the *quality* and *appropriateness* of its replies.

---

## 📂 Repository Structure

This repository includes all three project milestones with both presentation materials and code. The directory is organized to reflect the **growth and evolution** of our approach:

```bash
QuickAssist/
├── Project_Proposal_Presentation.pdf      # Initial idea and proposal
├── Mid_PPT/
│   ├── requirements.txt                   # Dependencies for mid-stage code
│   ├── Mid_Project_Presentation.pdf       # Mid-project presentation
│   └── README_Mid_PPT.md                  # Detailed mid-stage experiment overview
├── Final_PPT/
│   ├── requirements.txt                   # Final dependencies
│   ├── Final_Project_Presentation.pdf     # Final presentation slides
│   └── README_Final_PPT.md                # Full description of final model, experiments, and results
```

Each stage documents:

* Methodology & motivation
* Code progression
* Intermediate and final results

---

## 🎯 Project Summary

Modern customer support bots often *fail to generate useful answers* due to a lack of intent understanding. This project explores whether *explicitly adding the user's intent* to the chatbot input improves response quality.

We tested two main configurations:

* *Variant A*: Only the customer query
* *Variant B*: Customer query + intent label

We evaluated response quality through *automatic metrics* (e.g., BLEU, ROUGE, BERTScore) and *human-like scoring* using GPT-4.

🧪 For detailed results and technical insights, refer to the subfolders:

* 🔗 [Mid-PPT README](https://github.com/shaniKupiec/QuickAssist/blob/main/Mid_PPT/README_Mid_PPT.md)
* 🔗 [Final-PPT README](https://github.com/shaniKupiec/QuickAssist/blob/main/Final_PPT/README_Final_PPT.md)

---

## 🔍 Graphical Abstract

> A visual summary of our project, datasets, models, and results.

![Graphical Abstract](https://github.com/user-attachments/assets/0d0d9f3c-bef1-4430-939f-4c65823a5468)

---

## 👥 Team

* [Inbal Bolshinsky](https://github.com/InbalBolshinsky)
* [Shani Kupiec](https://github.com/shaniKupiec)
* [Almog Sasson](https://github.com/Almog-Sasson)
* [Nadav Margaliot](https://github.com/NadavMargaliot)

📍 NLP Course, Holon Institute of Technology (HIT)
