# Introduction
This project is a Toxic Comment Classification Challenge from Kaggle [(challenge link)](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) that I finished to further challenge myself in fine-tuning model for solving multi-label classification case. It is a multi label classification system that detect six types of toxicity in text comment using fine-tuned BERT model. This project features a fully functional backend system with FastAPI, simple frontend with Gradio, and containerized deployment with Docker Compose.

---

# Features
- Multi-label classification with six types of toxicity level:
  - toxic
  - severe toxic
  - obscene
  - threat
  - insult
  - identity hate
- Fine-tuned BERT model (bert-base-cased)
- Hugging Face Trainer API for training and evaluation
- REST API built with FastAPI
- Frontend built with Gradio
- Dockerized backend and frontend with Docker Compose

---

# Tech Stack
| Component   | Technology                |
|-------------|---------------------------|
| NLP Model   | BERT (bert-base-cased     |
| Framework   | Pytorch, Transformer      |
| Backend     | FastAPI                   |
| Frontend    | Gradio                    |
| Container   | Docker, Docker Compose    |

---

# Setup and Installation
```
# Git clone this repo
https://github.com/stnlyyg/toxic-comment-classifier.git

# Run this in your terminal (cd to your root dir)
pip install -r requirements.txt
```

---

# Data Processing, Training, and Evaluation
Dataset can be found in the kaggle competition page [(dataset link)](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

## Data Processing
This dataset consist of comment id, comment_text, and 6 level of toxicity separated by each columns. No lowercasing or data cleaning by removing stop words or symbols as the model can leverage them to better understand the context.
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/92f04d70-da30-44ca-9994-2879d5fd5aa3" />
The dataset is featured to only consist of comment_text and a single label column which was the 6 level of toxicity that was merged into one column values. This dataset is then converted to Trainer API friendly format and split into 80:20 ratio for training and model optimization purpose.
<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/eac68aa2-6279-4aac-92fd-d8ae3ff7feb6" />
```
# cd to toxic-comment-classifier/src/ and run this in your terminal
python evaluation.py
```

## Training
Fine-tuning the BERT model leveraging Hugging Face Trainer API.
```
# cd to toxic-comment-classifier/src/ and run this in your terminal
python train.py

Training result for 3 epochs
{'eval_loss': 0.042469725012779236, 'eval_f1': 0.7545822423232563, 'eval_roc_auc': 0.9898302497351615, 'eval_accuracy': 0.9261789127369575, 'eval_runtime': 198.2265, 'eval_samples_per_second': 161.003, 'eval_steps_per_second': 20.128, 'epoch': 1.0}

{'eval_loss': 0.03773674741387367, 'eval_f1': 0.7870643827525103, 'eval_roc_auc': 0.9922916654444472, 'eval_accuracy': 0.9284035719880934, 'eval_runtime': 199.447, 'eval_samples_per_second': 160.017, 'eval_steps_per_second': 20.005, 'epoch': 2.0}

{'eval_loss': 0.03934187442064285, 'eval_f1': 0.7955635062611807, 'eval_roc_auc': 0.991791863341847, 'eval_accuracy': 0.9285289049036504, 'eval_runtime': 199.0105, 'eval_samples_per_second': 160.368, 'eval_steps_per_second': 20.049, 'epoch': 3.0}
```

## Evaluation
Evaluation uses 
