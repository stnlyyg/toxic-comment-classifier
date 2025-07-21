# Introduction
This project is a Toxic Comment Classification Challenge from Kaggle that I did to further challenge myself in fine-tuning model for solving multi-label classification case. It is a multi label classification system that detect six types of toxicity in text comment using fine-tuned BERT model. This project features a fully functional backend system with FastAPI, simple frontend with Gradio, and containerized deployment with Docker Compose.

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
## Data Processing
This dataset consist of comment id, comment_text, and 6 level of toxicity separated by each columns. 
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/92f04d70-da30-44ca-9994-2879d5fd5aa3" />
The dataset is featured to only consist of comment_text and a single label column which was the 6 level of toxicity that was merged into one column values. This dataset is then converted to Trainer API friendly format and split into 80:20 ratio for training and model optimization purpose.
<img width="1919" height="912" alt="image" src="https://github.com/user-attachments/assets/eac68aa2-6279-4aac-92fd-d8ae3ff7feb6" />

