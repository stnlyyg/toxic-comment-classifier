# Introduction
This project is a multi label classification system that detect six types of toxicity in text comment using fine-tuned BERT model. This project features a fully functional backend system with FastAPI, simple frontend with Gradio, and containerized deployment with Docker Compose.

---

## Features
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

## Tech Stack
| Component   | Technology                |
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

# Run this in your terminal (cd in root dir)
pip install -r requirements.txt
```
