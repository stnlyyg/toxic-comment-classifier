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
Evaluation uses test labels which was a separated data and was provided by the challenge organizer specifically for evaluation.
<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/fe49331c-e116-4b24-b2eb-cec66c9b7358" />

Classification report and confusion matrix are used as this project evaluation metrics.
```
# cd to toxic-comment-classifier/src/ and run this in your terminal
python evaluation.py
```
<img width="1508" height="295" alt="image" src="https://github.com/user-attachments/assets/49f52c23-1b1f-4807-8383-710dc6d6e31b" />
<img width="1480" height="787" alt="image" src="https://github.com/user-attachments/assets/1f4b3dea-d70b-43fe-b6f8-abc07d7baf04" />

---

# Running the App
The app is separated into backend that contain the API and frontend with gradio for quick demo. There are two ways to run this, via local or docker compose.

## Run Locally
To run the app locally, you must first run the backend in your terminal, then follow by running frontend in separated terminal.
```
# cd to toxic-comment-classifier/src/backend-api/ and run this in your first terminal
uvicorn api:app --reload
```
<img width="1508" height="301" alt="image" src="https://github.com/user-attachments/assets/803298bd-d338-4d82-bd31-d8dd994a7cde" />

```
# cd to toxic-comment-classifier/src/frontend-gradio/ and run this in your second terminal
python gradio_app.py
```
<img width="1545" height="194" alt="image" src="https://github.com/user-attachments/assets/5f910189-d07a-48a1-a757-ae92879d39c2" />
<br/><br/>
You can open localhost:8000/docs on your browser to use the app through FastAPI UI
<img width="1918" height="1018" alt="image" src="https://github.com/user-attachments/assets/1fd66dd2-1a90-4bc9-8e55-91dc08550103" />
<br/><br/>
You can open localhost:7860 on your browser to use the gradio app frontend via connection to the backend
<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/ba71ce2f-9081-4ee2-8d8e-3b516fa6f76c" />

---

## Run with Docker Compose
To run the app via Docker, make sure your docker system is up and running before proceeding to next step.
```
# In root directory, run this in your terminal
docker-compose up --build

You can open localhost:8000/docs on your browser to use the app through FastAPI UI  
You can open localhost:7860 on your browser to use the gradio app via connection to the backend
```
