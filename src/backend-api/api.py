from fastapi import FastAPI
from transformers import BertForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API to classify toxic comment",
    version="1.0.0"
)

try:
    model = BertForSequenceClassification.from_pretrained(config.BEST_CHECKPOINT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.BEST_CHECKPOINT_PATH)
except Exception as e:
    print(f"Error loading {e}")
    exit()

@app.get("/")
def get_root():
    return {"Message": "Welcome to the Toxic Comment Classifier API. Use the /docs endpoint to see the documentation."}

class toxicComment(BaseModel):
    comment: str

@app.post("/comment_classifier/")
async def classify_comments(comment: toxicComment, threshold: float =  config.PROB_THRESHOLD):
    if not comment.comment or not comment.comment.strip():
        return {"Error": "Comment can't be blank or whitespace."}
    
    inputs = tokenizer(comment.comment, return_tensors='pt', truncation=True, padding=True, max_length=config.MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1

    preds = preds[0]

    where_is_one = np.where(preds==1)[0]
    toxic_classifications = [config.id2label[i] for i in where_is_one]
    toxic_classifications_clean = ", ".join(toxic_classifications)

    if not toxic_classifications:
        return {"result": "This comment is classified as non-toxic"}

    return {"result": f"This comment is classifier as {toxic_classifications_clean}"}