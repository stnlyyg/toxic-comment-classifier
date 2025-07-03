from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import gradio as gr
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

finetuned_model = BertForSequenceClassification.from_pretrained(config.BEST_CHECKPOINT_PATH, num_labels=config.NUM_LABELS, problem_type=config.PROBLEM_TYPE)
finetuned_tokenizer = AutoTokenizer.from_pretrained(config.BEST_CHECKPOINT_PATH)

def classify_comments(text: str, threshold=config.PROB_THRESHOLD):
    if not text or not text.strip():
        return {}
    
    inputs = finetuned_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=config.MAX_LENGTH)

    with torch.no_grad():
        outputs = finetuned_model(**inputs)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits)

    preds = np.zeros(probs.shape)
    preds[np.where(probs >= threshold)] = 1

    preds = preds[0]

    id2label = {
    0: 'toxic',
    1: 'severe toxic',
    2: 'obscene',
    3: 'threat',
    4: 'insult',
    5: 'identity hate',
    }

    where_is_one = np.where(preds==1)[0]
    toxic_classifications = [id2label[i] for i in where_is_one]
    toxic_classifications_clean = ", ".join(toxic_classifications)

    if not toxic_classifications:
        return f'Your comment is classified as non-toxic'

    return f'Your comment is classified as {toxic_classifications_clean}'

def main_app():
    classifier_app = gr.Interface(
        fn=classify_comments,
        inputs=gr.TextArea(placeholder="Enter a comment here...", label="Input comment"),
        outputs = gr.TextArea(label="Comment classified as: "),
        title="Toxic comment classifier"
    )

    classifier_app.launch()

if __name__ == "__main__":
    main_app()