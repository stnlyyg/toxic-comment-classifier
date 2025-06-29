#import library
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets import load_from_disk

#import directory
import config

this_model = BertForSequenceClassification.from_pretrained(config.BASE_MODEL, num_labels=config.NUM_LABELS, problem_type="multi_label_classification")
this_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true=y_true, y_score=probs, average='micro')
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result

training_args = TrainingArguments(**config.TRAINING_ARGS)

trainer = Trainer(
    model = this_model,
    args = training_args,
    train_dataset = load_from_disk(config.TRAIN_DATA_FILE),
    eval_dataset = load_from_disk(config.TEST_DATA_FILE),
    processing_class = this_tokenizer,
    compute_metrics = compute_metrics
)

if __name__ == "__main__":
    trainer.train()