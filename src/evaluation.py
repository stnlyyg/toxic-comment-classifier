import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, TrainingArguments, Trainer 
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from datasets import load_from_disk

import config

def model_evaluation():
    this_model = BertForSequenceClassification.from_pretrained(config.BEST_CHECKPOINT_PATH, num_labels=config.NUM_LABELS, problem_type="multi_label_classification")

    eval_args = TrainingArguments(**config.EVAL_ARGS)
    
    evaluator = Trainer(
        model=this_model,
        args=eval_args
    )

    prediction_output = evaluator.predict(load_from_disk(config.TEST_DATA_FILE))
    prediction_logits = prediction_output.predictions
    probs = 1/ (1 + np.exp(-prediction_logits))

    y_pred = (probs >= config.PROB_THRESHOLD).astype(int)
    y_true = prediction_output.label_ids

    mcm = multilabel_confusion_matrix(y_true, y_pred)

    print(f'=============== Classification Report ===============')
    print(classification_report(y_true, y_pred, target_names=config.LABELS, zero_division=0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (matrix, label) in enumerate(zip(mcm, config.LABELS)):
        ax = axes[i]
        sns.heatmap(matrix, annot=True, fmt='d', cmap="Greens", ax=ax, 
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"],
                    cbar=False)
        ax.set_title(f"Confusion Matrix for {label}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout
    fig.savefig(config.EVAL_RESULT_DIR)
    
    plt.show()

if __name__ == "__main__":
    model_evaluation()