import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer 
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from datasets import Dataset
import pandas as pd

import config

this_model = BertForSequenceClassification.from_pretrained(config.BEST_CHECKPOINT_PATH, num_labels=config.NUM_LABELS, problem_type=config.PROBLEM_TYPE)
this_tokenizer = AutoTokenizer.from_pretrained(config.BEST_CHECKPOINT_PATH)

def competition_evaluation():
    #check if test and test_labels dataset exist
    if not config.COMP_EVAL_TEST.is_file() and config.COMP_EVAL_TEST_LABELS.is_file():
        return f'test.csv dataset not found for evaluation'

    #create df
    df_test = pd.read_csv(config.COMP_EVAL_TEST)
    df_test_labels = pd.read_csv(config.COMP_EVAL_TEST_LABELS)

    #filter unscored rows
    df_test_labels_removed = df_test_labels[df_test_labels['toxic'] != -1].copy()

    #create label
    label_col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_test_labels_removed['label'] = df_test_labels_removed[label_col].values.tolist()
    df_test_labels_removed = df_test_labels_removed.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)

    #filter test dataset
    filtered_id = df_test_labels_removed['id']
    df_test_filtered = df_test[df_test['id'].isin(filtered_id)]
    df_test_filtered = df_test_filtered.reset_index(drop=True)

    comp_eval_test = Dataset.from_pandas(df_test_filtered)

    def tokenize_data(data):
        return this_tokenizer(data['comment_text'], truncation=True, padding=True, max_length=config.MAX_LENGTH)

    tokenized_comp_eval_dataset = comp_eval_test.map(tokenize_data, batched=True)

    #Evaluation
    eval_args = TrainingArguments(**config.COMP_EVAL_ARGS)
    
    evaluator = Trainer(
        model=this_model,
        args=eval_args
    )

    prediction_output = evaluator.predict(tokenized_comp_eval_dataset)
    prediction_logits = prediction_output.predictions
    probs = 1/ (1 + np.exp(-prediction_logits))

    y_pred = (probs >= config.PROB_THRESHOLD).astype(int)
    y_true = df_test_labels_removed['label'].tolist()

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

    plt.tight_layout()
    fig.savefig(config.COMP_EVAL_RESULT_DIR)
    
    plt.show()

if __name__ == "__main__":
    competition_evaluation()