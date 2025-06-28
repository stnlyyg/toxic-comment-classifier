#import from library
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

#import from directory
import config

def main():
    if config.TRAIN_DATA_FILE.exists() and config.TEST_DATA_FILE.exists():
        print("--Dataset exist and processed--")
        print(f"Train dataset location: {config.TRAIN_DATA_FILE}")
        print(f"Test dataset location: {config.TEST_DATA_FILE}")
        return
    
    print(f"--Dataset has not been processed--")
    print(f"--Starting data processing--")

    if not config.RAW_TRAIN_DATA_FILE.is_file():
        print(f"--Training dataset not found at {config.RAW_TRAIN_DATA_FILE}!--")
        return
    
    df_raw_data = pd.read_csv(config.RAW_TRAIN_DATA_FILE)
    df_raw_data.dropna(subset=['comment_text'], inplace=True)

    label_col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_raw_data['label'] = df_raw_data[label_col].values.tolist()

    df_train_data = df_raw_data[['comment_text', 'label']]

    training_dataset = Dataset.from_pandas(df_train_data)

    bert_cased_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    def tokenize_data(data):
        return bert_cased_tokenizer(data['comment_text'], truncation=True, padding=True, max_length=config.MAX_LENGTH)

    tokenized_dataset = training_dataset.map(tokenize_data, batched=True)

    train_eval_split_data = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_eval_split_data['train']
    eval_dataset = train_eval_split_data['test']

    #save to disk
    train_dataset.save_to_disk(config.TRAIN_DATA_FILE)
    eval_dataset.save_to_disk(config.TEST_DATA_FILE)

if __name__ == "__main__":
    main()