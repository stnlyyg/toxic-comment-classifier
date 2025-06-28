from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

#Directories
DATA_DIR = ROOT_DIR / "data"

#Data Processing
RAW_TRAIN_DATA_FILE = DATA_DIR / "train.csv"
RAW_TEST_DATA_FILE = DATA_DIR / "test.csv"

PROCESSED_DATA_FILE = DATA_DIR / "processed"
TRAIN_DATA_FILE = PROCESSED_DATA_FILE / "train"
TEST_DATA_FILE = PROCESSED_DATA_FILE / "test"

PROCESSED_DATA_FILE.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256

#Model
MODEL_DIR = ROOT_DIR / "model"
SAVED_MODEL_PATH = MODEL_DIR / "toxic-comment-classifier"
BEST_CHECKPOINT_PATH = SAVED_MODEL_PATH / "checkpoint-"

BASE_MODEL = "google-bert/bert-base-cased"

#Log
LOG_DIR = ROOT_DIR / "logs"

#Training
TRAINING_ARGS = {
    "output_dir": str(SAVED_MODEL_PATH),
    "seed": 42,
    'fp16': True,

    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_ratio": 0.5,

    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 8,

    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "save_total_limit": 1,

    "logging_dir": LOG_DIR,
    "logging_strategy": "steps",
    "logging_steps": 100,
    "report_to": "tensorboard"
}