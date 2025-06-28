from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

#Directories
DATA_DIR = ROOT_DIR / "data"

RAW_TRAIN_DATA_FILE = DATA_DIR / "train.csv"
RAW_TEST_DATA_FILE = DATA_DIR / "test.csv"
RAW_TESTING_DATA = DATA_DIR / "testtt.csv"

PROCESSED_DATA_FILE = DATA_DIR / "processed"
TRAIN_DATA_FILE = PROCESSED_DATA_FILE / "train"
TEST_DATA_FILE = PROCESSED_DATA_FILE / "test"

PROCESSED_DATA_FILE.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "google-bert/bert-base-cased"

MAX_LENGTH = 256