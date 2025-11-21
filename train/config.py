from pathlib import Path

from api.config import REASONING_MODEL_NAME

DATA_PATH = Path("train/dataset/train.jsonl")
OUTPUT_DIR = Path("models/reasoning_lora")
MODEL_NAME = REASONING_MODEL_NAME