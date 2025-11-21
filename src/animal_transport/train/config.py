from pathlib import Path

from ..api.config import REASONING_MODEL_NAME

DATA_PATH = Path("data/train/train.jsonl")
OUTPUT_DIR = Path("models/reasoning_lora")
MODEL_NAME = REASONING_MODEL_NAME