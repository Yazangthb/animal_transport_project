# api/config.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

# Models
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
REASONING_LORA_PATH = os.getenv("REASONING_LORA_PATH", "models/reasoning_lora")

# Device settings (Windows-safe)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RTX 1650 does NOT support bf16
DTYPE = "float16"
