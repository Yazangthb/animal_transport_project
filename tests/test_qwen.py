import sys
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration  # <-- CORRECT class

sys.path.insert(0, 'src')
from animal_transport.api.config import VLM_MODEL_NAME

model_name = VLM_MODEL_NAME

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map={"": "cpu"},   # required on RTX 1650
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("Loaded successfully!")
