import base64
import json
from io import BytesIO
from PIL import Image

import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from .config import VLM_MODEL_NAME


class VLMWrapper:
    def __init__(self):
        print("[VLM] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            VLM_MODEL_NAME, trust_remote_code=True
        )

        print("[VLM] Loading model on CPU (Windows-safe)...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_NAME,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        self.device = "cpu"

    def analyze_animal(self, img_base64: str):
        # Decode image
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Build prompt
        prompt = (
            "Analyze the provided animal image. "
            "Return ONLY valid JSON with EXACT keys: "
            "animal_category, size_class, is_domesticated, dangerous_to_humans."
        )

        messages = [
            {"role": "system", "content": "You classify animals from images."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # 1) Build text chat template (string)
        text = self.processor.apply_chat_template(
            messages, tokenize=False
        )

        # 2) Process image + text into model inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # 3) Generate model output
        output = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
        )

        decoded = self.processor.batch_decode(
            output, skip_special_tokens=True
        )[0]

        # Extract JSON
        start = decoded.find("{")
        end = decoded.rfind("}") + 1
        json_str = decoded[start:end]

        try:
            return json.loads(json_str)
        except Exception:
            print("[VLM] RAW OUTPUT:", decoded)
            raise ValueError("Failed to parse JSON from VLM output.")



_vlm_instance = None


def get_vlm_wrapper():
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = VLMWrapper()
    return _vlm_instance
