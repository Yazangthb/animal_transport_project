# api/inference_reasoning.py
import json
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .config import REASONING_MODEL_NAME, REASONING_LORA_PATH


class ReasoningWrapper:
    def __init__(self):
        # Windows + RTX 1650 safe dtype
        dtype = torch.float16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            REASONING_MODEL_NAME,
            trust_remote_code=True
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            REASONING_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # Try loading LoRA
        try:
            self.model = PeftModel.from_pretrained(base_model, REASONING_LORA_PATH)
            print(f"[INFO] Loaded LoRA adapters from {REASONING_LORA_PATH}")
        except Exception:
            print("[WARNING] No LoRA adapters found. Using base model only.")
            self.model = base_model

        self.model.eval()

    def _build_prompt(self, features: Dict) -> str:
        schema = (
            "You are a transport planning assistant for animals.\n"
            "Given a JSON input about an animal and a distance in kilometers,\n"
            "output a strictly valid JSON with allowed modes and time estimates.\n\n"
            "Output JSON structure:\n"
            "{\n"
            '  "available_modes": [\n'
            "    {\n"
            '      "mode": "truck | train | ship | passenger_plane_cabin | passenger_plane_cargo | cargo_plane",\n'
            '      "estimated_time_hours": float,\n'
            '      "notes": "short explanation"\n'
            "    }\n"
            "  ],\n"
            '  "disallowed_modes": ["truck", ...],\n'
            '  "reasoning": "explanation"\n'
            "}\n\n"
            "Respond **only** with JSON.\n\n"
        )

        input_json = json.dumps(features, ensure_ascii=False)
        return schema + "Input:\n" + input_json + "\n\nOutput JSON:\n"

    def plan_transport(self, features: Dict) -> Dict:
        prompt = self._build_prompt(features)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract JSON
        try:
            if "{" in generated:
                json_str = generated[generated.index("{"): generated.rindex("}") + 1]
            else:
                json_str = generated
            data = json.loads(json_str)
        except Exception:
            data = {
                "available_modes": [],
                "disallowed_modes": [],
                "reasoning": "Failed to produce valid JSON."
            }

        # Normalize missing keys
        if "available_modes" not in data:
            data["available_modes"] = []
        if "disallowed_modes" not in data:
            data["disallowed_modes"] = []
        if "reasoning" not in data:
            data["reasoning"] = ""

        return data


_reasoning_wrapper: Optional[ReasoningWrapper] = None


def get_reasoning_wrapper() -> ReasoningWrapper:
    global _reasoning_wrapper
    if _reasoning_wrapper is None:
        _reasoning_wrapper = ReasoningWrapper()
    return _reasoning_wrapper
