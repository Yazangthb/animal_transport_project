import json
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .config import REASONING_MODEL_NAME, REASONING_LORA_PATH
from .prompts import SYSTEM_PROMPT


class ReasoningWrapper:
    def __init__(self):
        # Decide device once and reuse
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Reasoning] Using device: {self.device}")

        print("[Reasoning] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            REASONING_MODEL_NAME,
            trust_remote_code=True
        )

        # Let HF load normally, we'll move to device ourselves
        print("[Reasoning] Loading base model...")
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            REASONING_MODEL_NAME,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Try loading LoRA
        try:
            print(f"[Reasoning] Loading LoRA adapters from {REASONING_LORA_PATH}")
            self.model = PeftModel.from_pretrained(base_model, REASONING_LORA_PATH)
        except Exception:
            print("[Reasoning] WARNING: No LoRA adapters found. Using base model only.")
            self.model = base_model

        # Move the full model to the chosen device
        self.model.to(self.device)
        self.model.eval()

    def _build_messages(self, features: Dict) -> list:
        input_json = json.dumps(features, ensure_ascii=False)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_json}
        ]

    def plan_transport(self, features: Dict) -> Dict:
        messages = self._build_messages(features)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and move inputs to the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Use input length from the tensor on the same device
        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        print("generated: ", generated)
        try:
            # Assume the generated text starts with JSON
            if "{" in generated:
                start = generated.index("{")
                end = generated.rindex("}") + 1
                json_str = generated[start:end]
            else:
                json_str = generated.strip()
            print("extracted json_str: ", repr(json_str))
            data = json.loads(json_str)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            data = {
                "available_modes": [],
                "disallowed_modes": [],
                "reasoning": "Failed to produce valid JSON."
            }

        # Normalize keys from model output to expected schema
        if "allowed_modes" in data:
            data["available_modes"] = data.pop("allowed_modes")
        if "disallowed_modes" not in data:
            data["disallowed_modes"] = []
        if "available_modes" not in data:
            data["available_modes"] = []
        if "reasoning" not in data:
            data["reasoning"] = ""

        # Handle different time formats if present
        if "estimated_time" in data:
            time_str = data["estimated_time"]
            if isinstance(time_str, str):
                try:
                    hours = float(time_str.split()[0])
                    data["available_modes"] = [{
                        "mode": "air",  # Assuming air from context
                        "estimated_time_hours": hours,
                        "notes": "Estimated time from model output."
                    }]
                except Exception:
                    pass
            data.pop("estimated_time", None)

        print("normalized data: ", data)
        return data


_reasoning_wrapper: Optional[ReasoningWrapper] = None


def get_reasoning_wrapper() -> ReasoningWrapper:
    global _reasoning_wrapper
    if _reasoning_wrapper is None:
        _reasoning_wrapper = ReasoningWrapper()
    return _reasoning_wrapper
