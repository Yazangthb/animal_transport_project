import json
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .config import REASONING_MODEL_NAME, REASONING_LORA_PATH
from .prompts import SYSTEM_PROMPT


class ReasoningWrapper:
    def __init__(self):
        print("[Reasoning] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            REASONING_MODEL_NAME,
            trust_remote_code=True
        )

        print("[Reasoning] Loading base model on CPU...")
        base_model = AutoModelForCausalLM.from_pretrained(
            REASONING_MODEL_NAME,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,   # avoid mixed precision on Windows CPU
            trust_remote_code=True
        )

        # Try loading LoRA
        try:
            print(f"[Reasoning] Loading LoRA adapters from {REASONING_LORA_PATH}")
            self.model = PeftModel.from_pretrained(base_model, REASONING_LORA_PATH)
        except Exception:
            print("[Reasoning] WARNING: No LoRA adapters found. Using base model only.")
            self.model = base_model

        self.model.eval()

    def _build_messages(self, features: Dict) -> list:
        input_json = json.dumps(features, ensure_ascii=False)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_json}
        ]

    def plan_transport(self, features: Dict) -> Dict:
        messages = self._build_messages(features)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated = self.tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
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

        data.setdefault("available_modes", [])
        data.setdefault("disallowed_modes", [])
        data.setdefault("reasoning", "")

        return data


_reasoning_wrapper: Optional[ReasoningWrapper] = None


def get_reasoning_wrapper() -> ReasoningWrapper:
    global _reasoning_wrapper
    if _reasoning_wrapper is None:
        _reasoning_wrapper = ReasoningWrapper()
    return _reasoning_wrapper
