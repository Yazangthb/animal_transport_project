"""
Generation callback module.

This module contains custom callbacks for the training process.
"""

import torch
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class GenerateCallback(TrainerCallback):
    """
    Callback that periodically generates text from a random training example.

    This helps monitor the model's progress during training by generating
    sample outputs at regular intervals.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        every_n_steps: int = 200,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the generate callback.

        Args:
            tokenizer: The tokenizer to use for generation
            dataset: Dataset to sample prompts from
            every_n_steps: Generate every N training steps
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _get_random_prompt(self):
        """
        Sample a random prompt from the dataset.

        Returns:
            str: A text prompt for generation
        """
        try:
            # Sample random item from dataset
            idx = torch.randint(0, len(self.dataset), (1,)).item()
            sample = self.dataset[idx]

            if isinstance(sample, dict):
                input_ids = sample.get("input_ids")
                attention_mask = sample.get("attention_mask", None)
            else:
                # Fallback for (input_ids, labels) tuple style datasets
                input_ids = sample[0]
                attention_mask = None

            if input_ids is None:
                raise ValueError("Sample from dataset does not contain 'input_ids'.")

            # Filter padding using attention_mask if available
            if attention_mask is not None:
                if isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor):
                    input_ids = input_ids[attention_mask.bool()]
                else:
                    # Assume list-like
                    input_ids = [i for i, m in zip(input_ids, attention_mask) if m == 1]

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

            # Fallback if decoding yields empty/garbage
            if not text.strip():
                text = "Hello, how are you?"

            return text

        except Exception as e:
            logger.warning(f"Failed to get random prompt: {e}")
            return "Hello, how are you?"

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.every_n_steps <= 0:
            return control

        if state.global_step == 0:
            return control

        if state.global_step % self.every_n_steps != 0:
            return control

        model = kwargs["model"]
        try:
            prompt = self._get_random_prompt()
        except Exception as e:
            logger.error(f"Failed to get random prompt: {e}")
            prompt = "Hello, how are you?"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(model.device)

        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        model.train()

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logger.info("=" * 50)
        logger.info(f"[Step {state.global_step}] SAMPLE GENERATION")
        logger.info("-" * 50)
        logger.info(decoded)
        logger.info("=" * 50)

        return control