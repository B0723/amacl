from __future__ import annotations

from abc import ABC, abstractmethod

from .config import RuntimeConfig


class BaseTextGenerator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class QwenLocalGenerator(BaseTextGenerator):
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class MockGenerator(BaseTextGenerator):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.cursor = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt, user_prompt
        if self.cursor >= len(self.responses):
            raise IndexError("MockGenerator responses exhausted.")
        response = self.responses[self.cursor]
        self.cursor += 1
        return response
