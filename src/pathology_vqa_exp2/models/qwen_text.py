from __future__ import annotations

from pathlib import Path
from typing import Any

from ..cache import configure_huggingface_cache
from ..records import ExperimentCase
from .base import ModelAdapter


class QwenTextAdapter(ModelAdapter):
    model_key = "qwen3_text"

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-8B",
        hf_home: str | Path,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        revision: str | None = None,
    ) -> None:
        cache_root = configure_huggingface_cache(hf_home)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Qwen3-8B inference requires a GPU; no CUDA device is visible"
                )
            device = "cuda"
        if not device.startswith("cuda"):
            raise ValueError("Qwen3-8B smoke inference is restricted to CUDA")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", revision)

    def generate(self, case: ExperimentCase) -> str:
        import torch

        messages: list[dict[str, str]] = []
        if case.system_prompt:
            messages.append({"role": "system", "content": case.system_prompt})
        messages.append({"role": "user", "content": case.user_prompt})
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
                eos_token_id=self.tokenizer.eos_token_id,
            )
        continuation = generated[0, input_length:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype_name,
            "max_new_tokens": self.max_new_tokens,
            "thinking": False,
            "do_sample": False,
        }
