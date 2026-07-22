from __future__ import annotations

from pathlib import Path
from typing import Any

from ..cache import configure_huggingface_cache
from ..records import ExperimentCase
from .base import ModelAdapter


class QwenVLAdapter(ModelAdapter):
    model_key = "qwen3_vl"

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        hf_home: str | Path,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        revision: str | None = None,
    ) -> None:
        cache_root = configure_huggingface_cache(hf_home)
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Qwen3-VL-8B inference requires a GPU; no CUDA device is visible"
                )
            device = "cuda"
        if not device.startswith("cuda"):
            raise ValueError("Qwen3-VL-8B inference is restricted to CUDA")

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
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", revision)

    @staticmethod
    def build_messages(case: ExperimentCase) -> list[dict[str, Any]]:
        if len(case.image_paths) != 1:
            raise ValueError(
                f"{case.case_id}: Qwen3-VL single-patch inference requires "
                f"exactly one image, found {len(case.image_paths)}"
            )
        messages: list[dict[str, Any]] = []
        if case.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": case.system_prompt}],
                }
            )
        content = [{"type": "image", "image": case.image_paths[0]}]
        content.append({"type": "text", "text": case.user_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def generate(self, case: ExperimentCase) -> str:
        import torch

        inputs = self.processor.apply_chat_template(
            self.build_messages(case),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        continuation = generated[0, input_length:]
        return self.processor.decode(
            continuation, skip_special_tokens=True
        ).strip()

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype_name,
            "max_new_tokens": self.max_new_tokens,
            "image_input_protocol": "one_frozen_patch_per_generation",
            "do_sample": False,
        }
