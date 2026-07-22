from __future__ import annotations

from pathlib import Path
from typing import Any

from ..cache import configure_huggingface_cache
from ..records import ExperimentCase
from .base import ModelAdapter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLAdapter(ModelAdapter):
    model_key = "internvl3"

    def __init__(
        self,
        *,
        model_id: str = "OpenGVLab/InternVL3-8B",
        hf_home: str | Path,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        image_size: int = 448,
        revision: str | None = None,
    ) -> None:
        cache_root = configure_huggingface_cache(hf_home)
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "InternVL3-8B inference requires a GPU; no CUDA device is visible"
                )
            device = "cuda"
        if not device.startswith("cuda"):
            raise ValueError("InternVL3-8B inference is restricted to CUDA")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.torch_dtype = dtype_map[dtype]
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
        ).eval().to(device)
        self.revision = getattr(self.model.config, "_commit_hash", revision)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def build_question(case: ExperimentCase) -> str:
        if len(case.image_paths) != 1:
            raise ValueError(
                f"{case.case_id}: InternVL3 single-patch inference requires "
                f"exactly one image, found {len(case.image_paths)}"
            )
        return "<image>\n" + case.user_prompt

    def _load_pixel_values(self, image_paths: tuple[str, ...]):
        import torch
        from PIL import Image

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.transform(image.convert("RGB")))
        return torch.stack(tensors).to(
            device=self.device,
            dtype=self.torch_dtype,
        )

    def generate(self, case: ExperimentCase) -> str:
        if len(case.image_paths) != 1:
            raise ValueError(
                f"{case.case_id}: InternVL3 single-patch inference requires "
                f"exactly one image, found {len(case.image_paths)}"
            )
        pixel_values = self._load_pixel_values(case.image_paths)
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        previous_system_message = self.model.system_message
        self.model.system_message = case.system_prompt or ""
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                self.build_question(case),
                generation_config,
                num_patches_list=[1],
                history=None,
                return_history=False,
            )
        finally:
            self.model.system_message = previous_system_message
        return str(response).strip()

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype_name,
            "max_new_tokens": self.max_new_tokens,
            "image_size": self.image_size,
            "image_preprocessing": "one resized tensor for one frozen patch",
            "image_input_protocol": "one_frozen_patch_per_generation",
            "dynamic_tiling": False,
            "do_sample": False,
        }
