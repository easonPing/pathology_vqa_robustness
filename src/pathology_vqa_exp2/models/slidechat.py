from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import configure_huggingface_cache
from ..records import ExperimentCase
from .base import ModelAdapter


@contextmanager
def _last_token_logits_only(output_embeddings):
    """Avoid materializing prompt-token logits that generation never reads.

    Transformers 4.42 computes logits for every input position in Qwen2.  For
    a roughly 20k-token WSI sequence that creates an otherwise unused tensor
    of more than 11 GiB after the FP32 cast.  Greedy autoregressive generation
    only consumes the logits at the final position, so cropping the hidden
    states immediately before the unchanged output projection is exact.
    """

    original_forward = output_embeddings.forward

    def forward_last_token(hidden_states):
        return original_forward(hidden_states[..., -1:, :])

    output_embeddings.forward = forward_last_token
    try:
        yield
    finally:
        output_embeddings.forward = original_forward


class SlideChatAdapter(ModelAdapter):
    model_key = "slidechat"

    def __init__(
        self,
        *,
        slidechat_repo: str | Path,
        checkpoint: str | Path,
        hf_home: str | Path,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 64,
        max_feature_tokens: int = 38_400,
        revision: str | None = None,
    ) -> None:
        cache_root = configure_huggingface_cache(hf_home)
        repo = Path(slidechat_repo).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not (repo / "xtuner" / "model" / "llava.py").is_file():
            raise FileNotFoundError(f"Not a SlideChat checkout: {repo}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SlideChat checkpoint not found: {checkpoint_path}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from xtuner.model.llava import LLaVAModel

        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SlideChat inference requires a GPU; no CUDA device is visible"
                )
            device = "cuda"
        if not device.startswith("cuda"):
            raise ValueError("SlideChat smoke inference is restricted to CUDA")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        torch_dtype = dtype_map[dtype]

        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.max_feature_tokens = max_feature_tokens
        self.repo = repo
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
            encode_special_tokens=True,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_root / "hub",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = LLaVAModel(
            llm=llm,
            freeze_llm=False,
            pretrained_pth=None,
            train_stage="2",
            use_activation_checkpointing=False,
            hidden_size=512,
            enable_long_net=True,
        )
        state_dict, loaded_checkpoint = self._load_checkpoint_state(checkpoint_path)
        self.checkpoint_path = loaded_checkpoint
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        loaded_count = len(self.model.state_dict()) - len(incompatible.missing_keys)
        if loaded_count <= 0:
            raise RuntimeError("The SlideChat checkpoint did not match any model weights")
        del state_dict

        gpu_total_bytes = torch.cuda.get_device_properties(device).total_memory
        self.gpu_total_memory_gib = gpu_total_bytes / (1024**3)
        self.low_vram_component_swapping = gpu_total_bytes < 32 * (1024**3)
        if self.low_vram_component_swapping:
            # A 24 GB Turing card cannot hold the 7B LLM and LongNet's
            # xFormers workspace for a ~20k-token slide at the same time.
            # Keep only the component currently doing work on the GPU.
            self.model.llm.to(device)
            self.model.LongNet_encoder.to(device="cpu", dtype=torch_dtype)
            self.model.projector.to(device="cpu", dtype=torch_dtype)
        else:
            self.model.to(device)
            self.model.LongNet_encoder.to(device=device, dtype=torch_dtype)
            self.model.projector.to(device=device, dtype=torch_dtype)
        self.model.eval()
        # SlideChat's official test_full.py disables KV caching for generation.
        self.model.llm.config.use_cache = False
        self.revision = getattr(self.model.llm.config, "_commit_hash", revision)
        self.missing_key_count = len(incompatible.missing_keys)
        self.unexpected_key_count = len(incompatible.unexpected_keys)
        self._cached_feature_path: str | None = None
        self._cached_pixel_values = None

        try:
            self.repo_commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            self.repo_commit = None

    @staticmethod
    def _load_checkpoint_state(checkpoint_path: Path):
        import torch

        candidate = checkpoint_path
        if checkpoint_path.is_dir():
            model_state = checkpoint_path / "mp_rank_00_model_states.pt"
            if model_state.is_file():
                candidate = model_state
            else:
                from xtuner.model.utils import guess_load_checkpoint

                return guess_load_checkpoint(str(checkpoint_path)), checkpoint_path

        # Official DeepSpeed checkpoints contain metadata in addition to tensors.
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported SlideChat checkpoint payload: {candidate}")
        for key in ("state_dict", "module"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state, candidate
        if payload and all(hasattr(value, "shape") for value in payload.values()):
            return payload, candidate
        raise ValueError(
            f"SlideChat checkpoint has no state_dict or module mapping: {candidate}"
        )

    def _load_projected_feature(self, feature_path: str):
        import torch

        if feature_path == self._cached_feature_path:
            return self._cached_pixel_values
        if self.low_vram_component_swapping:
            self._cached_feature_path = None
            self._cached_pixel_values = None
            self.model.llm.to("cpu")
            torch.cuda.empty_cache()
            self.model.LongNet_encoder.to(self.device)
            self.model.projector.to(self.device)
        frame = pd.read_csv(feature_path)
        if frame.shape[1] < 512:
            raise ValueError(
                f"Feature must have at least 512 columns, found {frame.shape[1]}: "
                f"{feature_path}"
            )
        values = frame.iloc[:, :512].apply(pd.to_numeric, errors="raise").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Feature contains NaN or infinity: {feature_path}")
        if len(values) > self.max_feature_tokens:
            indices = np.linspace(
                0, len(values) - 1, self.max_feature_tokens, dtype=int
            )
            values = values[indices]
        feature = torch.from_numpy(values).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            encoded = self.model.LongNet_encoder(
                src_tokens=None,
                token_embeddings=feature.to(self.model.llm.dtype).permute(1, 0, 2),
            )["encoder_out"]
            projected = self.model.projector(encoded.permute(1, 0, 2))
        if self.low_vram_component_swapping:
            del feature, encoded
            self.model.LongNet_encoder.to("cpu")
            self.model.projector.to("cpu")
            torch.cuda.empty_cache()
            self.model.llm.to(self.device)
        self._cached_feature_path = feature_path
        self._cached_pixel_values = projected
        return projected

    def _encode_prompt(self, case: ExperimentCase):
        import torch
        from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE

        template = PROMPT_TEMPLATE.qwen_chat
        image_input = f"{DEFAULT_IMAGE_TOKEN}\n{case.user_prompt}"
        rendered = ""
        if case.system_prompt:
            rendered += template["SYSTEM"].format(system=case.system_prompt)
        rendered += template["INSTRUCTION"].format(input=image_input, round=1)

        chunks: list[list[int]] = []
        for index, chunk in enumerate(rendered.split(DEFAULT_IMAGE_TOKEN)):
            chunks.append(
                self.tokenizer.encode(chunk, add_special_tokens=(index == 0))
            )
        if len(chunks) != 2:
            raise ValueError("SlideChat prompt must contain exactly one image token")
        input_ids: list[int] = []
        for index, chunk in enumerate(chunks):
            input_ids.extend(chunk)
            if index == 0:
                input_ids.append(IMAGE_TOKEN_INDEX)
        return torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def generate(self, case: ExperimentCase) -> str:
        import torch
        from transformers import GenerationConfig, StoppingCriteriaList
        from xtuner.model.utils import prepare_inputs_labels_for_multimodal
        from xtuner.utils import PROMPT_TEMPLATE, StopWordStoppingCriteria

        if not case.feature_path:
            raise ValueError(f"{case.qid}: SlideChat requires a feature path")
        pixel_values = self._load_projected_feature(case.feature_path)
        input_ids = self._encode_prompt(case)
        multimodal_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.model.llm,
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        multimodal_inputs.pop("labels", None)
        stop_criteria = StoppingCriteriaList(
            [
                StopWordStoppingCriteria(self.tokenizer, word)
                for word in PROMPT_TEMPLATE.qwen_chat.get("STOP_WORDS", [])
            ]
        )
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        output_embeddings = self.model.llm.get_output_embeddings()
        with torch.inference_mode(), _last_token_logits_only(output_embeddings):
            generated = self.model.llm.generate(
                **multimodal_inputs,
                generation_config=generation_config,
                stopping_criteria=stop_criteria,
                bos_token_id=self.tokenizer.bos_token_id,
                use_cache=False,
            )
        return self.tokenizer.decode(
            generated[0], skip_special_tokens=True
        ).strip()

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "revision": self.revision,
            "slidechat_repo": str(self.repo),
            "slidechat_repo_commit": self.repo_commit,
            "checkpoint": str(self.checkpoint_path),
            "device": self.device,
            "dtype": self.dtype_name,
            "max_new_tokens": self.max_new_tokens,
            "max_feature_tokens": self.max_feature_tokens,
            "use_cache": False,
            "generation_logits": "last_token_only_exact",
            "gpu_total_memory_gib": round(self.gpu_total_memory_gib, 3),
            "low_vram_component_swapping": self.low_vram_component_swapping,
            "missing_checkpoint_keys": self.missing_key_count,
            "unexpected_checkpoint_keys": self.unexpected_key_count,
            "do_sample": False,
        }
