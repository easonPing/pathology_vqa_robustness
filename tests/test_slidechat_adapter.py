import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

from pathology_vqa_exp2.models.slidechat import (
    SlideChatAdapter,
    _build_xformers_cutlass_attention,
    _configure_longnet_attention_backend,
    _last_token_logits_only,
)


class SlideChatCheckpointTest(unittest.TestCase):
    def test_keeps_available_slidechat_attention_backend(self):
        existing_backend = object()
        flash_module = SimpleNamespace(flash_attn_func=existing_backend)
        multihead_module = SimpleNamespace(flash_attn_func=existing_backend)

        with patch(
            "pathology_vqa_exp2.models.slidechat.importlib.import_module",
            side_effect=[flash_module, multihead_module],
        ), patch(
            "pathology_vqa_exp2.models.slidechat._build_xformers_cutlass_attention",
            side_effect=AssertionError("fallback should not be built"),
        ):
            backend_name = _configure_longnet_attention_backend()

        self.assertEqual(backend_name, "slidechat_auto")
        self.assertIs(flash_module.flash_attn_func, existing_backend)
        self.assertIs(multihead_module.flash_attn_func, existing_backend)

    def test_installs_xformers_fallback_in_both_vendor_modules(self):
        flash_module = SimpleNamespace(flash_attn_func=None)
        multihead_module = SimpleNamespace(flash_attn_func=None)
        fallback = object()

        with patch(
            "pathology_vqa_exp2.models.slidechat.importlib.import_module",
            side_effect=[flash_module, multihead_module],
        ) as import_module, patch(
            "pathology_vqa_exp2.models.slidechat._build_xformers_cutlass_attention",
            return_value=fallback,
        ):
            backend_name = _configure_longnet_attention_backend()

        self.assertEqual(backend_name, "xformers_cutlass_fallback")
        self.assertEqual(
            import_module.call_args_list,
            [
                call("torchscale.component.flash_attention"),
                call("torchscale.component.multihead_attention"),
            ],
        )
        self.assertIs(flash_module.flash_attn_func, fallback)
        self.assertIs(multihead_module.flash_attn_func, fallback)

    def test_xformers_fallback_preserves_slidechat_attention_contract(self):
        calls = []

        def forward(q, k, v, **kwargs):
            calls.append((q, k, v, kwargs))
            return "attention", "logsumexp"

        fmha = SimpleNamespace(
            memory_efficient_attention_forward_requires_grad=forward,
            cutlass=SimpleNamespace(FwOp="cutlass-forward"),
        )

        class LowerTriangularMask:
            pass

        attn_bias_module = SimpleNamespace(
            LowerTriangularMask=LowerTriangularMask
        )
        with patch(
            "pathology_vqa_exp2.models.slidechat.importlib.import_module",
            side_effect=[fmha, attn_bias_module],
        ):
            fallback = _build_xformers_cutlass_attention()
        result = fallback(
            "q",
            "k",
            "v",
            dropout=0.0,
            softmax_scale=0.25,
            is_causal=True,
        )

        self.assertEqual(result, ("attention", "logsumexp"))
        self.assertEqual(calls[0][:3], ("q", "k", "v"))
        self.assertIsInstance(calls[0][3]["attn_bias"], LowerTriangularMask)
        self.assertEqual(calls[0][3]["p"], 0.0)
        self.assertEqual(calls[0][3]["scale"], 0.25)
        self.assertEqual(calls[0][3]["op"], "cutlass-forward")

    def test_xformers_fallback_has_actionable_missing_dependency_error(self):
        with patch(
            "pathology_vqa_exp2.models.slidechat.importlib.import_module",
            side_effect=ModuleNotFoundError("xformers"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "xformers==0.0.25.post1"
            ):
                _build_xformers_cutlass_attention()

    def test_generation_output_head_projects_only_last_hidden_state(self):
        head = torch.nn.Linear(3, 5, bias=False)
        original_forward = head.forward
        hidden_states = torch.randn(2, 7, 3)

        with _last_token_logits_only(head):
            logits = head(hidden_states)

        self.assertEqual(tuple(logits.shape), (2, 1, 5))
        self.assertTrue(
            torch.equal(logits, original_forward(hidden_states[:, -1:, :]))
        )
        self.assertEqual(tuple(head(hidden_states).shape), (2, 7, 5))

    def test_extracts_deepspeed_module_state_from_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mp_rank_00_model_states.pt"
            torch.save({"module": {"weight": torch.tensor([1.0])}}, checkpoint)
            state, loaded_path = SlideChatAdapter._load_checkpoint_state(checkpoint)
            self.assertEqual(loaded_path, checkpoint)
            self.assertTrue(torch.equal(state["weight"], torch.tensor([1.0])))

    def test_directory_prefers_model_state_without_optimizer_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            model_state = checkpoint_dir / "mp_rank_00_model_states.pt"
            torch.save({"module": {"weight": torch.tensor([2.0])}}, model_state)
            state, loaded_path = SlideChatAdapter._load_checkpoint_state(
                checkpoint_dir
            )
            self.assertEqual(loaded_path, model_state)
            self.assertEqual(state["weight"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
