import tempfile
import unittest
from pathlib import Path

import torch

from pathology_vqa_exp2.models.slidechat import (
    SlideChatAdapter,
    _last_token_logits_only,
)


class SlideChatCheckpointTest(unittest.TestCase):
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
