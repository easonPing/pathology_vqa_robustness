import tempfile
import unittest
from pathlib import Path

import torch

from pathology_vqa_exp2.models.slidechat import SlideChatAdapter


class SlideChatCheckpointTest(unittest.TestCase):
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
