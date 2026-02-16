import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import sys

# Mock modules before importing app
sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.utils'] = MagicMock()
sys.modules['src'] = MagicMock()
sys.modules['src.models'] = MagicMock()
sys.modules['src.models.unet_2d_condition'] = MagicMock()
sys.modules['src.models.unet_3d_emo'] = MagicMock()
sys.modules['src.models.whisper'] = MagicMock()
sys.modules['src.models.whisper.audio2feature'] = MagicMock()
sys.modules['src.pipelines'] = MagicMock()
sys.modules['src.pipelines.pipeline_echomimicv2_acc'] = MagicMock()
sys.modules['src.models.pose_encoder'] = MagicMock()
sys.modules['src.utils'] = MagicMock()
sys.modules['src.utils.dwpose_util'] = MagicMock()

# Import the function to test
from app import _prepare_clip_inputs_safe

class TestPrefetch(unittest.TestCase):
    @patch('app.wave')
    @patch('app.tempfile')
    @patch('app.prepare_pose_tensor')
    @patch('app.os')
    def test_prepare_clip_inputs(self, mock_os, mock_prep_pose, mock_tempfile, mock_wave):
        # Setup mocks
        mock_tempfile.mkstemp.return_value = (123, "/tmp/test.wav")
        mock_prep_pose.return_value = "mock_pose_tensor"
        
        # Test data
        full_audio = np.zeros(24000, dtype=np.float32) # 1.5s at 16k
        sample_rate = 16000
        pose_dir = "/tmp/poses"
        pose_files = ["p1.npy", "p2.npy"]
        pose_idx = 0
        clip_frames = 12
        W, H = 512, 512
        device = "cuda"
        dtype = "float16"
        fps = 24
        
        # Run
        wav_path, poses, next_pidx = _prepare_clip_inputs_safe(
            full_audio, sample_rate, pose_dir, pose_files, pose_idx, clip_frames,
            W, H, device, dtype, fps
        )
        
        # Verify
        self.assertEqual(wav_path, "/tmp/test.wav")
        self.assertEqual(poses, "mock_pose_tensor")
        self.assertEqual(next_pidx, 12 % 2) # (0 + 12) % 2 = 0
        
        # Verify calls
        mock_tempfile.mkstemp.assert_called_once_with(suffix=".wav")
        mock_prep_pose.assert_called_once()
        
    def test_next_pose_index_wrapping(self):
        # Test wrapping logic
        pass

if __name__ == '__main__':
    unittest.main()
