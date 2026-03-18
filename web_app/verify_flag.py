import os
import sys
from unittest.mock import patch, MagicMock

# Add current dir to path to import core
sys.path.insert(0, os.path.abspath("."))

# Mock dependencies before anything imports them
torch_mock = MagicMock()
torch_mock.__path__ = [] # Essential to make it a package for sub-imports
sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.get_device_name"] = MagicMock(return_value="GeForce RTX 3090")
sys.modules["torch.onnx"] = MagicMock()

# Mock other external modules
sys.modules["omegaconf"] = MagicMock()
sys.modules["src"] = MagicMock()
sys.modules["src.models"] = MagicMock()
sys.modules["src.models.unet_3d_emo"] = MagicMock()
sys.modules["src.models.attention"] = MagicMock()

# Import manager and modules to be patched
import core.trt.manager
from core.trt.manager import TRTEngineManager
import core.trt.export_unet
import core.trt.builder



def test_flag(skip_val, expected_sim_call):
    os.environ["TRT_SKIP_SIMPLIFY"] = skip_val
    manager = TRTEngineManager(echomimic_dir="dummy_echo")
    
    # Patch where they are loaded from OR where they are used
    # If using 'from .export_unet import export_unet_to_onnx', 
    # the function is NOT in core.trt.manager namespace until called,
    # and even then it's local.
    # The most reliable way is to mock the source module if it's imported within the function.
    
    with patch("core.trt.export_unet.export_unet_to_onnx") as mock_export, \
         patch("onnx.load") as mock_load, \
         patch("onnx.save") as mock_save, \
         patch("subprocess.run") as mock_run, \
         patch("core.trt.builder.build_unet_engine") as mock_build, \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs") as mock_makedirs, \
         patch("gc.collect") as mock_gc, \
         patch("torch.cuda.empty_cache") as mock_empty_cache:
        
        def exists_side_effect(path):
            if ".engine" in path: return False
            if ".onnx" in path: return False
            return True
            
        mock_exists.side_effect = exists_side_effect
        
        # We need to make sure onnx and subprocess are mocked when imported locally
        # patch("onnx.load") etc. should work if we have sys.modules["onnx"] set up
        
        print(f"--- Running test with TRT_SKIP_SIMPLIFY={skip_val} ---")
        try:
            # We call get_unet_engine which triggers the logic
            # unet_pt_path, base_model_path=None, force_rebuild=False, ...
            manager.get_unet_engine("dummy_pt", force_rebuild=True)
        except Exception as e:
            # We might hit issues if onnx.load returns something that fails later
            # but for our purposes, we just want to see if subprocess.run was called
            print(f"Caught error: {e}")
            
        if expected_sim_call:
            if mock_run.called:
                print(f"SUCCESS: onnxsim (subprocess.run) was called.")
            else:
                print(f"FAILURE: onnxsim (subprocess.run) was NOT called.")
        else:
            if not mock_run.called:
                print(f"SUCCESS: onnxsim (subprocess.run) was NOT called.")
            else:
                print(f"FAILURE: onnxsim (subprocess.run) was called unexpectedly.")

if __name__ == "__main__":
    # Ensure onnx is in sys.modules so patch("onnx.load") works even if imported locally
    import onnx
    import subprocess
    
    test_flag("true", False)
    print()
    test_flag("false", True)
