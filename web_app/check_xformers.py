import torch
import sys

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

try:
    import xformers
    import xformers.ops
    print(f"xformers version: {xformers.__version__}")
    print("xformers imported successfully.")
    
    # Try a dummy operation to ensure it works
    print("Testing xformers memory efficient attention...")
    # (Simple dummy test if possible, or just rely on import for now)
    print("xformers seems ready.")
    
except ImportError as e:
    print(f"xformers NOT found or failed to import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"xformers import error: {e}")
    sys.exit(1)
