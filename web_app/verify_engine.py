import tensorrt as trt
import os

engine_file = "/workspace/personaplex-work/echomimic_v2/engines/nvidia_geforce_rtx_4090/denoising_unet_acc_f18.engine"

if not os.path.exists(engine_file):
    print(f"❌ ERROR: File not found: {engine_file}")
    exit(1)

logger = trt.Logger(trt.Logger.INFO)
with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
    try:
        print(f"Reading {engine_file}...")
        engine_data = f.read()
        print(f"Deserializing ({len(engine_data)} bytes)...")
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine:
            print("✅ SUCCESS: Engine loaded perfectly!")
            print(f"Device memory required: {engine.device_memory_size / 1024**2:.2f} MiB")
        else:
            print("❌ FAILURE: Engine could not be deserialized.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
