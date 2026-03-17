import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
import subprocess
from omegaconf import OmegaConf

# Add echomimic_v2 to path
ECHOMIMIC_DIR = "/workspace/personaplex-work/echomimic_v2"
if ECHOMIMIC_DIR not in sys.path:
    sys.path.insert(0, ECHOMIMIC_DIR)

from src.models.unet_3d_emo import EMOUNet3DConditionModel
from core.trt.export_unet import patch_transformer_for_trt

def verify_metadata(onnx_path):
    print("\n--- Step 1: Metadata Verification ---", flush=True)
    try:
        # For models > 2GB, we should pass the path to check_model to avoid serialization errors
        onnx.checker.check_model(onnx_path)
        print("✅ Model structure is internally consistent (validated via path).", flush=True)
        
        # Load metadata only (without tensors) to print info if possible, or just load with external data
        model = onnx.load(onnx_path, load_external_data=False) 
        print(f"✅ IR Version: {model.ir_version}", flush=True)
        print(f"✅ Opset Version: {model.opset_import[0].version}", flush=True)
        
        # Check specific input shapes
        for input_node in model.graph.input:
            if input_node.name == "audio_cond_fea":
                # Note: dim_value might be 0 for dynamic axes, check dim_param
                shape = []
                for dim in input_node.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(dim.dim_param if dim.dim_param else "unk")
                
                print(f"✅ Input 'audio_cond_fea' shape: {shape}", flush=True)
                # check if 3rd dim is 130
                if len(shape) >= 3 and (shape[2] == 130 or shape[2] == 'seq_len'):
                    print("✅ Sequence length correctly identified (130/dynamic).", flush=True)
                else:
                    print(f"❌ Sequence length mismatch: {shape[2] if len(shape) >= 3 else 'N/A'}", flush=True)
        return True
    except Exception as e:
        print(f"❌ Metadata Check Failed: {e}", flush=True)
        return False

def verify_trtexec(onnx_path):
    print("\n--- Step 2: TensorRT Compatibility (trtexec) ---", flush=True)
    
    # Try multiple locations for trtexec
    possible_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        os.path.join(os.getcwd(), ".venv/bin/trtexec"),
        "trtexec" # If in PATH
    ]
    
    trtexec_path = None
    for p in possible_paths:
        if os.path.exists(p) or subprocess.run(["which", p], capture_output=True).returncode == 0:
            trtexec_path = p
            break
    
    if not trtexec_path:
        # Last ditch: search in site-packages/tensorrt_cu12
        try:
            import tensorrt_cu12
            trtexec_path = os.path.join(os.path.dirname(tensorrt_cu12.__file__), "bin/trtexec")
            if not os.path.exists(trtexec_path):
                trtexec_path = None
        except ImportError:
            pass

    if not trtexec_path:
        print("⚠️ trtexec not found. Skipping compatibility test.", flush=True)
        return True

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        "--shapes=sample:2x4x13x64x64,timestep:2,audio_cond_fea:2x13x130x384,face_musk_fea:2x320x13x64x64",
        "--fp16",
        "--skipInference",
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd)}", flush=True)
    log_file_path = "trtexec_output.log"
    try:
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            
            process.wait(timeout=300)
        
        if process.returncode == 0:
            print(f"\n✅ TensorRT parser success. Logs saved to {log_file_path}", flush=True)
            return True
        else:
            print(f"\n❌ TensorRT parser failed with return code {process.returncode}. Logs saved to {log_file_path}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        if 'process' in locals():
            process.kill()
        print(f"\n❌ trtexec timed out. Partial logs saved to {log_file_path}", flush=True)
        return False
    except Exception as e:
        print(f"❌ trtexec error: {e}", flush=True)
        return False

def verify_accuracy(onnx_path, pt_path, base_model_path):
    print("\n--- Step 3: Inference Accuracy Comparison ---", flush=True)
    
    # Load Config
    config_path = os.path.join(ECHOMIMIC_DIR, "configs", "inference", "inference_v2.yaml")
    infer_config = OmegaConf.load(config_path)

    # 1. Load PyTorch Model
    print("Loading PyTorch model...", flush=True)
    model = EMOUNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        use_safetensors=False,
    ).to(dtype=torch.float16, device="cuda")

    state_dict = torch.load(pt_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    patch_transformer_for_trt(model, video_length=13)
    model.eval()

    # 2. Load ONNX Model
    print("Loading ONNX model into ONNX Runtime (CUDA)...", flush=True)
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider'])

    # 3. Prepare Inputs
    B, F, H, W = 2, 13, 64, 64
    sample = torch.randn(B, 4, F, H, W, dtype=torch.float16, device="cuda")
    timestep = torch.tensor([1.0] * B, dtype=torch.float16, device="cuda")
    audio_cond_fea = torch.randn(B, F, 130, 384, dtype=torch.float16, device="cuda")
    face_musk_fea = torch.randn(B, 320, F, H, W, dtype=torch.float16, device="cuda")

    # 4. PyTorch Inference
    print("Running PyTorch inference...", flush=True)
    with torch.no_grad():
        pt_output = model(
            sample,
            timestep,
            encoder_hidden_states=None,
            audio_cond_fea=audio_cond_fea,
            face_musk_fea=face_musk_fea,
            return_dict=False,
        )[0]

    # 5. ONNX Inference
    print("Running ONNX inference...", flush=True)
    onnx_inputs = {
        "sample": sample.cpu().numpy(),
        "timestep": timestep.cpu().numpy(),
        "audio_cond_fea": audio_cond_fea.cpu().numpy(),
        "face_musk_fea": face_musk_fea.cpu().numpy(),
    }
    onnx_output = session.run(None, onnx_inputs)[0]
    onnx_output = torch.from_numpy(onnx_output).cuda()

    # 6. Comparison
    abs_diff = torch.abs(pt_output - onnx_output)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"Max Absolute Difference: {max_diff:.6f}", flush=True)
    print(f"Mean Absolute Difference: {mean_diff:.6f}", flush=True)

    if max_diff < 1e-2:
        print("✅ Accuracy Check Passed (Max Diff < 0.01)", flush=True)
    else:
        print("❌ Accuracy Check Failed (Max Diff >= 0.01)", flush=True)

if __name__ == "__main__":
    onnx_file = "/workspace/personaplex-work/echomimic_v2/engines/nvidia_geforce_rtx_4090/denoising_unet_acc_f12_512x512.onnx"
    pt_file = "/workspace/personaplex-work/echomimic_v2/pretrained_weights/denoising_unet_acc.pth"
    base_model = "/workspace/personaplex-work/echomimic_v2/pretrained_weights/sd-image-variations-diffusers"

    if not os.path.exists(onnx_file):
        print(f"ERROR: ONNX file not found: {onnx_file}", flush=True)
        sys.exit(1)

    m_ok = verify_metadata(onnx_file)
    t_ok = verify_trtexec(onnx_file)
    
    if m_ok:
        verify_accuracy(onnx_file, pt_file, base_model)
