# Deployment Guide: Migrating to RTX 4090 with TensorRT

This guide outlines the steps to transition your development environment from an RTX 3070 to an RTX 4090 while leveraging TensorRT for maximum performance.

## Prerequisites
1. **NVIDIA Drivers**: Ensure you have the latest Game Ready or Studio drivers installed on the 4090 machine.
2. **CUDA & cuDNN**: Matches the versions used during development (e.g., CUDA 12.x).
3. **Environment**: Sync your `uv` environment or `pip` requirements.

## Step 1: Transferring Models
- **Transfer ONNX Files**: Transfer the `.onnx` files for UNet, VAE Encoder, VAE Decoder, and Pose Encoder.
- **DO NOT transfer `.engine` files**: TensorRT engines are specific to the 3070's architecture and will fail or underperform on the 4090.

## Step 2: Engine Generation (Automated)
The application is designed with a "Build-on-Load" strategy. 
1. Run the application on the new 4090 machine:
   ```bash
   python app.py --use-rt
   ```
2. **First Run Delay**: The first run will take several minutes. The `TRTEngineManager` will:
   - Detect the RTX 4090 hardware.
   - Detect that optimized engines (UNet, VAE, Pose) are missing for this architecture.
   - Automatically compile each ONNX source into a new optimized engine for the 4090.

## Step 3: Verification
- Check the console logs for: `[TRT] Engine for RTX 4090 not found. Compiling from ONNX...`
- Once compiled, subsequent starts will be near-instantaneous as it loads the cached engine.

## Troubleshooting
- **Out of Memory**: If compilation fails, ensure no other heavy GPU processes are running. TensorRT requires significant VRAM during the "tuning" phase of compilation.
- **Version Mismatch**: Ensure the TensorRT Python library version is identical on both machines.
