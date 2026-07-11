#!/bin/bash
set -e

# Ensure we are in the echomimic_v2 directory
if [ ! -d "src" ] || [ ! -d "configs" ]; then
    echo "Error: Please run this script from the root of the echomimic_v2 project."
    exit 1
fi

# Ensure git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs not found. Installing git-lfs..."
    apt-get update && apt-get install -y git-lfs
fi

echo "Starting download of EchoMimicV2 weights and dependencies..."

# 1. Main EchoMimicV2 Weights
if [ ! -d "pretrained_weights" ]; then
    echo "Cloning main EchoMimicV2 weights..."
    git lfs install
    git clone https://huggingface.co/BadToBest/EchoMimicV2 pretrained_weights
else
    echo "pretrained_weights folder already exists. Skipping main clone."
fi

# Navigate into the weights folder for the rest of the downloads
cd pretrained_weights

# 2. Whisper Model (audio_processor/tiny.pt)
echo "Downloading Whisper tiny.pt..."
mkdir -p audio_processor
if [ ! -f "audio_processor/tiny.pt" ]; then
    wget -O audio_processor/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
else
    echo "audio_processor/tiny.pt already exists. Skipping."
fi

# 3. MuseTalk Weights (audio_processor/musetalk/unet.pth)
echo "Downloading MuseTalk unet.pth..."
mkdir -p audio_processor/musetalk
if [ ! -f "audio_processor/musetalk/unet.pth" ]; then
    wget -O audio_processor/musetalk/unet.pth https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin
else
    echo "audio_processor/musetalk/unet.pth already exists. Skipping."
fi

# 4. SD-VAE-FT-MSE
echo "Cloning sd-vae-ft-mse..."
if [ ! -d "sd-vae-ft-mse" ]; then
    git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
else
    echo "sd-vae-ft-mse already exists. Skipping."
fi

# 5. SD-Image-Variations-Diffusers
echo "Cloning sd-image-variations-diffusers..."
if [ ! -d "sd-image-variations-diffusers" ]; then
    git clone https://huggingface.co/lambdalabs/sd-image-variations-diffusers
else
    echo "sd-image-variations-diffusers already exists. Skipping."
fi

echo "=========================================================="
echo "All weights have been downloaded and organized successfully!"
echo "=========================================================="
