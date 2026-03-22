# EchoMimic-v2 Web App Setup Guide

This guide covers the necessary steps to set up the environment and install dependencies for the EchoMimic-v2 Web App.

## 1. System Requirements

The web app requires a few system-level dependencies for audio/video processing and building extensions.

**Ubuntu/Debian:**
```bash
apt-get update && apt-get install -y ffmpeg cmake
```

## 2. Install Package Managers & Tools

### 2. UV (Python Package Manager)
Ensure you have `uv` installed. You can install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

We recommend using `uv` for fast Python dependency management, and `gdown` if you need to download models from Google Drive.

```bash
pip install gdown
```

## 3. Clone and Organize Directories

Ensure your directory structure looks like this before proceeding:

```text
parent_dir/
├── echomimic_v2/      # pipeline code + pretrained_weights/
└── web_app/           # this repository
```

*(Note: The `echomimic_v2` directory should contain the required pretrained weights. If you need to download them from Google Drive, you can use the `gdown` tool installed earlier).*

## 4. Install Project Dependencies

Navigate to the `web_app` directory and use `uv` to install the dependencies:

```bash
cd web_app/
uv sync
```

*(Alternatively, you can use `pip install -e .`)*

## 5. Optional Dependencies

For enhanced performance and features, you may want to install the following:

- **Flash Attention** (Recommended for RTX 3090/4090+ GPUs):
  ```bash
  pip install xformers
  ```
- **Wav2Vec2 Support** (Alternative audio model):
  ```bash
  pip install transformers librosa
  ```

## 6. Running the App

Once everything is installed, you can start the server

*(See the `README.md` for more advanced running configurations and performance flags).*
