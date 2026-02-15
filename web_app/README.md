# EchoMimic-v2 Web App

Browser-based real-time face animation powered by the EchoMimic-v2 accelerated diffusion pipeline. Streams mic audio from the browser via WebSocket, runs it through the GPU pipeline, and streams back synchronised video frames + audio for playback.

## Architecture

```
Browser ──(WS: float32 PCM @ 16 kHz)──► Server ──► video_generation_thread
Browser ◄──(WS: 0x01+JPEG / 0x02+PCM)── Server ◄── frame_queue / audio_out_queue
```

- **0x01 + JPEG** — video frame
- **0x02 + float32 PCM** — audio clip (echoed back for synchronised playback)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA GPU (fp16 inference) — falls back to CPU but fp16 is not supported there
- The `echomimic_v2/` directory as a sibling folder containing the pipeline code and pretrained weights

Expected directory layout:

```
parent_dir/
├── echomimic_v2/      # pipeline code + pretrained_weights/
└── web_app/           # this package
    ├── app.py
    ├── index.html
    ├── pyproject.toml
    └── README.md
```

## Setup

```bash
cd web_app/
uv sync
```

Or with pip:

```bash
cd web_app/
pip install -e .
```

## Running

### With uv

```bash
uv run --directory web_app python app.py --port 8080
```

### With pip (after install)

```bash
cd web_app/
python app.py --port 8080
```

Then open **http://localhost:8080** in your browser.

## RunPod Deployment

1. Create a GPU pod and expose **HTTP port 8080**
2. SSH into the pod:
   ```bash
   cd /workspace
   # upload/clone echomimic_v2/ and web_app/ as siblings
   cd web_app && uv sync
   uv run python app.py --port 8080
   ```
3. Open the RunPod proxy URL in your browser:
   ```
   https://<POD_ID>-8080.proxy.runpod.net
   ```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `echomimic_v2/configs/prompts/infer_acc.yaml` | Pipeline config path |
| `--reference-image` | `echomimic_v2/assets/therapist_ref.png` | Reference face image |
| `--pose-dir` | `echomimic_v2/assets/halfbody_demo/pose/01` | Directory of `.npy` pose files |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--fps` | `24` | Output video FPS |
| `--clip-frames` | `12` | Frames per generation clip |
| `--width` | `512` | Output width |
| `--height` | `512` | Output height |
| `--steps` | `4` | Denoising steps |
| `--cfg` | `1.0` | Classifier-free guidance scale |
| `--port` | `8080` | HTTP server port |
