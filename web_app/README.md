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
uv run --directory web_app python -m web_app.app --port 8080
```

### With pip (after install)

```bash
cd web_app/
python -m web_app.app --port 8080
```

Then open **http://localhost:8080** in your browser.

## RunPod Deployment

1. Create a GPU pod and expose **HTTP port 8080**
2. SSH into the pod:
   ```bash
   cd /workspace
   # upload/clone echomimic_v2/ and web_app/ as siblings
   cd web_app && uv sync
   uv run python -m web_app.app --port 8080
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
| `--steps` | `6` | Denoising steps |
| `--cfg` | `1.0` | Classifier-free guidance scale |
| `--port` | `8080` | HTTP server port |
| `--vad-threshold` | `0.005` | Server-side silence threshold (RMS). Clips below this are discarded. Set to `0.0` to disable. |
| `--use-init-latent` | `True` | **Enable latent state preservation** for smooth pose continuity between clips |
| `--no-use-init-latent` | N/A | Disable continuity (old behavior: each clip starts from random noise) |

## Features

### Latent State Preservation (Continuity)

**Enabled by default** — ensures smooth pose transitions between consecutive video clips.

**How it works:**
- Each video clip is generated in 0.5s segments (12 frames at 24 FPS)
- The final latent state from clip N is used to initialize the first frame of clip N+1
- This eliminates the "reset to idle pose" issue where the character would jump back to a neutral position every 0.5 seconds

**Usage:**

```bash
# NEW BEHAVIOR (default): Smooth continuity
uv run python -m web_app.app --port 8080

# OLD BEHAVIOR: Discontinuous (for comparison)
uv run python -m web_app.app --port 8080 --no-use-init-latent
```

**Visual difference:**
- **With continuity (--use-init-latent):** Natural, flowing motion across clip boundaries
- **Without continuity (--no-use-init-latent):** Character "resets" to idle pose every 0.5s

Console output will show:
```
[CONTINUITY] Latent state preservation ENABLED    # with --use-init-latent (default)
[CONTINUITY] Latent state preservation DISABLED (old behavior)  # with --no-use-init-latent
```

After the second clip onwards, you'll see:
```
[CONTINUITY] Initializing first frame from previous clip's final latent
```

## Development

The key implementation files are:

- **[app.py](app.py)** — WebSocket server, generation threads, and continuity logic
- **[../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py](../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py)** — Modified pipeline with `init_latents` parameter and `final_latent` output
- **[index.html](index.html)** — Browser UI with Web Audio API for microphone streaming

## Logs

All server output is logged to `web_app/logs/run_<timestamp>.log` for debugging.
