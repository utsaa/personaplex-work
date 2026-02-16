# EchoMimic-v2 Web App

Browser-based real-time face animation powered by the EchoMimic-v2 accelerated diffusion pipeline. Streams mic audio from the browser via WebSocket, runs it through the GPU pipeline, and streams back synchronised video frames + audio for playback.

**Now Optimized with:**
- **Reference Caching**: drastically reduces latency by pre-computing reference image features.
- **Rolling Audio Buffer**: 1.5s context window for superior lip-sync accuracy.
- **Flash Attention**: Automatically enabled for supported GPUs (RTX 3090/4090+).
- **Latent State Preservation**: Smooth continuity between clips.

## Architecture

```
Browser ──(WS: float32 PCM @ 16 kHz)──► Server ──► video_generation_thread
                                                       │
                                                       ▼
Browser ◄──(WS: 0x01+JPEG / 0x02+PCM)── Server ◄── frame_queue / audio_out_queue
```

- **0x01 + JPEG** — video frame
- **0x02 + float32 PCM** — audio clip (echoed back for synchronised playback)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA GPU (fp16 inference) — **RTX 3090/4090 Recommended**
- The `echomimic_v2/` directory as a sibling folder containing the pipeline code and pretrained weights
- **Optional (but recommended):** `xformers` for Flash Attention memory efficiency (`pip install xformers`)

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

## Optimization Features (New)

These features are **enabled by default** to provide real-time performance and high-quality lip-sycn.

### 1. Reference Caching (Latency Fix)
- **What it does:** Pre-computes the Reference U-Net features and Self-Attention control keys *once* when the server starts.
- **Benefit:** Removes the massive overhead of encoding the reference image for every single 0.5s clip.
- **Log Output:** You will see `[INIT] Caching reference UNet states...` at startup.

### 2. Rolling Audio Buffer (Lip-Sync Fix)
- **What it does:** Instead of feeding raw 0.5s audio chunks to Whisper (which confuses it), we maintain a **1.5s rolling window** (1.0s history + 0.5s new audio).
- **Benefit:** Whisper gets enough context to accurately predict mouth shapes, while the pipeline intelligently trims the output to only generate video for the new 0.5s.
- **Log Output:** `[GEN] Audio Context: Rolling 1.5s window`

### 3. Flash Attention (Hardware Speedup)
- **What it does:** Uses `xformers` memory efficient attention.
- **Benefit:** Reduces VRAM usage and speeds up the denoising loop on modern GPUs.
- **Log Output:** `[INIT] Enabling xformers memory efficient attention...`

### 4. Latent State Preservation (Continuity)
- **What it does:** Uses the final latent of clip N to initialize clip N+1.
- **Benefit:** Prevents the character from "resetting" to a neutral pose every 0.5s.
- **Log Output:** `[CONTINUITY] Initializing first frame from previous clip's final latent`

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
| `--audio-margin` | `2` | Audio feature context margin (frames). |

## Development

The key implementation files are:

- **[app.py](app.py)** — WebSocket server, generation threads, and continuity logic
- **[../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py](../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py)** — Modified pipeline with `encode_reference` caching, `audio_context_frames` trimming, and `final_latent` output.
- **[index.html](index.html)** — Browser UI with Web Audio API for microphone streaming

## Logs

All server output is logged to `web_app/logs/run_<timestamp>.log` for debugging.
