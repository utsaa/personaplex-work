# EchoMimic-v2 Web App

Browser-based real-time face animation powered by the EchoMimic-v2 accelerated diffusion pipeline. Streams mic audio from the browser via WebSocket, runs it through the GPU pipeline, and streams back synchronised video frames + audio for playback.

**Now Optimized with:**
- **GPU-Parallel Batching**: Input preparation overlaps with GPU inference (zero CPU idle time).
- **Model Compilation**: `torch.compile` optimization for 20-30% faster denoising.
- **Reference Caching**: Pre-computes reference features once, eliminating redundant encoding.
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
- **Optional (but recommended):** `xformers` for Flash Attention (`pip install xformers`)

Expected directory layout:

```
parent_dir/
├── echomimic_v2/      # pipeline code + pretrained_weights/
└── web_app/           # this package
    ├── app.py
    └── ...
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

**Recommended Command (Optimized):**

```bash
# Using uv (from parent directory)
uv run --directory web_app python -m web_app.app \
    --port 8080 \
    --audio-margin 6 \
    --use-init-latent \
    --compile-unet
```

Or with python directly (inside `web_app/`):

```bash
python app.py --port 8080 --audio-margin 6 --use-init-latent --compile-unet
```

Then open **http://localhost:8080** in your browser.

## Optimization Features (Comprehensive Summary)

The web app includes several advanced optimizations for low-latency, real-time performance:

### 1. GPU-Parallel Batching (Latency Fix)
- **What it does:** Uses a separate thread pool to pre-process inputs (audio wav writing, pose tensor creation) for the *next* clip while the GPU is busy denoising the *current* clip.
- **Benefit:** Completely eliminates CPU idle time between GPU batches. The GPU never waits for data.
- **Log Output:** `[PREFETCH] Starting inputs for next clip...`

### 2. Model Compilation (Inference Speedup)
- **What it does:** Uses `torch.compile(mode="reduce-overhead", backend="inductor")` to JIT-compile the Denoising UNet into optimized CUDA kernels.
- **Benefit:** Reduces per-step inference time by ~20-30% (after initial warmup).
- **Flag:** `--compile-unet`

### 3. Reference Caching (Latency Fix)
- **What it does:** Pre-computes the Reference U-Net features and Self-Attention control keys *once* at startup.
- **Benefit:** Removes the massive overhead of encoding the reference image for every single 0.5s clip.
- **Log Output:** `[INIT] Caching reference UNet states...`

### 4. Rolling Audio Buffer (Lip-Sync Fix)
- **What it does:** Maintains a **1.5s rolling window** (1.0s history + 0.5s new audio) to feed Whisper.
- **Benefit:** Whisper receives sufficient context to accurately predict phonetic structure, significantly improving lip-sync.
- **Flag:** `--audio-margin` controls how many frames of this context are used for alignment. Recommended: `6`.

### 5. Flash Attention (Memory/Speed)
- **What it does:** Uses `xformers` memory efficient attention if installed.
- **Benefit:** Reduces VRAM usage and speeds up attention blocks.

### 6. Latent State Preservation (Visual Continuity)
- **What it does:** Initializes each new clip's diffusion process using the final latent state of the previous clip (blended).
- **Benefit:** Prevents the character from "resetting" or flickering between clips.
- **Flag:** `--use-init-latent`

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--port` | `8080` | HTTP server port |
| `--audio-margin` | `2` | **Recommended: 6**. Audio feature context margin. Higher values use more overlap for better sync. |
| `--use-init-latent` | `True` | **Recommended: Enable**. Preserves latent state for smooth continuity. |
| `--compile-unet` | `False` | **Recommended: Enable**. Compiles UNet for faster inference (adds startup delay). |
| `--config` | `.../infer_acc.yaml` | Pipeline config path |
| `--reference-image` | `.../therapist_ref.png` | Reference face image |
| `--pose-dir` | `.../pose/01` | Directory of `.npy` pose files |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--fps` | `24` | Output video FPS |
| `--clip-frames` | `12` | Frames per generation clip |
| `--steps` | `6` | Denoising steps |
| `--cfg` | `1.0` | Classifier-free guidance scale |

## Development

The key implementation files are:

- **[app.py](app.py)** — WebSocket server, generation threads, and continuity logic
- **[../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py](../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py)** — Modified pipeline with `encode_reference` caching, `audio_context_frames` trimming, and `final_latent` output.
- **[index.html](index.html)** — Browser UI with Web Audio API for microphone streaming

## Logs

All server output is logged to `web_app/logs/run_<timestamp>.log` for debugging.
