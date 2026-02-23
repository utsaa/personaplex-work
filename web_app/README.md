# EchoMimic-v2 Web App

Browser-based real-time face animation powered by the EchoMimic-v2 accelerated diffusion pipeline. Streams mic audio from the browser via WebSocket, runs it through the GPU pipeline, and streams back synchronised video frames + audio for playback.

**Now Optimized with:**
- **GPU-Parallel Batching**: Input preparation overlaps with GPU inference (zero CPU idle time).
- **Model Compilation**: `torch.compile` optimization for 20-30% faster denoising.
- **Reference Caching**: Pre-computes reference features once, eliminating redundant encoding.
- **Rolling Audio Buffer**: 1.5s context window for superior lip-sync accuracy.
- **Flash Attention**: Automatically enabled for supported GPUs (RTX 3090/4090+).
- **Latent State Preservation**: Smooth continuity between clips.
- **Wav2Vec2 Integration**: Alternative audio feature extractor for high-fidelity lip-sync.

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
- **Optional:** `transformers` & `librosa` for Wav2Vec2 support (`pip install transformers librosa`)

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

### Performance Flags
- `--compile-unet`: Enables `torch.compile` for the denoising phase.
- `--compile-unet-mode [mode]`: Choose compilation strategy (Default: `reduce-overhead`).
  - `reduce-overhead`: Balanced startup time (~60s) and high performance. Recommended.
  - `max-autotune`: Aggressive benchmarking for peak GPU performance. One-time 5-10 min delay.
  - `default`: Standard PyTorch compilation.
- `--quantize-fp8`: Enables 8-bit weight quantization via `torchao`. Reduces VRAM and boosts speed on RTX 4090/H100.

**Run Optimized (Whisper + RTX 4090 Recommended):**
```bash
uv run python app.py --port 8080 --compile-unet --compile-unet-mode reduce-overhead --quantize-fp8 --steps 4 --audio-margin 6 --use-init-latent --audio-model-type whisper
```

**Run for Wav2Vec2:**
```bash
uv run python app.py --port 8080 --audio-model-type wav2vec2 --steps 4 --audio-margin 6 --use-init-latent
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

### 7. Low RAM Mode (Stability Fix)
- **What it does:** Disables the pre-loading of pose sequences (`.npy` files) into system RAM. Instead, it reads them from disk on-the-fly during generation.
- **Benefit:** Reduces system RAM footprint by ~1GB. Essential for systems with limited RAM or when running extremely long pose sequences.
- **Flag:** `--low-ram`

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--port` | `8080` | HTTP server port |
| `--audio-margin` | `2` | **Recommended: 6**. Audio feature context margin. Higher values use more overlap for better sync. |
| `--use-init-latent` | `True` | **Recommended: Enable**. Preserves latent state for smooth continuity. |
| `--compile-unet` | `False` | **Recommended: Enable**. Compiles UNet for faster inference (adds startup delay). |
| `--quantize-fp8` | `False` | **Optional**. Quantize UNet to FP8 (requires L4/H100/4090). ~1.8x speedup. |
| `--config` | `.../infer_acc.yaml` | Pipeline config path |
| `--reference-image` | `.../therapist_ref.png` | Reference face image |
| `--pose-dir` | `.../pose/01` | Directory of `.npy` pose files |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--fps` | `24` | Output video FPS |
| `--clip-frames` | `12` | Frames per generation clip |
| `--width` | `512` | Output width |
| `--height` | `512` | Output height |
| `--steps` | `6` | Denoising steps |
| `--audio-model-type` | `whisper` | Choice between `whisper` and `wav2vec2`. |
| `--cfg` | `1.0` | Classifier-free guidance scale |
| `--port` | `8080` | HTTP server port |
| `--vad-threshold` | `0.005` | Server-side silence threshold (RMS). Clips below this are discarded. Set to `0.0` to disable. |
| `--use-init-latent` | `True` | **Enable latent state preservation** for smooth pose continuity between clips |
| `--audio-margin` | `2` | Audio feature context margin (frames). |
| `--low-ram` | `False` | **Optional**. Disable pose pre-loading to save ~1GB RAM (increases latency). |

## Audio Models

The web app now supports two different audio models for feature extraction:

1.  **Whisper (Default)**: Uses OpenAI's Whisper model (Tiny by default). Robust and handles noise well. Uses `audio_model_path` in your config.
2.  **Wav2Vec2**: Uses Facebook's Wav2Vec2 model. Often provides more detailed phonetic alignment. Uses `wav2vec2_audio_guider_path` in your config.

To switch models, use the `--audio-model-type` flag:
```bash
python app.py --audio-model-type wav2vec2
```

The key implementation files are:

- **[app.py](app.py)** — WebSocket server, generation threads, and continuity logic
- **[../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py](../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py)** — Modified pipeline with `encode_reference` caching, `audio_context_frames` trimming, and `final_latent` output.
- **[index.html](index.html)** — Browser UI with Web Audio API for microphone streaming

## Logs

All server output is logged to `web_app/logs/run_<timestamp>.log` for debugging.
