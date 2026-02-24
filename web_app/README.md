# EchoMimic-v2 Web App

Browser-based real-time face animation powered by the EchoMimic-v2 accelerated diffusion pipeline. Streams mic audio from the browser via WebSocket, runs it through the GPU pipeline, and streams back synchronised video frames + audio for playback.

**Now Optimized with:**
- **Dynamic Multi-GPU (N×CUDA)**: Auto-detects GPUs, loads N pipeline replicas, uses pipelined overlap-blend for parallel generation.
- **GPU-Parallel Batching**: Input preparation overlaps with GPU inference (zero CPU idle time).
- **Model Compilation**: `torch.compile` optimization for 20-30% faster denoising.
- **Reference Caching**: Pre-computes reference features once, eliminating redundant encoding.
- **Rolling Audio Buffer**: 2.0s context window for superior lip-sync accuracy.
- **Flash Attention**: Automatically enabled for supported GPUs (RTX 3090/4090+).
- **Latent State Preservation**: Smooth continuity between clips (single-GPU) or overlap blending (multi-GPU).
- **Wav2Vec2 Integration**: Alternative audio feature extractor for high-fidelity lip-sync.

## Architecture

```
Browser ──(WS: float32 PCM @ 16 kHz)──► Server ──► video_generation_thread
                                                       │
                                                  MultiGPUManager
                                                  ┌──── cuda:0 ────┐
                                                  │     cuda:1     │  (round-robin)
                                                  └── ... cuda:N ──┘
                                                       │
                                                       ▼
Browser ◄──(WS: 0x01+JPEG / 0x02+PCM)── Server ◄── frame_queue / audio_out_queue
```

- **0x01 + JPEG** — video frame
- **0x02 + float32 PCM** — audio clip (echoed back for synchronised playback)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA GPU (fp16 inference) — **RTX 3090/4090 Recommended** (multi-GPU: 2+ GPUs for parallel generation)
- The `echomimic_v2/` directory as a sibling folder containing the pipeline code and pretrained weights
- **Optional (but recommended):** `xformers` for Flash Attention (`pip install xformers`)
- **Optional:** `transformers` & `librosa` for Wav2Vec2 support (`pip install transformers librosa`)

Expected directory layout:

```
parent_dir/
├── echomimic_v2/      # pipeline code + pretrained_weights/
└── web_app/           # this package
    ├── app.py         # entry point
    ├── index.html
    ├── core/
    │   ├── gpu.py     # GPU detection, MultiGPUManager, overlap-blend
    │   ├── models.py  # pipeline loading (per-device)
    │   ├── workers.py # input prep, video gen (1..N GPU), post-processing
    │   ├── server.py  # WebSocket server + queue orchestration
    │   ├── pose.py    # pose data loading
    │   ├── audio.py   # audio utilities
    │   └── monitoring.py # GPU/CPU performance logging
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
uv run python app.py --port 8080 --compile-unet --compile-unet-mode reduce-overhead --quantize-fp8 --steps 6 --audio-margin 6 --use-init-latent --audio-model-type whisper
```

**Run for Wav2Vec2:**
```bash
uv run python app.py --port 8080 --audio-model-type wav2vec2 --steps 6 --audio-margin 6 --use-init-latent
```

Then open **http://localhost:8080** in your browser.

## Optimization Features (Comprehensive Summary)

The web app includes several advanced optimizations for low-latency, real-time performance:

### 1. Dynamic Multi-GPU (N×CUDA) ⚡
- **What it does:** Auto-detects all CUDA GPUs at startup via `MultiGPUManager`. Loads N pipeline replicas in parallel threads (one per GPU). For streaming, alternates clip generation across GPUs with K overlap frames blended via linear crossfade.
- **Single GPU (N=1):** Sequential generation with `init_latent` continuity (original behavior, zero overhead).
- **Multi GPU (N≥2):** Fire-and-forget dispatch — batches are submitted immediately to the next GPU via round-robin. Each GPU generates `clip_frames + K` frames from noise with real audio (K frames from previous batch's audio are inserted into the history). Overlap zones are crossfaded, only the last `clip_frames` are sent to the client. A `deque` of pending futures maintains strict output order.
- **First chunk:** Generates only `clip_frames` (no overlap needed). Tail K frames are saved for blending with the next chunk.
- **Audio context budget:** The 2.0s history buffer is shortened by K frames (`history = 2.0s - K/fps`), so inserting K prev-audio frames keeps total context at exactly 2.0s.
- **Benefit:** ~1.5-2× throughput for streaming, near-linear speedup for batch inference (`infer_acc.py`).
- **Flag:** `--overlap-frames` (default: 6). Set K=0 to disable blending.
- **Timeout:** Each GPU result has a 5-minute timeout (`GPU_RESULT_TIMEOUT_S = 300`).
- **Log Output:** `[GPU] Detected N CUDA GPU(s). Overlap frames (K): 6`

### 2. GPU-Parallel Batching (Latency Fix)
- **What it does:** Uses a separate thread pool to pre-process inputs (audio wav writing, pose tensor creation) for the *next* clip while the GPU is busy denoising the *current* clip.
- **Benefit:** Completely eliminates CPU idle time between GPU batches. The GPU never waits for data.
- **Log Output:** `[PREFETCH] Starting inputs for next clip...`

### 3. Model Compilation (Inference Speedup)
- **What it does:** Uses `torch.compile(mode="reduce-overhead", backend="inductor")` to JIT-compile the Denoising UNet into optimized CUDA kernels.
- **Benefit:** Reduces per-step inference time by ~20-30% (after initial warmup).
- **Flag:** `--compile-unet`

### 4. Reference Caching (Latency Fix)
- **What it does:** Pre-computes the Reference U-Net features and Self-Attention control keys *once* at startup (on all GPUs).
- **Benefit:** Removes the massive overhead of encoding the reference image for every single 0.5s clip.
- **Log Output:** `[INIT] Encoding reference on N GPU(s)...`

### 5. Rolling Audio Buffer (Lip-Sync Fix)
- **What it does:** Maintains a **2.0s rolling window** of audio history. For multi-GPU, the history is shortened by K frames (`2.0s - K/fps`) so that inserting K frames of previous audio keeps the total context budget at exactly 2.0s.
- **Benefit:** Whisper receives sufficient context to accurately predict phonetic structure, significantly improving lip-sync.
- **Flag:** `--audio-margin` controls how many frames of this context are used for alignment. Recommended: `6`.

### 6. Flash Attention (Memory/Speed)
- **What it does:** Uses `xformers` memory efficient attention if installed.
- **Benefit:** Reduces VRAM usage and speeds up attention blocks.

### 7. Latent State Preservation (Visual Continuity)
- **What it does:** In single-GPU mode, initializes each new clip using the final latent of the previous clip. In multi-GPU mode, overlap-blend crossfade handles continuity instead.
- **Benefit:** Prevents the character from "resetting" or flickering between clips.
- **Flag:** `--use-init-latent` (single-GPU only; multi-GPU uses overlap blend automatically)

### 8. Low RAM Mode (Stability Fix)
- **What it does:** Disables the pre-loading of pose sequences (`.npy` files) into system RAM.
- **Benefit:** Reduces system RAM footprint by ~1GB.
- **Flag:** `--low-ram`

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--port` | `8080` | HTTP server port |
| `--overlap-frames` | `6` | **Multi-GPU only.** Overlap frames (K) for crossfade blending between GPU chunks. |
| `--audio-margin` | `2` | **Recommended: 6**. Audio feature context margin (frames). Higher = better sync. |
| `--use-init-latent` | `True` | Latent state preservation (single-GPU continuity). Auto-disabled for multi-GPU. |
| `--compile-unet` | `False` | **Recommended**. Compiles UNet for faster inference (adds startup delay). |
| `--compile-unet-mode` | `reduce-overhead` | Compilation strategy: `default`, `reduce-overhead`, `max-autotune`. |
| `--quantize-fp8` | `False` | Quantize UNet to FP8 (requires L4/H100/4090). ~1.8x speedup. |
| `--audio-model-type` | `whisper` | Choice between `whisper` and `wav2vec2`. |
| `--config` | `.../infer_acc.yaml` | Pipeline config path |
| `--reference-image` | `.../therapist_ref.png` | Reference face image |
| `--pose-dir` | `.../pose/01` | Directory of `.npy` pose files |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--fps` | `24` | Output video FPS |
| `--clip-frames` | `12` | Frames per generation clip |
| `--width` | `512` | Output width |
| `--height` | `512` | Output height |
| `--steps` | `6` | Denoising steps |
| `--cfg` | `2.5` | Classifier-free guidance scale |
| `--vad-threshold` | `0.015` | Server-side silence threshold (RMS). Set to `0.0` to disable. |
| `--low-ram` | `False` | Disable pose pre-loading to save ~1GB RAM. |

## Audio Models

The web app now supports two different audio models for feature extraction:

1.  **Whisper (Default)**: Uses OpenAI's Whisper model (Tiny by default). Robust and handles noise well. Uses `audio_model_path` in your config.
2.  **Wav2Vec2**: Uses Facebook's Wav2Vec2 model. Often provides more detailed phonetic alignment. Uses `wav2vec2_audio_guider_path` in your config.

To switch models, use the `--audio-model-type` flag:
```bash
python app.py --audio-model-type wav2vec2
```

The key implementation files are:

- **[app.py](app.py)** — Entry point, GPU detection, pipeline init, CLI args
- **[core/gpu.py](core/gpu.py)** — `MultiGPUManager`, `detect_gpus()`, `blend_overlap()` crossfade utility
- **[core/workers.py](core/workers.py)** — Input prep, video generation (1..N GPUs), post-processing threads
- **[core/server.py](core/server.py)** — WebSocket server, queue orchestration
- **[core/models.py](core/models.py)** — Pipeline loading (per-device)
- **[../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py](../echomimic_v2/src/pipelines/pipeline_echomimicv2_acc.py)** — Modified pipeline with `encode_reference` caching, `audio_context_frames` trimming, and `final_latent` output.
- **[index.html](index.html)** — Browser UI with Web Audio API for microphone streaming

## Logs

All server output is logged to `web_app/logs/run_<timestamp>.log` for debugging.
