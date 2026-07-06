# Force Flush Mechanism

This document describes the manual "Force Flush" feature implemented in the PersonaPlex web application pipeline.

## Overview
The "Force Flush" feature allows the user to click a button on the UI to instantly bypass the normal minimum buffer constraints and flush all generated frames directly to the browser for playback. 

When `--stream-padding` is disabled, this feature also dynamically terminates the current audio clip without adding any artificial silent padding frames at the end of the video, resulting in zero-latency early termination.

## Pipeline Flow
When a user clicks "Force Flush", the system relies on an end-to-end sentinel propagation model to ensure that the video stream is gracefully ended exactly after the user's voice clip:

1. **Frontend Request:** The user clicks the button, and the browser sends a `0x00` (text) WebSocket packet containing the string `ClientSignal.FLUSH_REQUEST`.
2. **Signal Routing (`server.py`):** The server intercepts this text signal, maps it to the `ClientSignal` enum, and pushes it directly into the backend `audio_queue`.
3. **Input Preparation (`workers.py`):** The input preparation thread processes audio chunks. Upon reading the `FLUSH_REQUEST`, it immediately processes any leftover audio in the buffer (calculating the exact frames required using `actual_clip_frames`), skipping standard padding. Next, it pushes a special `FLUSH_SENTINEL` into the `prepared_queue`.
4. **GPU Generation (`workers.py`):** The GPU sequential thread takes 20-80 seconds per video chunk. The `FLUSH_SENTINEL` queues directly behind the final chunk. Once the final frames are processed, the GPU thread drains its pending overlap states and forwards the `FLUSH_SENTINEL` to the `raw_clip_queue`.
5. **Post-Processing (`workers.py`):** The post-process thread reads the sentinel and pushes a raw byte sequence `b'\x03'` (Tag 03) into the WebSocket `frame_queue`.
6. **Delivery (`server.py`):** The server sends `b'\x03'` over the WebSocket to the browser.
7. **Bypass (`index.html`):** The browser receives Tag 03. It instantly bypasses the `MIN_BUFFER` wait (e.g., 4 seconds) and immediately executes `sendFramesToSourceBuffer()` to play whatever video it currently holds.

## Edge Cases Handled
- **Dynamic Chunk Padding:** If the buffer holds very little audio (e.g., 900 samples), Whisper's feature extractor will crash with an `IndexError` if fed an empty array. The pipeline mitigates this by calculating `actual_clip_frames = max(1, ...)` and dynamically padding the audio array with zero-silence so that Whisper always receives at least one minimum valid frame duration.
- **Consecutive Clicks:** If the user spam-clicks the button, all consecutive requests are collapsed into a single `manual_flush_requested` boolean state to avoid sending duplicate sentinels through the pipeline.
- **Latency Perception:** The GPU thread is typically backed up (e.g., taking 80s per chunk). Because the `FLUSH_SENTINEL` waits sequentially in line behind these chunks, it will only reach the frontend after the final chunk is generated. This is intentional to keep the flush synchronized precisely with the end of the video stream.
