import os
import sys
import queue
import threading
import asyncio
import numpy as np
import argparse
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

_HERE = os.path.dirname(os.path.abspath(__file__))
_ECHOMIMIC_DIR = os.path.join(os.path.dirname(_HERE), "echomimic_v2")
if _ECHOMIMIC_DIR not in sys.path:
    sys.path.insert(0, _ECHOMIMIC_DIR)

import torch
from core.gpu import MultiGPUManager, detect_gpus
from core.pose import load_pose_files, PreloadedPoseProvider, OnTheFlyPoseProvider
from core.workers import input_preparation_thread, video_generation_thread, postprocess_thread
from core.signals import ClientSignal

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("s2v_ipc")

app = FastAPI()

s2v_input_queue = queue.Queue()
s2v_prepared_queue = queue.Queue(maxsize=4)
s2v_raw_clip_queue = queue.Queue(maxsize=4)
s2v_audio_out_queue = queue.Queue(maxsize=50)
s2v_frame_queue = queue.Queue(maxsize=300)

stop_event = threading.Event()
warmup_done_event = threading.Event()
active_clips = [0]
connected_clients = set()

def init_s2v_models(args):
    logger.info("[INIT] Initializing S2V Models...")
    n_gpus = detect_gpus()
    logger.info(f"[INIT] Detected {n_gpus} CUDA GPU(s).")

    gpu_manager = MultiGPUManager(
        config_path=args.config,
        echomimic_dir=_ECHOMIMIC_DIR,
        weight_dtype=torch.float16,
        audio_model_type=args.audio_model_type,
        overlap_frames=args.overlap_frames,
        use_trt=args.use_trt,
        fp8=args.quantize_fp8,
        force_blend=args.use_blend,
        clip_frames=args.clip_frames,
        width=args.width,
        height=args.height,
        compile_unet=args.compile_unet,
    )

    if not os.path.exists(args.reference_image):
        logger.error(f"[ERROR] Not found: {args.reference_image}")
        sys.exit(1)
    ref_image = Image.open(args.reference_image).convert("RGB")
    
    gpu_manager.encode_references(ref_image, args.width, args.height, args.steps, args.cfg)
    
    pose_files = load_pose_files(args.pose_dir)
    pose_device = gpu_manager.devices[0]
    posedata_args = (args.pose_dir, pose_files, args.width, args.height, pose_device, torch.float16)
    
    pose_provider = OnTheFlyPoseProvider(*posedata_args)
    
    input_thread = threading.Thread(
        target=input_preparation_thread,
        args=(s2v_input_queue, s2v_prepared_queue, stop_event, pose_provider, args.sample_rate, args.fps, args.clip_frames, active_clips),
        kwargs={
            "vad_threshold": args.vad_threshold,
            "audio_margin": args.audio_margin,
            "overlap_frames": gpu_manager.overlap_frames,
            "audio_model_type": args.audio_model_type,
            "debug_way": args.debug_way,
            "compile_unet": args.compile_unet,
            "stream_video": args.stream_video,
            "stream_padding": args.stream_padding,
        },
        daemon=True
    )
    input_thread.start()

    gen_thread = threading.Thread(
        target=video_generation_thread,
        args=(gpu_manager, ref_image, s2v_prepared_queue, s2v_raw_clip_queue, stop_event, args.sample_rate, args.fps, args.clip_frames, args.width, args.height, args.steps, args.cfg, args.use_init_latent, args.audio_margin, active_clips, pose_provider, args, warmup_done_event),
        daemon=True,
    )
    gen_thread.start()

    post_thread = threading.Thread(
        target=postprocess_thread,
        args=(s2v_raw_clip_queue, s2v_audio_out_queue, s2v_frame_queue, stop_event, active_clips),
        daemon=True,
    )
    post_thread.start()
    logger.info("[INIT] S2V Initialization Complete.")

async def broadcast_loop():
    frame_interval = 1.0 / 24.0
    while not stop_event.is_set():
        sent_any = False
        
        while True:
            try:
                payload = s2v_audio_out_queue.get_nowait()
                for client in list(connected_clients):
                    await client.send_bytes(payload)
                sent_any = True
            except queue.Empty:
                break
                
        while True:
            try:
                payload = s2v_frame_queue.get_nowait()
                if payload == b'\x03':
                    logger.info("[S2V IPC] Sent explicit Flush Signal (Tag 03)")
                for client in list(connected_clients):
                    await client.send_bytes(payload)
                sent_any = True
            except queue.Empty:
                break
                
        if not sent_any:
            await asyncio.sleep(frame_interval)

@app.on_event("startup")
async def startup_event():
    # Parse generic S2V args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(_ECHOMIMIC_DIR, "configs", "prompts", "infer_acc.yaml"))
    parser.add_argument("--reference-image", type=str, default=os.path.join(_ECHOMIMIC_DIR, "assets", "refimg_aligned", "aligned_therapist_512.png"))
    parser.add_argument("--pose-dir", type=str, default=os.path.join(_ECHOMIMIC_DIR, "assets", "halfbody_demo", "pose", "01"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-frames", type=int, default=24) # User requested
    parser.add_argument("--width", type=int, default=512)      # User requested
    parser.add_argument("--height", type=int, default=512)     # User requested
    parser.add_argument("--steps", type=int, default=4)        # User requested
    parser.add_argument("--cfg", type=float, default=1.0)      # User requested
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vad-threshold", type=float, default=0.015)
    parser.add_argument("--use-init-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-margin", type=int, default=2)
    parser.add_argument("--compile-unet", action="store_true")
    parser.add_argument("--compile-unet-mode", type=str, default="reduce-overhead")
    parser.add_argument("--quantize-fp8", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--audio-model-type", type=str, default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--debug-way", action="store_true", default=False) # User manually edited this to False
    parser.add_argument("--overlap-frames", type=int, default=6)
    parser.add_argument("--use-trt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-blend", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stream-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-padding", action="store_true", default=False)
    parser.add_argument("--low-ram", action="store_true", default=True)   # User requested
    
    args, _ = parser.parse_known_args()
    init_s2v_models(args)
    asyncio.create_task(broadcast_loop())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down S2V workers...")
    stop_event.set()

@app.websocket("/ws/ipc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"🌐 [S2V IPC] Orchestrator connected: {websocket.client}")
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                text = message["text"]
                if text == ClientSignal.FLUSH_REQUEST.value:
                    logger.info("[S2V IPC] Received FLUSH_REQUEST")
                    s2v_input_queue.put_nowait(ClientSignal.FLUSH_REQUEST)
            elif "bytes" in message:
                bdata = message["bytes"]
                indata = np.frombuffer(bdata, dtype=np.float32)
                s2v_input_queue.put_nowait(indata)
                logger.info(f"[S2V IPC] Queued {len(indata)} audio samples.")
    except WebSocketDisconnect:
        logger.info(f"🌐 [S2V IPC] Orchestrator disconnected: {websocket.client}")
        connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"❌ [S2V IPC] Error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

if __name__ == "__main__":
    port = int(os.environ.get("S2V_PORT", 8182))
    logger.info(f"🚀 Starting S2V IPC Server on port {port}")
    uvicorn.run("ipc_server:app", host="127.0.0.1", port=port, log_level="info")
