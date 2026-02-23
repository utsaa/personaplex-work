
import asyncio
import queue
import threading
import numpy as np
from aiohttp import web
from core.workers import (
    input_preparation_thread,
    video_generation_thread,
    postprocess_thread,
    warmup_pipeline
)

async def run_server(pipe, ref_image, pose_provider, args, index_html_path):
    print(f"[INIT] Caching reference UNet states...")
    reference_cache = pipe.encode_reference(
        ref_image, args.width, args.height, args.steps, args.cfg,
        dtype=pipe.dtype, device=pipe.device
    )
    print(f"[INIT] Reference cached.")

    if args.compile_unet:
        # Run warmup while we have context
        warmup_pipeline(pipe, ref_image, pose_provider, args, reference_cache)

    # Queues for Parallel Workers
    input_queue = queue.Queue()          # WS -> Input Prep
    prepared_queue = queue.Queue(maxsize=4) # Input Prep -> GPU
    raw_clip_queue = queue.Queue(maxsize=4) # GPU -> Post Proc
    audio_out_queue = queue.Queue(maxsize=50) # Post Proc -> WS
    frame_queue = queue.Queue(maxsize=300)    # Post Proc -> WS
    
    stop_event = threading.Event()

    # 1. Input Preparation Thread (CPU)
    input_thread = threading.Thread(
        target=input_preparation_thread,
        args=(
            input_queue, prepared_queue, stop_event,
            pose_provider,
            args.sample_rate, args.fps, args.clip_frames,
        ),
        kwargs={
            "vad_threshold": args.vad_threshold,
            "audio_margin": args.audio_margin
        },
        daemon=True
    )
    input_thread.start()

    # 2. Video Generation Thread (GPU)
    gen_thread = threading.Thread(
        target=video_generation_thread,
        args=(
            pipe, ref_image, prepared_queue, raw_clip_queue, stop_event,
            reference_cache,
            args.sample_rate, args.fps, args.clip_frames,
            args.width, args.height, args.steps, args.cfg,
        ),
        kwargs={
            "use_init_latent": args.use_init_latent,
            "audio_margin": args.audio_margin,  
        },
        daemon=True,
    )
    gen_thread.start()

    # 3. Post-processing Thread (CPU)
    post_thread = threading.Thread(
        target=postprocess_thread,
        args=(raw_clip_queue, audio_out_queue, frame_queue, stop_event),
        daemon=True,
    )
    post_thread.start()

    # Read index.html once at startup
    with open(index_html_path, "r", encoding="utf-8") as f:
        index_html = f.read()

    async def index_handler(request):
        return web.Response(text=index_html, content_type="text/html")

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("[WEB] Client connected.")

        # Clear any stale audio from previous sessions
        while not input_queue.empty():
            try:
                input_queue.get_nowait()
            except queue.Empty:
                break

        fps = args.fps

        async def send_frames():
            frame_interval = 1.0 / fps
            while not ws.closed and not stop_event.is_set():
                try:
                    sent_any = False

                    # Priority 1: drain ALL pending audio
                    while True:
                        try:
                            audio_data = audio_out_queue.get_nowait()
                            await ws.send_bytes(audio_data)
                            sent_any = True
                        except queue.Empty:
                            break

                    # Priority 2: drain ALL pending video frames
                    while True:
                        try:
                            frame_data = frame_queue.get_nowait()
                            await ws.send_bytes(frame_data)
                            sent_any = True
                        except queue.Empty:
                            break

                    if not sent_any:
                        await asyncio.sleep(frame_interval)
                except (ConnectionResetError, ConnectionError):
                    break

        send_task = asyncio.create_task(send_frames())

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    chunk = np.frombuffer(msg.data, dtype=np.float32)
                    try:
                        input_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        finally:
            send_task.cancel()
            print("[WEB] Client disconnected.")

        return ws

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print(f"[WEB] Server running at http://0.0.0.0:{args.port} (IPv4)")
    print(f"[WEB] Open http://127.0.0.1:{args.port} in your browser.\n")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[WEB] Interrupted.")
    finally:
        stop_event.set()
        # Wait for threads to finish? No, daemon threads die with main.
        # But we can try to join for clean shutdown if we want.
        # gen_thread.join(timeout=1)
        await runner.cleanup()
        print("[WEB] Done.")
