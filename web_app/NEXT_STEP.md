Run this command to start the server:

```bash
uv run python app.py --port 8080 --steps 12 --audio-model-type whisper --low-ram --cfg 1.0 --vad-threshold 0.005 --use-init-latent --clip-frames 48
uv run python app.py --port 8080 --steps 6 --audio-model-type whisper --low-ram --cfg 1.0 --vad-threshold 0.005 --no-use-init-latent --clip-frames 48 --debug-way
uv run python app.py --port 8080 --steps 6 --vad-threshold 0.005 --no-use-init-latent --clip-frames 149 --debug-way --low-ram --cfg 1.0 --width 768 --height 768
uv run python app.py --port 8080 --steps 6 --clip-frames 48 --low-r --cfg 1.0 --width 512 --height 512 --reference-image ../echomimic_v2/assets/refimg_aligned/aligned_therapist_512.png --sample-rate 16000 --stream-video
```
