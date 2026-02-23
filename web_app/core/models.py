
import os
import sys
import torch
from omegaconf import OmegaConf

# We assume `echomimic_v2` is importable. The entry point (app.py) handles sys.path
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_accelerate_available
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder

def load_pipeline(config_path: str, device: str, weight_dtype: torch.dtype, echomimic_dir: str, audio_model_type: str = "whisper") -> EchoMimicV2Pipeline:
    """Load all EchoMimic-v2 ACC models and return an assembled pipeline."""
    print(f"[INIT] Loading EchoMimic-v2 (ACC) models (Audio Type: {audio_model_type}) ...")
    config = OmegaConf.load(config_path)
    # Handle relative paths in config based on echomimic_dir
    infer_config_path = config.inference_config
    if not os.path.isabs(infer_config_path):
        infer_config_path = os.path.join(echomimic_dir, infer_config_path)
    
    infer_config = OmegaConf.load(infer_config_path)

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(echomimic_dir, path)

    # VAE
    print("  loading VAE ...")
    vae = AutoencoderKL.from_pretrained(
        _resolve(config.pretrained_vae_path), local_files_only=True, torch_dtype=weight_dtype,
    ).to(device=device, dtype=weight_dtype)

    # Reference UNet (2D)
    print("  loading reference UNet ...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        _resolve(config.pretrained_base_model_path), subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(_resolve(config.reference_unet_path), map_location="cpu"),
    )

    # Denoising UNet (3D + motion module)
    print("  loading denoising UNet (ACC) ...")
    motion_path = _resolve(config.motion_module_path)
    base_path = _resolve(config.pretrained_base_model_path)
    if os.path.exists(motion_path):
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            base_path, motion_path, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            base_path, "", subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(_resolve(config.denoising_unet_path), map_location="cpu"), strict=False,
    )

    # Pose encoder
    print("  loading pose encoder ...")
    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256),
    ).to(device=device, dtype=weight_dtype)
    pose_net.load_state_dict(
        torch.load(_resolve(config.pose_encoder_path), map_location="cpu"),
    )

    # Audio processor
    adapter_path = None
    if audio_model_type == "wav2vec2":
        # Select path from config (wav2vec2 specific vs default)
        final_audio_model_path = _resolve(getattr(config, "wav2vec2_audio_guider_path", config.audio_model_path))
        adapter_path = _resolve(getattr(config, "wav2vec2_audio_adapter_path", "./pretrained_weights/audio_processor/wav2vec2/trained_adapter.pt"))
    else:
        final_audio_model_path = _resolve(config.audio_model_path)

    print(f"  loading audio processor ({audio_model_type}) from {final_audio_model_path} ...")
    if adapter_path:
        print(f"  optional adapter path: {adapter_path}")
        
    audio_processor = load_audio_model(
        model_path=final_audio_model_path, device=device, model_type=audio_model_type, adapter_path=adapter_path
    )

    # Scheduler
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # Assemble
    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    # Enable Flash Attention / Xformers if available
    if is_accelerate_available():
        print("[INIT] Enabling xformers memory efficient attention (Flash Attention)...")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] Failed to enable xformers: {e}")
    
    print("[READY] Pipeline loaded.\n")
    return pipe
