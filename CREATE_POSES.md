# EchoMimic: Reference Images vs. Driving Poses

When working with EchoMimic, it is crucial to understand that there are **two entirely different components** that must be combined to generate a video. Mixing them up will lead to distortions or errors.

## 1. The Reference Image (`.png`)
This is the static picture of the person you want to animate (e.g., `therapist_ref.png`).
- It is a single image file.
- **Requirement:** The person in the image MUST be perfectly aligned (spatially) with the default EchoMimic skeleton framework.
- **How to Align:** Use the `align_image.py` script. It detects the face and shoulders in your raw picture and shifts them into the perfect coordinates, outputting an aligned `.png`.
  
  ```bash
  uv run python align_image.py --input raw_image.png --output aligned_image.png
  ```

  **Note on DWPose Weights**: The `align_image.py` script requires DWPose models to detect the skeleton. If it crashes with a `NoSuchFile` error, you must manually download the missing weights. Run this from your `echomimic_v2` directory:
  
  ```bash
  mkdir -p pretrained_weights/DWPose
  curl -L -o pretrained_weights/DWPose/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true
  curl -L -o pretrained_weights/DWPose/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true
  ```

## 2. The Driving Poses (`.npy` files)
This is the skeleton animation (the actual movement of the head and body over time). 
- It is **not** an image. It is a folder (like `pose/02/` or `pose/therapist_g/`) containing hundreds of `.npy` data files.
- Each `.npy` file contains the math coordinates for the skeleton for a single frame of the video.

## Do you need to create new `.npy` files?

**NO (Recommended for most uses)**
If you just want to animate your custom reference picture using the existing default movements provided by EchoMimic (such as the `01` or `02` pose folders), you DO NOT need to create new `.npy` files. 
Simply align your reference image using `align_image.py`, and pass it to the server. If you request `512x512` resolution, the server will automatically shrink the existing `.npy` poses to perfectly match your aligned image!

**YES (Only for custom movements)**
If you recorded a brand new MP4 video of yourself moving around, and you want your AI avatar to copy *your exact movements* from that video, then you must extract new `.npy` poses from your MP4. 
To do this, you would use the `## Extract pose from driving video` section of the `demo.ipynb` notebook to process your MP4 into a new folder of `.npy` files.
