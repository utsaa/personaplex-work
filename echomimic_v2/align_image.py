import sys
import os
import cv2
import numpy as np
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Import from EchoMimic src
from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor

def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new

    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re,cb:ce,:] = img_resize
    return img_new

def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        imh_new = max_size

        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new
        
    return imh_new, imw_new, rb, re, cb, ce

def get_pose_params(detected_poses, max_size, height, width):
    print('Extracting pose params to match target box...')
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]['num'] = num
        candidate_body = detected_pose['bodies']['candidate']
        score_body = detected_pose['bodies']['score']
        candidate_face = detected_pose['faces']
        score_face = detected_pose['faces_score']
        candidate_hand = detected_pose['hands']
        score_hand = detected_pose['hands_score']

        # face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
            detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index*18:(index+1)*18,:]
            score_body = score_body[index]
            score_hand = score_hand[(index*2):(index*2+2),:]
            candidate_hand = candidate_hand[(index*2):(index*2+2),:,:]
        else:
            score_body = score_body[0]
            
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score>0.8]

        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]

        face_pose = candidate_face
        hand_pose = candidate_hand

        h_min, h_max = np.min(face_pose[:,1]), np.max(body_pose[:7,1])
        h_ = h_max - h_min
        
        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2
        
        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)

    margin_ratio = 0.25
    h_margin = (h_max-h_min)*margin_ratio
    
    h_min = max(h_min-h_margin*0.65, 0)
    h_max = min(h_max+h_margin*0.05, 1)

    h_new = h_max - h_min
    
    h_min_real = int(h_min*height)
    h_max_real = int(h_max*height)
    mid_real = int(mid*width)
    
    height_new = h_max_real-h_min_real+1
    width_new = height_new
    w_min_real = mid_real - width_new // 2
    if w_min_real < 0:
      w_min_real = 0
      width_new = mid_real * 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(height_new, width_new, max_size)
    res = {'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce], 
           'pose_params': [w_min, w_max, h_min, h_max],
           'video_params': [h_min_real, h_max_real, w_min_real, w_max_real],
           }
    return res

def get_img_pose(img_path: str, max_size: int):
  print(f"Reading image: {img_path}")
  frame = cv2.imread(img_path)
  if frame is None:
      raise ValueError(f"Could not read image: {img_path}")
  height, width, _ = frame.shape
  short_size = min(height, width)
  resize_ratio = max(max_size / short_size, 1.0)
  frame = cv2.resize(frame, (int(resize_ratio * width), int(resize_ratio * height)))
  height, width, _ = frame.shape
  
  print("Extracting DWPose from image...")
  detected_poses = [dwprocessor(frame)]
  dwprocessor.release_memory()

  return detected_poses, height, width, frame

def save_aligned_img(ori_frame, video_params, max_size, save_path):
  h_min_real, h_max_real, w_min_real, w_max_real = video_params
  img = ori_frame[h_min_real:h_max_real,w_min_real:w_max_real,:]
  img_aligened = resize_and_pad(img, max_size=max_size)
  print(f"Aligned image shape: {img_aligened.shape}")
  
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  cv2.imwrite(save_path, img_aligened)
  print(f"Successfully saved aligned image to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Align a reference image to EchoMimic format.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input reference image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save aligned output image")
    parser.add_argument("--max-size", "-s", type=int, default=768, help="Max size of the output image (e.g. 512 or 768)")
    args = parser.parse_args()

    detected_poses, height, width, ori_frame = get_img_pose(args.input, args.max_size)
    res_params = get_pose_params(detected_poses, args.max_size, height, width)
    save_aligned_img(ori_frame, res_params['video_params'], args.max_size, args.output)

if __name__ == "__main__":
    main()
