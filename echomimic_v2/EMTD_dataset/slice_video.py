import os
import cv2
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def split_video(info):
    input_fp, star_timecode, end_timecode, output_fp = info
    try:
        # 确保输出目录存在
        if not os.path.exists(os.path.dirname(output_fp)):
            os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        cap = cv2.VideoCapture(input_fp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        s_hours, s_minutes, s_seconds = map(float, star_timecode.split(':'))
        start_second_timestamp = s_hours*3600+s_minutes*60+s_seconds+1/fps*3
        e_hours, e_minutes, e_seconds = map(float, end_timecode.split(':'))
        end_second_timestamp = e_hours*3600+e_minutes*60+e_seconds-1/fps*3

        command = ['ffmpeg', '-i', input_fp, '-ss', str(start_second_timestamp), '-to', str(end_second_timestamp), output_fp, '-y']
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print('Split {:s} successfully!'.format(output_fp))
        else:
            print("Fail to split {:s}, error info:\n{:s}".format(output_fp, result.stderr))
    except Exception as e:
        print(f"error: {e}")


if __name__ == '__main__':

    ori_video_dir = "EMTD_dataset/"
    output_dir = "temp"
    df = pd.read_csv("EMTD_dataset_original_list.txt")
    infos = np.array(df).tolist()

    input_video_paths = []
    star_timecodes = []
    end_timecodes = []
    output_video_paths = []

    times = []
    for idx, info in enumerate(infos):
        tem_url = info[0]
        tem_stime = info[1]
        tem_etime = info[2]
        tem_video_path = os.path.join(ori_video_dir, tem_url.split('v=')[1]+".mp4")
        if os.path.exists(tem_video_path):
            input_video_paths.append(tem_video_path)
            star_timecodes.append(tem_stime)
            end_timecodes.append(tem_etime)
            output_video_paths.append(os.path.join(output_dir, "{:03d}_{}.mp4".format(idx, tem_url.split('v=')[1])))

    workers = 4
    pool = Pool(processes=workers)
    for chunks_data in tqdm(pool.imap_unordered(split_video, zip(input_video_paths,star_timecodes,end_timecodes, output_video_paths))):
        None