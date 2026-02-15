#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
'''
pip install youtube-dl==2020.12.12
'''
import subprocess
import pandas as pd

def download_youtube_video(video_url, output_path):
    """
    :param video_url: youtube video url
    :param output_dir: file path to save
    """
    # video_url, output_path = info
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # download command
        command = ['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', '--merge-output-format',
               'mp4', '--output', output_path , video_url]
        # subprocess.run
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            print('Download {:s} successfully!'.format(video_url))
        else:
            print("Fail to download {:s}, error info:\n{:s}".format(video_url, result.stderr))
    except Exception as e:
        print(f"error: {e}")

if __name__ == '__main__':
    df = pd.read_csv("./echomimicv2_benchmark_url+start_timecode+end_timecode.txt")
    save_dir = "ori_video_dir"
    urls = list(set(df['URL']))
    video_output_paths = [os.path.join(save_dir, url.split('v=')[1]+".mp4") for url in urls]

    for video_url, output_path in zip(urls, video_output_paths):
        download_youtube_video(video_url, output_path)
