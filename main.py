import argparse
import subprocess
import cv2 as cv
import os

run_dir = "runs/video"
os.makedirs(run_dir, exist_ok=True)
files = os.listdir(run_dir)
for file in files:
    file_path = os.path.join(run_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)


def check_and_convert_video(video_path):
    cap = cv.VideoCapture(video_path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if width > 1920 and height > 1080:
        ffmpeg_cmd = [
            "ffmpeg", "-i",
            video_path, "-vf", "scale=1920:1080",
            "-vcodec", "libx264", "-an",
            'C:/Users/drkum/Desktop/output.mp4'
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
        except KeyboardInterrupt:
            pass
        cap.release()
    subprocess.run(["streamlit", "run", "app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Region Based Object Counting")
    parser.add_argument('--video_path', type=str, required=True, help="Video's Path File")
    parser.add_argument("--model_path", type=str, required=False, help="Model's Path")
    args = parser.parse_args()
    check_and_convert_video(args.video_path, args.model_path)
