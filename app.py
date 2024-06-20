import os
import subprocess
import tempfile
import webbrowser
import cv2 as cv
import streamlit as st
from object_counting import process_video_and_count
HOME = os.getcwd()

# App title
st.set_page_config(page_title="Region Based Object Counting", page_icon="random")
st.title('Welcome to this Region Based Object Counting')

# Delete temporary file from the temp folder
run_dir = "runs/temp"
out_dir = "runs/output"
os.makedirs(run_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
for directory in [run_dir, out_dir]:
    files = os.listdir(run_dir)
    for file in files:
        file_path = os.path.join(run_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Video file uploader
uploaded_video = st.file_uploader("Upload a Video (mp4)...", type=["mp4", "mov"])

video_path = None

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=run_dir)
    temp_file.write(uploaded_video.getvalue())  # Use getvalue() to read the content of the uploaded file
    video_path = temp_file.name

    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()
    cv.imwrite("poly_zone/image/image.png", frame)
    cap.release()

    # Display video
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

# Global class_names for use across scripts
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"
               ]

# Selection of objects to detect
selected_classes = st.selectbox(
    'Select object class',

    options=class_names
)
selected_classes = selected_classes.split(",")

# Getting the Polygon Zone from the Image

url = os.getcwd() + r'\poly_zone\index.html'

if st.button('Click to get the Polygon Zone: '):
    webbrowser.open_new_tab(url)

polygons_zone = st.text_input(label="Polygon Zone")

# Convert class names to class IDs
class_ids = [class_names.index(cls) for cls in selected_classes if cls in class_names]

if uploaded_video is not None and len(selected_classes) > 0 and polygons_zone.__len__() > 0:
    with st.spinner('Processing...'):
        # Process video
        output_video_path = process_video_and_count(video_path, class_ids, polygons_zone)
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", output_video_path, "-vcodec", "libx264", out_dir+'/new.mp4']
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(HOME+f'/{output_video_path}')
        video_file = open(f'{out_dir}/new.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
