# YOLOv8 Conditional Color Counter & Traffic Analyzer

An interactive computer vision pipeline combining **YOLOv8** real-time object detection with HSV color filtering and positional line-crossing analytics. 

This project enables users to dynamically select target object classes and filter specific color profiles via interactive mouse clicks on live or recorded video feeds.

---

## Key Features

* **Real-time Object Detection**: Powered by Ultralytics YOLOv8 (`yolov8n.pt` / `yolov8s.pt`) to detect common traffic classes (cars, buses, trucks, motorcycles, traffic lights).
* **Interactive Mouse Selection**: Double-click any bounding box in the video window to instantly lock onto that object class and sample its exact color profile.
* **HSV-based Color Filtering & Shade Detection**: Maps sampled pixels to nearest color names (`colors.csv.txt`) and computes lower/upper HSV bounds—handling wide saturation variations for whites, greys, blacks, and chromatic colors.
* **Positional Analytics & Line Counting**: Evaluates object positions relative to a virtual boundary line to track items currently crossing, overlapping, or remaining uncrossed.
* **Automated Video Export**: Processes frames with real-time UI annotations and logs them directly to an output MP4 file.

---

## Repository Structure

```text
yolov8-conditional-color-counter/
├── code/
│   ├── main.py            # Primary execution script
│   └── colors.csv.txt     # RGB color mapping database
├── video/
│   ├── roadview.mp4       # Input video feed
│   └── output.mp4         # Output video that will be generated when you will execute the code
├── yolov8n.pt             # Lightweight YOLOv8 Nano model weights
├── yolov8s.pt             # YOLOv8 Small model weights
├── requirements.txt       # Project dependencies
└── README.md
```
---

## Quick Start & Installation

(only for windows is here, for mac and linux you can get the commands of the steps)
Firstly get into the file/directory then
1. clone the Repository
>git clone https://github.com/ABDURREHMAN-BUILDS/yolov8-conditional-color-counter.git

>cd yolov8-conditional-color-counter
2. Set Up Virtual Environment
>python -m venv env
Activate the environment
>env\Scripts\activate.bat
3. Install Dependencies
>pip install ultralytics opencv-python pandas numpy
4.Run Code
>cd code
>python main.py

---






















