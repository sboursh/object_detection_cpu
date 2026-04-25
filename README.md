# Realtime Object Detection on CPU (OpenCV DNN)

This project runs real-time object detection from your webcam using a pre-trained
TensorFlow SSD MobileNetV3 model through OpenCV's DNN module.

## Features

- Real-time webcam detection on CPU
- COCO class labels (loaded from `coco.names`)
- Confidence filtering and Non-Maximum Suppression (NMS)
- Bounding box + label rendering
- FPS overlay

## Project Structure

```text
real_time_object_detection_cpu/
├── README.md
└── model_data/
    ├── main.py
    ├── Detector.py
    ├── coco.names
    ├── frozen_inference_graph.pb
    └── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
```

## Requirements

- Python 3.8+
- OpenCV with DNN support (`opencv-python`)
- NumPy
- Webcam

## Installation

From the `real_time_object_detection_cpu` folder:

```bash
pip install opencv-python numpy
```

## Run

The current entry script is inside `model_data`.

From `real_time_object_detection_cpu`:

```bash
python model_data/main.py
```

Press `q` to quit the video window.

## How It Works

1. `main.py` builds paths to:
   - model config (`.pbtxt`)
   - frozen model graph (`.pb`)
   - class labels (`coco.names`)
2. `Detector` loads the model with `cv2.dnn_DetectionModel`.
3. Input preprocessing is configured:
   - size: `320x320`
   - scale: `1/127.5`
   - mean: `(127.5, 127.5, 127.5)`
   - channel swap: BGR to RGB
4. Each frame is passed through `net.detect(...)`.
5. Detections are filtered by confidence threshold.
6. NMS removes overlapping duplicate detections.
7. Final boxes/labels are drawn and shown with FPS.

## Model Info

- Backbone/detector: SSD MobileNetV3 (COCO pretrained)
- Files used:
  - `frozen_inference_graph.pb`
  - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
- Labels: `coco.names`

## Key Parameters (Current Defaults)

In `Detector.py`:

- `confThreshold = 0.5`
- `score_threshold = 0.5` (for NMS)
- `nms_threshold = 0.2`
- `setInputSize(320, 320)`

You can tune these values for speed/accuracy trade-offs.

## Notes and Limitations

- Detection quality depends on webcam quality, lighting, and object scale.
- CPU-only real-time performance varies by machine.
