# 👁️ Smart-Eye

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Jetson_Orin_NX-76b900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-YOLO26s-ff6b35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Depth-Metric3D_ViT--Small-4d9fff?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p align="center">
  Real-time UAV detection, 3D tracking, and self-adaptive fine-tuning on edge hardware.
</p>

---

## Overview

**Smart-Eye** is an edge-deployed UAV perception system running on **NVIDIA Jetson Orin NX**. It detects and tracks micro air vehicles in real time using a single monocular camera, estimates full 3D position and velocity, and continuously adapts its detection model to new environments — fog, rain, night — without human intervention.

```
Camera → YOLO26s + ByteTrack → Metric3D ROI Depth → KalmanFilter3D → WebSocket Dashboard
                                        ↕
                          FinetuneDaemon (background thread)
                   Sample Collection → Scene Dedup → Synthesis → Fine-tune → Hot Reload
```

---

## Features

- 🎯 **Multi-target detection & tracking** — YOLO26s + ByteTrack, persistent track IDs
- 📐 **Absolute monocular depth** — Metric3D ViT-Small, ROI-only inference with adaptive padding
- 📡 **9-DOF Kalman filter** — position / velocity / acceleration per target, occlusion handling
- 🌫️ **Weather-aware scene deduplication** — multi-feature type classification + per-bucket pHash
- 🔀 **Synthetic augmentation** — alpha-blend compositing with elliptical mask + Gaussian blur
- 🧠 **Self-supervised fine-tuning** — layered learning rate + cosine annealing, idle-triggered
- ⚡ **Hot weight reload** — updated model applied without restarting inference
- 🖥️ **Live dashboard** — WebSocket frontend with video, 3D trajectory, radar, and charts

---

## System Architecture

| Component | Role | Detail |
|---|---|---|
| `YOLO26s` | Object detection | ByteTrack multi-object tracking |
| `Metric3D ViT-Small` | Monocular depth | ROI mode, adaptive padding, every-other-frame |
| `KalmanFilter3D` | State estimation | 9-dim state, occlusion propagation |
| `SceneFeatureDB` | Scene deduplication | Multi-feature type → per-type pHash bucket |
| `AlphaBlendSynthesizer` | Data augmentation | Ellipse alpha mask + Gaussian blur |
| `FinetuneExecutor` | Model adaptation | Layered LR + cosine annealing, hot-reload |
| `SocketClient` | Data transport | JPEG frames + JSON detections over TCP |
| `Dashboard` | Visualization | WebSocket: video / chart / radar / trajectory |

---

## Fine-tuning Strategy

Smart-Eye adapts automatically during idle periods (no detections for >5s).

**Layered learning rate (YOLO26s):**

| Layer Range | Type | Learning Rate |
|---|---|---|
| 0 – 4 | Shallow backbone (P1–P3) | Frozen |
| 5 – 10 | Mid backbone (P4–P5, SPPF, C2PSA) | `base_lr × 0.1` |
| 11+ | Neck + Detect26 head | `base_lr` |

- BN / bias parameters excluded from weight decay in all groups
- Cosine annealing at **step granularity** over the full run
- Optimizer: AdamW, `eps=1e-5` for FP16 stability on Jetson
- Best-loss epoch saved (not last epoch)
- New weights validated by dry-run inference before deployment

---

## Installation

```bash
# 1. Clone repo
git clone https://github.com/smarteye-project/smart-eye.git
cd smart-eye

# 2. Clone Metric3D
git clone https://github.com/YvanYin/Metric3D.git ./Metric3D

# 3. Install dependencies
pip install ultralytics mmengine opencv-python numpy --break-system-packages
```

**Metric3D ViT-Small weights** are downloaded automatically on first run.  
For offline environments, download manually and set `ckpt_file` to the local path.

---

## Quick Start

```python
from finetune_daemon import FinetuneDaemon, FinetuneDaemonConfig

# 1. Init daemon after detector
daemon_cfg = FinetuneDaemonConfig(
    base_model_path='./models/yolo26s.pt',
    device='cuda',
)
finetune_daemon = FinetuneDaemon(daemon_cfg, detector.yolo_model)
finetune_daemon.start()

# 2. In main loop, after detect_and_track()
finetune_daemon.on_detection_result(frame, drone_data, frame_count)

# 3. On exit
finetune_daemon.stop()
```

---

## Configuration

Key parameters in `FinetuneDaemonConfig`:

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | `0.85` | Pseudo-label confidence threshold |
| `idle_trigger_seconds` | `5.0` | Idle time before fine-tuning triggers |
| `min_samples_to_train` | `30` | Minimum samples required |
| `finetune_cooldown` | `300.0` | Minimum interval between runs (s) |
| `finetune_epochs` | `3` | Epochs per fine-tuning run |
| `finetune_batch_size` | `4` | Use `2` for 8GB Jetson variant |
| `finetune_lr` | `0.0001` | Base learning rate |
| `freeze_layers` | `5` | Shallow layers to freeze (YOLO26s) |
| `synthetic_ratio` | `0.4` | Target synthetic sample fraction |

---

## Output Format

Each frame emits a JSON detection payload per target:

```json
{
  "track_id": 1,
  "bbox_2d": [x1, y1, x2, y2],
  "position_3d": [X, Y, Z],
  "velocity_3d": [Vx, Vy, Vz],
  "distance": 42.3,
  "confidence": 0.91,
  "is_occluded": false,
  "predicted_trajectory": [[X1,Y1,Z1], [X2,Y2,Z2], "..."]
}
```

---

## Requirements

- NVIDIA Jetson Orin NX (16GB recommended, 8GB supported)
- JetPack 5.x+
- Python 3.8+
- PyTorch 2.0+ (JetPack-compatible build)
- Ultralytics 8.0+ (YOLO26s support)
- OpenCV 4.5+

---

## License

MIT License — see [LICENSE](LICENSE) for details.
