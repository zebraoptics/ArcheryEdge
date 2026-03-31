# ArcheryEdge

An edge AI archery coaching system built on NVIDIA Jetson Orin Nano Super. ArcheryEdge uses dual CSI cameras, real-time human pose estimation, and a retrieval-augmented generation (RAG) pipeline with a locally running LLM to deliver technique feedback — entirely on-device, no cloud required.

---

## Motivation

Archery technique coaching traditionally requires either an experienced coach present in person, or recording video and reviewing it manually after the fact. Neither approach provides real-time, objective, structured feedback that an athlete can act on immediately during a training session.

ArcheryEdge aims to bridge that gap by combining computer vision and on-device AI into a portable coaching assistant that:

- Captures and analyzes an archer's form in real time using stereo cameras
- Estimates full-body pose with sub-degree joint angle resolution
- Retrieves relevant coaching knowledge and delivers actionable feedback via a local LLM
- Runs entirely on the edge — no internet connection, no cloud latency, no data privacy concerns

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  NVIDIA Jetson Orin Nano Super               │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │  CSI Cam 0   │    │        Pose Estimation           │  │
│  │  (left view) │───▶│  HRNet / ViTPose + 1€ Filter     │  │
│  └──────────────┘    │  → Joint angles & keypoints      │  │
│                      └──────────────┬───────────────────┘  │
│  ┌──────────────┐                   │                       │
│  │  CSI Cam 1   │    ┌──────────────▼───────────────────┐  │
│  │ (right view) │───▶│       Stereo 3D Reconstruction   │  │
│  └──────────────┘    │  ChArUco stereo calibration      │  │
│                      │  → 3D pose keypoints              │  │
│                      └──────────────┬───────────────────┘  │
│                                     │                       │
│                      ┌──────────────▼───────────────────┐  │
│                      │         RAG Pipeline             │  │
│                      │  ChromaDB vector store           │  │
│                      │  Qwen2.5-3B (local inference)    │  │
│                      │  → Coaching feedback             │  │
│                      └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Component | Details |
|-----------|---------|
| Edge compute | NVIDIA Jetson Orin Nano Super (8GB) |
| Cameras | Dual CSI cameras (stereo configuration) |
| Mount | Custom stereo rig with fixed baseline |
| Display | HDMI monitor or remote SSH / VNC |

---

## Software Stack

| Layer | Technology |
|-------|-----------|
| OS | JetPack (Ubuntu-based) |
| Camera interface | GStreamer + OpenCV (CSI pipeline) |
| Pose estimation | HRNet / ViTPose |
| Keypoint smoothing | 1€ filter |
| Stereo calibration | OpenCV — ChArUco-based intrinsic + stereo extrinsic |
| Vector database | ChromaDB |
| Local LLM | Qwen2.5-3B (on-device inference) |
| Knowledge ingestion | Gemini API (offline pipeline) |
| Language | Python |

---

## Pipeline Details

### 1. Camera Calibration

Stereo calibration is performed offline before deployment using a ChArUco board displayed on a screen (see [CameraCalibrationBoardAndroidApp](https://github.com/zebraoptics/CameraCalibrationBoardAndroidApp)):

- **Intrinsic calibration** — performed independently for each CSI camera (focal length, principal point, distortion coefficients)
- **Stereo extrinsic calibration** — estimates the rotation and translation between the two cameras
- Calibration data is saved and loaded at runtime for 3D reconstruction

### 2. Pose Estimation

- Each frame from both cameras is processed through HRNet or ViTPose to extract 2D body keypoints
- The **1€ filter** is applied per keypoint to smooth temporal jitter without introducing lag — critical for real-time feedback on fast movements like the draw and release
- Stereo triangulation reconstructs 3D joint positions from the calibrated camera pair

### 3. Knowledge Base & RAG Pipeline

The coaching knowledge base is built offline:

- Archery coaching content is sourced from YouTube instructional videos
- **Gemini API** transcribes and extracts structured coaching knowledge, chunked into JSON
- Chunks are embedded and indexed into **ChromaDB** as a local vector store

At inference time:

- Pose metrics (joint angles, alignment, timing) are computed from the 3D keypoints
- Relevant coaching knowledge is retrieved from ChromaDB based on the detected technique pattern
- **Qwen2.5-3B** runs locally on the Jetson to synthesize the retrieved context into natural language feedback

---

## Project Status

| Component | Status |
|-----------|--------|
| Stereo camera calibration | ✅ Complete |
| 2D pose estimation (development) | ✅ Complete (Mac) |
| 1€ filter integration | ✅ Complete |
| Knowledge ingestion pipeline | ✅ Complete |
| ChromaDB RAG pipeline | ✅ Complete |
| Qwen2.5-3B local inference | ✅ Complete |
| Jetson hardware deployment | 🔄 In progress |
| 3D stereo reconstruction | 🔄 In progress |
| End-to-end integration | 🔄 In progress |
| Real-time feedback UI | 📋 Planned |

---

## Roadmap

- [ ] Full end-to-end pipeline running on Jetson hardware
- [ ] 3D joint angle computation and biomechanical metrics
- [ ] Real-time overlay visualization of pose and feedback
- [ ] Shot detection (automatic trigger on draw/release)
- [ ] Session logging and progress tracking over time
- [ ] Expand knowledge base with additional coaching content

---

## Repository Structure

```
archeryedge/
├── calibration/           # ChArUco stereo camera calibration
├── vision/                # Camera streaming and 2D pose estimation
├── reconstruction/        # Stereo 3D reconstruction and 1€ filter
├── knowledge/             # Knowledge base: ingestion and embedding
│   └── ingestion/         # Offline Gemini pipeline (transcription → JSON)
├── rag/                   # RAG pipeline: retrieval and local LLM inference
├── analysis/              # Biomechanical metrics and shot analysis
├── feedback/              # Feedback delivery (overlay, audio, HUD)
│   └── static/            # Web assets for HUD server
├── data/                  # Runtime data (gitignored contents)
│   ├── calibration/       # Saved calibration parameters
│   ├── calibration_images/
│   │   ├── left/
│   │   └── right/
│   ├── sessions/          # Recorded session data
│   └── videos/            # Source archery footage
├── models/                # Model weights — gitignored, download separately
├── tests/                 # Test suite
├── main.py                # Entry point
├── config.yaml            # Runtime configuration
└── requirements.txt
```

---

## Related Projects

- **[opencv_contrib_android](https://github.com/zebraoptics/opencv_contrib_android)** — Pre-built OpenCV 4.12 Android SDK with all contrib modules, used for ChArUco detection on Android.
- **[CameraCalibrationBoardAndroidApp](https://github.com/zebraoptics/CameraCalibrationBoardAndroidApp)** — Android app for displaying checkerboard and ChArUco calibration patterns on screen. Used as the calibration target for ArcheryEdge's stereo camera setup.

---

## License

MIT