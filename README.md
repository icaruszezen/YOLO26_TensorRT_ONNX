# Yolo26_Detect_CPP_TensorRT_ONNX

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.17+-green.svg)](https://cmake.org)

English | [简体中文](./README_CN.md)

A high-performance YOLOv26 object detection inference library for C++, featuring custom TensorRT implementation with compile-time selectable CPU/GPU/OpenVINO backends.

## Features

- **Custom TensorRT Implementation**: Dedicated YOLOv26 inference engine optimized for TensorRT with custom layer support and efficient memory management
- **OpenVINO Support**: Intel-optimized inference via OpenVINO, loading ONNX models directly for accelerated execution on Intel CPUs/GPUs/NPUs
- **Compile-Time Backend Selection**: CPU, GPU and OpenVINO backends are completely isolated at compile time - choose CPU (ONNX Runtime), GPU (TensorRT/CUDA) or Intel (OpenVINO) without linking unused dependencies
- **Runtime Backend Selection**: New `SetModel(path, backend)` API allows specifying the inference backend (CPU / NVIDIA / INTEL) at runtime
- **High Performance**: Optimized inference engine with minimal overhead and GPU acceleration

## Requirements

### Common Dependencies

- CMake 3.17+
- C++17 compatible compiler (MSVC 2019+ on Windows)
- OpenCV 4.x

### CPU Backend

- ONNX Runtime 1.15+

### GPU Backend (NVIDIA)

- CUDA 11.x / 12.x
- TensorRT 8.x / 10.x

### Intel Backend (OpenVINO)

- OpenVINO 2023.0+

## Quick Start

### Model Preparation

#### 1. Export ONNX Model (for CPU backend)

Use [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) to export the YOLOv26 model to ONNX format:

```bash
yolo export model=yolo26n.pt format=onnx
```

The exported `yolo26n.onnx` file is used for CPU inference with ONNX Runtime.

#### 2. Build TensorRT Engine (for GPU backend)

Convert the ONNX model to TensorRT plan format using `trtexec`:

```bash
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.plan
```

The exported `yolo26n.plan` file is used for GPU inference with TensorRT.

### Build

```bash
# CPU build (ONNX Runtime)
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=CPU

# GPU build (TensorRT/NVIDIA)
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=GPU

# OpenVINO build (Intel)
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=OPENVINO

# All backends
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=ALL

# Build
cmake --build . --config Release
```

### Usage

```cpp
#include "YoloDet26Api.h"
#include <opencv2/opencv.hpp>

int main() {
    YoloDet detector;
    
    // Load model with explicit backend selection:
    // YoloDetBackend::CPU    - ONNX Runtime (CPU)
    // YoloDetBackend::NVIDIA - TensorRT (NVIDIA GPU)
    // YoloDetBackend::INTEL  - OpenVINO (Intel CPU/GPU/NPU)
    detector.SetModel("model.onnx", YoloDetBackend::CPU);
    // or: detector.SetModel("model.onnx", YoloDetBackend::INTEL);
    // or: detector.SetModel("model.plan", YoloDetBackend::NVIDIA);
    
    // Auto-detect backend (backward compatible):
    // .onnx -> CPU, .plan/.engine -> GPU
    // detector.SetModel("model.onnx");
    
    // Load image
    cv::Mat image = cv::imread("image.jpg");
    
    // Run inference
    YoloDetResultEx result;
    detector.Inference(image, result);
    
    // Process results
    for (int i = 0; i < result.num; ++i) {
        std::cout << "Class: " << result.classes[i]
                  << " Score: " << result.scores[i]
                  << " Box: [" << result.boxes[i].left << ", " 
                  << result.boxes[i].top << ", "
                  << result.boxes[i].right << ", "
                  << result.boxes[i].bottom << "]"
                  << std::endl;
    }
    
    return 0;
}
```

## API Reference

### Classes

#### `YoloDet`

Main detector class.

| Method | Description |
|--------|-------------|
| `YoloDet()` | Default constructor |
| `YoloDet(std::string modelPath)` | Construct with model path (auto-detect backend) |
| `YoloDet(std::string modelPath, YoloDetBackend backend)` | Construct with model path and explicit backend |
| `int SetModel(std::string modelPath)` | Load model file (auto-detect backend) |
| `int SetModel(std::string modelPath, YoloDetBackend backend)` | Load model with explicit backend (CPU/NVIDIA/INTEL) |
| `int Inference(cv::Mat image, YoloDetResult& res)` | Single image inference |
| `int Inference(cv::Mat image, YoloDetResultEx& res)` | Single image inference with timing |
| `int Inference(tag_camera_data4det image, ...)` | Raw buffer inference |
| `int InferenceBatch(...)` | Batch inference |
| `bool IsModelLoaded() const` | Check model status |
| `void SetConfThreshold(float threshold)` | Set confidence threshold (default: 0.5) |

### Structures

#### `DetBox`

Bounding box with helper methods:
- `left`, `top`, `right`, `bottom`: Coordinates
- `width()`, `height()`: Dimension getters
- `centerX()`, `centerY()`: Center point getters

#### `YoloDetResultEx`

Extended result with timing info:
- `num`: Number of detections
- `classes`: Class indices
- `scores`: Confidence scores
- `boxes`: Bounding boxes
- `backend`: Used backend (CPU/GPU/INTEL)
- `infer_time_ms`: Inference time in milliseconds

#### `YoloDetBackend`

Backend enum for selecting inference engine:
- `CPU` (0): ONNX Runtime on CPU
- `GPU` / `NVIDIA` (1): TensorRT on NVIDIA GPU
- `INTEL` (2): OpenVINO on Intel hardware

## Configuration

### CMake Options

| Option | Values | Description |
|--------|--------|-------------|
| `YOLODET26_BACKEND` | `CPU` / `GPU` / `OPENVINO` / `BOTH` / `ALL` | Select inference backend(s) |
| `TENSORRT_DIR` | Path | TensorRT root directory |
| `ONNXRUNTIME_DIR` | Path | ONNX Runtime root directory |
| `OPENVINO_DIR` | Path | OpenVINO root directory |

### Environment Variables

The build system automatically detects these environment variables:

- `TENSORRT_DIR` / `TENSORRT_ROOT`
- `ONNXRUNTIME_DIR` / `ONNXRUNTIME_ROOT`
- `CUDA_PATH` / `CUDA_TOOLKIT_ROOT_DIR`
- `OPENVINO_DIR` / `INTEL_OPENVINO_DIR`

## Testing

```bash
# Run CPU test (requires yolo26n.onnx model)
./test_cpu_bus

# Run GPU test (requires yolo26n.plan engine)
./test_gpu_bus

# Run OpenVINO test (requires yolo26n.onnx model)
./test_openvino_bus
```

## Directory Structure

```
DetectHelperCPP/
├── src/              # Source code
│   ├── cpu/          # CPU backend (ONNX Runtime)
│   ├── trt/          # GPU backend (TensorRT)
│   ├── openvino/     # Intel backend (OpenVINO)
│   └── *.h, *.cpp    # Core API implementation
├── tests/            # Test programs
├── depends/          # External dependencies (optional but recommended)
│   ├── opencv/       # OpenCV 4.x prebuilt
│   │   └── build/
│   │       ├── include/
│   │       └── x64/vc16/
│   │           ├── lib/
│   │           └── bin/
│   ├── onnxruntime/  # ONNX Runtime for CPU backend
│   │   ├── include/
│   │   └── lib/
│   │       ├── onnxruntime.dll
│   │       └── onnxruntime_providers_shared.dll
│   ├── TensorRT/     # TensorRT for GPU backend
│   │   ├── include/
│   │   └── lib/
│   │       ├── nvinfer.dll
│   │       ├── nvinfer_plugin.dll
│   │       └── nvinfer_dispatch_10.dll (TensorRT 10+)
│   ├── openvino/     # OpenVINO for Intel backend
│   │   └── runtime/
│   │       ├── include/
│   │       ├── lib/
│   │       └── bin/
│   └── models/       # Test models and images
│       ├── yolo26n.onnx
│       ├── yolo26n.plan
│       └── bus.jpg
├── CMakeLists.txt    # CMake configuration
└── README.md         # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow the existing code style and add tests for new features.
