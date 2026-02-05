# YoloDet26

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.17+-green.svg)](https://cmake.org)

English | [简体中文](./README_CN.md)

A high-performance YOLOv2.6 object detection inference library for C++, featuring custom TensorRT implementation with compile-time selectable CPU/GPU backends.

## Features

- **Custom TensorRT Implementation**: Dedicated YOLOv2.6 inference engine optimized for TensorRT with custom layer support and efficient memory management
- **Compile-Time Backend Selection**: CPU and GPU backends are completely isolated at compile time - choose either CPU (ONNX Runtime) or GPU (TensorRT/CUDA) without linking unused dependencies
- **High Performance**: Optimized inference engine with minimal overhead and GPU acceleration

## Requirements

### Common Dependencies

- CMake 3.17+
- C++17 compatible compiler (MSVC 2019+ on Windows)
- OpenCV 4.x

### CPU Backend

- ONNX Runtime 1.15+

### GPU Backend

- CUDA 11.x / 12.x
- TensorRT 8.x / 10.x

## Quick Start

### Model Preparation

#### 1. Export ONNX Model (for CPU backend)

Use [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) to export the YOLOv2.6 model to ONNX format:

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
# CPU build
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=CPU

# GPU build
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=GPU

# Build
cmake --build . --config Release
```

### Usage

```cpp
#include "YoloDet26Api.h"
#include <opencv2/opencv.hpp>

int main() {
    // Load detector
    YoloDet detector("model.onnx");
    
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
| `YoloDet(std::string modelPath)` | Construct with model path |
| `int SetModel(std::string modelPath)` | Load model file |
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
- `backend`: Used backend (CPU/GPU)
- `infer_time_ms`: Inference time in milliseconds

## Configuration

### CMake Options

| Option | Values | Description |
|--------|--------|-------------|
| `YOLODET26_BACKEND` | `CPU` / `GPU` | Select inference backend |
| `TENSORRT_DIR` | Path | TensorRT root directory |
| `ONNXRUNTIME_DIR` | Path | ONNX Runtime root directory |

### Environment Variables

The build system automatically detects these environment variables:

- `TENSORRT_DIR` / `TENSORRT_ROOT`
- `ONNXRUNTIME_DIR` / `ONNXRUNTIME_ROOT`
- `CUDA_PATH` / `CUDA_TOOLKIT_ROOT_DIR`

## Testing

```bash
# Run CPU test (requires yolo26n.onnx model)
./test_cpu_bus

# Run GPU test (requires yolo26n.plan engine)
./test_gpu_bus
```

## Directory Structure

```
DetectHelperCPP/
├── src/              # Source code
│   ├── cpu/          # CPU backend (ONNX Runtime)
│   ├── trt/          # GPU backend (TensorRT)
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
