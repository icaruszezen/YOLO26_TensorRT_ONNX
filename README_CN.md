# Yolo26_Detect_CPP_TensorRT_ONNX

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.17+-green.svg)](https://cmake.org)

[English](./README.md) | 简体中文

高性能 YOLOv26 目标检测 C++ 推理库，采用自定义 TensorRT 实现，支持编译时选择 CPU/GPU 后端。

## 特性

- **自定义 TensorRT 实现**：专为 YOLOv26 优化的 TensorRT 推理引擎，支持自定义算子和高效内存管理
- **编译时后端选择**：CPU 和 GPU 后端在编译时完全隔离 —— 可任选 CPU (ONNX Runtime) 或 GPU (TensorRT/CUDA)，无需链接未使用的依赖
- **高性能**：优化的推理引擎，开销极低，支持 GPU 加速

## 环境要求

### 通用依赖

- CMake 3.17+
- C++17 兼容编译器（Windows 上 MSVC 2019+）
- OpenCV 4.x

### CPU 后端

- ONNX Runtime 1.15+

### GPU 后端

- CUDA 11.x / 12.x
- TensorRT 8.x / 10.x

## 快速开始

### 模型准备

#### 1. 导出 ONNX 模型（CPU 后端）

使用 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 将 YOLOv26 模型导出为 ONNX 格式：

```bash
yolo export model=yolo26n.pt format=onnx
```

导出的 `yolo26n.onnx` 文件用于 ONNX Runtime CPU 推理。

#### 2. 构建 TensorRT 引擎（GPU 后端）

使用 `trtexec` 将 ONNX 模型转换为 TensorRT plan 格式：

```bash
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.plan
```

导出的 `yolo26n.plan` 文件用于 TensorRT GPU 推理。

### 编译

```bash
# CPU 编译
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=CPU

# GPU 编译
mkdir build && cd build
cmake .. -DYOLODET26_BACKEND=GPU

# 构建
cmake --build . --config Release
```

### 使用示例

```cpp
#include "YoloDet26Api.h"
#include <opencv2/opencv.hpp>

int main() {
    // 加载检测器
    YoloDet detector("model.onnx");
    
    // 加载图像
    cv::Mat image = cv::imread("image.jpg");
    
    // 执行推理
    YoloDetResultEx result;
    detector.Inference(image, result);
    
    // 处理结果
    for (int i = 0; i < result.num; ++i) {
        std::cout << "类别: " << result.classes[i]
                  << " 置信度: " << result.scores[i]
                  << " 边界框: [" << result.boxes[i].left << ", " 
                  << result.boxes[i].top << ", "
                  << result.boxes[i].right << ", "
                  << result.boxes[i].bottom << "]"
                  << std::endl;
    }
    
    return 0;
}
```

## API 参考

### 类

#### `YoloDet`

主检测器类。

| 方法 | 说明 |
|------|------|
| `YoloDet()` | 默认构造函数 |
| `YoloDet(std::string modelPath)` | 带模型路径的构造函数 |
| `int SetModel(std::string modelPath)` | 加载模型文件 |
| `int Inference(cv::Mat image, YoloDetResult& res)` | 单图推理 |
| `int Inference(cv::Mat image, YoloDetResultEx& res)` | 单图推理（含耗时信息）|
| `int Inference(tag_camera_data4det image, ...)` | 原始缓冲区推理 |
| `int InferenceBatch(...)` | 批量推理 |
| `bool IsModelLoaded() const` | 检查模型加载状态 |
| `void SetConfThreshold(float threshold)` | 设置置信度阈值（默认：0.5）|

### 结构体

#### `DetBox`

边界框，包含辅助方法：
- `left`, `top`, `right`, `bottom`：坐标
- `width()`, `height()`：尺寸获取
- `centerX()`, `centerY()`：中心点获取

#### `YoloDetResultEx`

扩展结果，包含耗时信息：
- `num`：检测目标数量
- `classes`：类别索引
- `scores`：置信度分数
- `boxes`：边界框
- `backend`：使用的后端（CPU/GPU）
- `infer_time_ms`：推理耗时（毫秒）

## 配置

### CMake 选项

| 选项 | 可选值 | 说明 |
|------|--------|------|
| `YOLODET26_BACKEND` | `CPU` / `GPU` | 选择推理后端 |
| `TENSORRT_DIR` | 路径 | TensorRT 根目录 |
| `ONNXRUNTIME_DIR` | 路径 | ONNX Runtime 根目录 |

### 环境变量

构建系统自动检测以下环境变量：

- `TENSORRT_DIR` / `TENSORRT_ROOT`
- `ONNXRUNTIME_DIR` / `ONNXRUNTIME_ROOT`
- `CUDA_PATH` / `CUDA_TOOLKIT_ROOT_DIR`

## 测试

```bash
# 运行 CPU 测试（需要 yolo26n.onnx 模型）
./test_cpu_bus

# 运行 GPU 测试（需要 yolo26n.plan 引擎）
./test_gpu_bus
```

## 目录结构

```
DetectHelperCPP/
├── src/              # 源代码
│   ├── cpu/          # CPU 后端 (ONNX Runtime)
│   ├── trt/          # GPU 后端 (TensorRT)
│   └── *.h, *.cpp    # 核心 API 实现
├── tests/            # 测试程序
├── depends/          # 外部依赖（可选但推荐）
│   ├── opencv/       # OpenCV 4.x 预编译版本
│   │   └── build/
│   │       ├── include/
│   │       └── x64/vc16/
│   │           ├── lib/
│   │           └── bin/
│   ├── onnxruntime/  # ONNX Runtime（CPU 后端）
│   │   ├── include/
│   │   └── lib/
│   │       ├── onnxruntime.dll
│   │       └── onnxruntime_providers_shared.dll
│   ├── TensorRT/     # TensorRT（GPU 后端）
│   │   ├── include/
│   │   └── lib/
│   │       ├── nvinfer.dll
│   │       ├── nvinfer_plugin.dll
│   │       └── nvinfer_dispatch_10.dll (TensorRT 10+)
│   └── models/       # 测试模型和图像
│       ├── yolo26n.onnx
│       ├── yolo26n.plan
│       └── bus.jpg
├── CMakeLists.txt    # CMake 配置
└── README_CN.md      # 本文件
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](./LICENSE) 文件。

## 贡献

欢迎提交贡献！请遵循现有代码风格，并为新功能添加测试。
