#pragma once
// 内部实现头文件 - 仅供 DLL 内部使用，不对外暴露

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <cstddef>

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"

#include "YoloDet26Api.h"
#include "YoloDet26Config.h"

// YoloDet 的内部实现类
class YoloDetImpl
{
public:
	YoloDetImpl();
	~YoloDetImpl();

	int SetModel(std::string modelPath);
	int Inference(cv::Mat image, YoloDetResult& res);
	int Inference(tag_camera_data4det image, YoloDetResult& res);
	int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results);

	bool IsModelLoaded() const { return m_model_status; }
	void SetConfThreshold(float threshold) { m_conf_threshold = threshold; }

private:
	// TensorRT相关成员
	nvinfer1::IExecutionContext* m_engine_context = nullptr;
	nvinfer1::ICudaEngine* m_engine = nullptr;
	nvinfer1::IRuntime* m_runtime = nullptr;
	cudaStream_t m_stream = nullptr;

	// GPU缓冲区
	void* m_buffers[2] = { nullptr, nullptr };

	// 主机缓冲区
	std::vector<float> m_h_input;
	std::vector<float> m_h_output;
	float* m_h_input_pinned = nullptr;
	float* m_h_output_pinned = nullptr;
	bool m_use_pinned = false;
	size_t m_input_bytes = 0;
	size_t m_output_bytes = 0;

#if NV_TENSORRT_MAJOR >= 10
	// TensorRT 10+ 使用 tensor name 绑定
	std::string m_input_tensor_name;
	std::string m_output_tensor_name;
#endif

	// 预处理缓存，避免重复分配
	cv::Mat m_letterbox;
	cv::Mat m_resized;
	cv::Mat m_rgb;
	cv::Mat m_float;
	std::vector<cv::Mat> m_chw;

	// 模型状态
	bool m_model_status = false;
	float m_conf_threshold = 0.50f;

	// 内部辅助方法
	cv::Mat createMatFromData(tag_camera_data4det* camera_Data);
	cv::Mat createMatFromData(int width, int height, int channels, unsigned char* data);

	// 预处理
	void preprocess(cv::Mat& img, float data[]);

	// 后处理：将网络输出转换为检测结果
	void postprocess(float* output, int img_width, int img_height, YoloDetResult& res);

	// 释放资源
	void release();
};
