#pragma once
// ONNX Runtime CPU 实现（仅供 DLL 内部使用）

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

#include "onnxruntime_cxx_api.h"

#include "YoloDet26Api.h"
#include "YoloDet26Config.h"

class YoloDetOnnxImpl
{
public:
	YoloDetOnnxImpl();
	~YoloDetOnnxImpl();

	int SetModel(std::string modelPath);
	int Inference(cv::Mat image, YoloDetResult& res);
	int Inference(tag_camera_data4det image, YoloDetResult& res);
	int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results);

	bool IsModelLoaded() const { return m_model_status; }
	void SetConfThreshold(float threshold) { m_conf_threshold = threshold; }

private:
	cv::Mat createMatFromData(tag_camera_data4det* camera_Data);
	cv::Mat createMatFromData(int width, int height, int channels, unsigned char* data);

	void preprocess(cv::Mat& img, float data[]);
	void postprocess(float* output, size_t det_count, int img_width, int img_height, YoloDetResult& res);

	void release();

private:
	Ort::Env m_env;
	Ort::SessionOptions m_session_options;
	std::unique_ptr<Ort::Session> m_session;
	std::vector<std::string> m_input_names;
	std::vector<std::string> m_output_names;
	std::vector<int64_t> m_input_dims;

	std::vector<float> m_h_input;

	// 预处理缓存，避免重复分配
	cv::Mat m_letterbox;
	cv::Mat m_resized;
	cv::Mat m_rgb;
	cv::Mat m_float;
	std::vector<cv::Mat> m_chw;

	bool m_model_status = false;
	float m_conf_threshold = 0.50f;
};
