#pragma once
// OpenVINO 实现（仅供 DLL 内部使用）

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

#include <openvino/openvino.hpp>

#include "YoloDet26Api.h"
#include "YoloDet26Config.h"

class YoloDetOpenVinoImpl
{
public:
	YoloDetOpenVinoImpl();
	~YoloDetOpenVinoImpl();

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
	ov::Core m_core;
	std::shared_ptr<ov::Model> m_model;
	ov::CompiledModel m_compiled_model;
	ov::InferRequest m_infer_request;

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
