#pragma once
// Internal backend interface (not part of public API).

#include "YoloDet26Api.h"

#include <memory>

class YoloDetBackendImpl
{
public:
	virtual ~YoloDetBackendImpl() = default;
	virtual int SetModel(const std::string& modelPath) = 0;
	virtual int Inference(cv::Mat image, YoloDetResult& res) = 0;
	virtual int Inference(tag_camera_data4det image, YoloDetResult& res) = 0;
	virtual int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results) = 0;
	virtual bool IsModelLoaded() const = 0;
	virtual void SetConfThreshold(float threshold) = 0;
	virtual YoloDetBackend Backend() const = 0;
};

#if defined(YOLODET26_ENABLE_GPU)
std::unique_ptr<YoloDetBackendImpl> CreateGpuBackend();
#endif

#if defined(YOLODET26_ENABLE_CPU)
std::unique_ptr<YoloDetBackendImpl> CreateCpuBackend();
#endif
