#include "YoloDet26BackendImpl.h"
#include "YoloDet26OnnxImpl.h"

#include <memory>

#if defined(YOLODET26_ENABLE_CPU)
class YoloDetOnnxBackend final : public YoloDetBackendImpl
{
public:
	YoloDetOnnxBackend() : m_impl(std::make_unique<YoloDetOnnxImpl>()) {}

	int SetModel(const std::string& modelPath) override
	{
		return m_impl->SetModel(modelPath);
	}

	int Inference(cv::Mat image, YoloDetResult& res) override
	{
		return m_impl->Inference(image, res);
	}

	int Inference(tag_camera_data4det image, YoloDetResult& res) override
	{
		return m_impl->Inference(image, res);
	}

	int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results) override
	{
		return m_impl->InferenceBatch(images, results);
	}

	bool IsModelLoaded() const override
	{
		return m_impl && m_impl->IsModelLoaded();
	}

	void SetConfThreshold(float threshold) override
	{
		if (m_impl) {
			m_impl->SetConfThreshold(threshold);
		}
	}

	YoloDetBackend Backend() const override
	{
		return YoloDetBackend::CPU;
	}

private:
	std::unique_ptr<YoloDetOnnxImpl> m_impl;
};

std::unique_ptr<YoloDetBackendImpl> CreateCpuBackend()
{
	return std::make_unique<YoloDetOnnxBackend>();
}
#endif
