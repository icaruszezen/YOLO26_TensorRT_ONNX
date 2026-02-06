#include "YoloDet26BackendImpl.h"
#include "YoloDet26OpenVinoImpl.h"

#include <memory>

#if defined(YOLODET26_ENABLE_OPENVINO)
class YoloDetOpenVinoBackend final : public YoloDetBackendImpl
{
public:
	YoloDetOpenVinoBackend() : m_impl(std::make_unique<YoloDetOpenVinoImpl>()) {}

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
		return YoloDetBackend::INTEL;
	}

private:
	std::unique_ptr<YoloDetOpenVinoImpl> m_impl;
};

std::unique_ptr<YoloDetBackendImpl> CreateOpenVinoBackend()
{
	return std::make_unique<YoloDetOpenVinoBackend>();
}
#endif
