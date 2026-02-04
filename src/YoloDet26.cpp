#include "YoloDet26Api.h"
#include "YoloDet26BackendImpl.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <string>
#include <utility>

#if !defined(YOLODET26_ENABLE_CPU) && !defined(YOLODET26_ENABLE_GPU)
#error "YOLODET26_ENABLE_CPU or YOLODET26_ENABLE_GPU must be defined."
#endif

namespace {
std::string ToLower(std::string value)
{
	for (char& ch : value) {
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
	return value;
}

bool EndsWith(const std::string& text, const std::string& suffix)
{
	if (text.size() < suffix.size()) {
		return false;
	}
	return text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void CopyResult(const YoloDetResult& src, YoloDetResultEx& dst)
{
	dst.num = src.num;
	dst.classes = src.classes;
	dst.scores = src.scores;
	dst.boxes = src.boxes;
}

void CopyResult(const YoloDetResultEx& src, YoloDetResult& dst)
{
	dst.num = src.num;
	dst.classes = src.classes;
	dst.scores = src.scores;
	dst.boxes = src.boxes;
}
}

YoloDet::YoloDet()
{
#if defined(YOLODET26_ENABLE_GPU)
	m_backend = YoloDetBackend::GPU;
#elif defined(YOLODET26_ENABLE_CPU)
	m_backend = YoloDetBackend::CPU;
#endif
}

YoloDet::YoloDet(std::string modelPath)
	: YoloDet()
{
	SetModel(std::move(modelPath));
}

YoloDet::~YoloDet() = default;

YoloDet::YoloDet(YoloDet&& other) noexcept = default;
YoloDet& YoloDet::operator=(YoloDet&& other) noexcept = default;

int YoloDet::SetModel(std::string modelPath)
{
	const std::string lower = ToLower(modelPath);
	if (EndsWith(lower, ".onnx")) {
#if defined(YOLODET26_ENABLE_CPU)
		auto impl = CreateCpuBackend();
		if (!impl) {
			std::cerr << "Error: Failed to create CPU backend." << std::endl;
			return -1;
		}
		impl->SetConfThreshold(m_conf_threshold);
		const int ret = impl->SetModel(modelPath);
		if (ret == 0) {
			m_impl = std::move(impl);
			m_backend = YoloDetBackend::CPU;
		}
		return ret;
#else
		std::cerr << "Error: CPU backend not available in this build." << std::endl;
		return -1;
#endif
	}

	if (EndsWith(lower, ".plan") || EndsWith(lower, ".engine")) {
#if defined(YOLODET26_ENABLE_GPU)
		auto impl = CreateGpuBackend();
		if (!impl) {
			std::cerr << "Error: Failed to create GPU backend." << std::endl;
			return -1;
		}
		impl->SetConfThreshold(m_conf_threshold);
		const int ret = impl->SetModel(modelPath);
		if (ret == 0) {
			m_impl = std::move(impl);
			m_backend = YoloDetBackend::GPU;
		}
		return ret;
#else
		std::cerr << "Error: GPU backend not available in this build." << std::endl;
		return -1;
#endif
	}

	std::cerr << "Error: Unsupported model format: " << modelPath << std::endl;
	return -1;
}

int YoloDet::Inference(cv::Mat image, YoloDetResult& res)
{
	YoloDetResultEx res_ex;
	const int ret = Inference(image, res_ex);
	if (ret == 0) {
		CopyResult(res_ex, res);
	}
	return ret;
}

int YoloDet::Inference(cv::Mat image, YoloDetResultEx& res)
{
	res = YoloDetResultEx();
	res.backend = m_backend;
	if (!m_impl) {
		std::cerr << "Error: Backend not initialized." << std::endl;
		res.infer_time_ms = 0.0f;
		return -1;
	}
	auto start = std::chrono::high_resolution_clock::now();
	YoloDetResult tmp;
	const int ret = m_impl->Inference(image, tmp);

	auto end = std::chrono::high_resolution_clock::now();
	res.infer_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

	if (ret == 0) {
		CopyResult(tmp, res);
	}
	return ret;
}

int YoloDet::Inference(tag_camera_data4det image, YoloDetResult& res)
{
	YoloDetResultEx res_ex;
	const int ret = Inference(image, res_ex);
	if (ret == 0) {
		CopyResult(res_ex, res);
	}
	return ret;
}

int YoloDet::Inference(tag_camera_data4det image, YoloDetResultEx& res)
{
	res = YoloDetResultEx();
	res.backend = m_backend;
	if (!m_impl) {
		std::cerr << "Error: Backend not initialized." << std::endl;
		res.infer_time_ms = 0.0f;
		return -1;
	}
	auto start = std::chrono::high_resolution_clock::now();
	YoloDetResult tmp;
	const int ret = m_impl->Inference(image, tmp);

	auto end = std::chrono::high_resolution_clock::now();
	res.infer_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

	if (ret == 0) {
		CopyResult(tmp, res);
	}
	return ret;
}

int YoloDet::InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results)
{
	std::vector<YoloDetResultEx> results_ex;
	const int ret = InferenceBatch(images, results_ex);
	if (ret != 0) {
		return ret;
	}
	results.clear();
	results.reserve(results_ex.size());
	for (const auto& item : results_ex) {
		YoloDetResult res;
		CopyResult(item, res);
		results.push_back(std::move(res));
	}
	return 0;
}

int YoloDet::InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResultEx>& results)
{
	if (!m_impl) {
		std::cerr << "Error: Backend not initialized." << std::endl;
		results.clear();
		return -1;
	}
	results.clear();
	results.reserve(images.size());

	for (auto& image : images) {
		YoloDetResultEx res;
		const int ret = Inference(image, res);
		if (ret != 0) {
			return ret;
		}
		results.push_back(std::move(res));
	}

	return 0;
}

bool YoloDet::IsModelLoaded() const
{
	return m_impl != nullptr && m_impl->IsModelLoaded();
}

void YoloDet::SetConfThreshold(float threshold)
{
	m_conf_threshold = threshold;
	if (m_impl) {
		m_impl->SetConfThreshold(threshold);
	}
}
