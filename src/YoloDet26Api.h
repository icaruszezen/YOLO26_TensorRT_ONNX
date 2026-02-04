#pragma once
// Public API header.

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

#ifdef YOLODET26_EXPORTS
#define YOLODET26_API __declspec(dllexport)
#else
#define YOLODET26_API __declspec(dllimport)
#endif

struct YOLODET26_API DetBox {
	float left;
	float top;
	float right;
	float bottom;

	DetBox() : left(0), top(0), right(0), bottom(0) {}
	DetBox(float left, float top, float right, float bottom)
		: left(left), top(top), right(right), bottom(bottom) {
	}

	float width() const { return right - left; }
	float height() const { return bottom - top; }
	float centerX() const { return (left + right) / 2.0f; }
	float centerY() const { return (top + bottom) / 2.0f; }
};

struct YOLODET26_API YoloDetResult {
	int num = 0;
	std::vector<int> classes;
	std::vector<float> scores;
	std::vector<DetBox> boxes;
};

enum class YoloDetBackend {
	CPU = 0,
	GPU = 1
};

struct YOLODET26_API YoloDetResultEx {
	int num = 0;
	std::vector<int> classes;
	std::vector<float> scores;
	std::vector<DetBox> boxes;
	YoloDetBackend backend = YoloDetBackend::CPU;
	float infer_time_ms = 0.0f;
};

struct YOLODET26_API tag_camera_data4det
{
	int width;
	int height;
	int channels;
	unsigned char* data;
};

class YoloDetBackendImpl;

#ifdef _WIN32
class YOLODET26_API YoloDet
#else
class YoloDet
#endif
{
public:
	YoloDet();
	YoloDet(std::string modelPath);
	~YoloDet();

	YoloDet(const YoloDet&) = delete;
	YoloDet& operator=(const YoloDet&) = delete;

	YoloDet(YoloDet&& other) noexcept;
	YoloDet& operator=(YoloDet&& other) noexcept;

	int SetModel(std::string modelPath);
	int Inference(cv::Mat image, YoloDetResult& res);
	int Inference(cv::Mat image, YoloDetResultEx& res);
	int Inference(tag_camera_data4det image, YoloDetResult& res);
	int Inference(tag_camera_data4det image, YoloDetResultEx& res);
	int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results);
	int InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResultEx>& results);
	bool IsModelLoaded() const;
	void SetConfThreshold(float threshold);

private:
	std::unique_ptr<YoloDetBackendImpl> m_impl;
	YoloDetBackend m_backend = YoloDetBackend::CPU;
	float m_conf_threshold = 0.50f;
};
