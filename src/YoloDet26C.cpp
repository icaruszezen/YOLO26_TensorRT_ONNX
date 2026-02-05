#include "YoloDet26C.h"

#include "YoloDet26Api.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <exception>

namespace {
YoloDet* GetHandle(YoloDet26Handle handle)
{
	return reinterpret_cast<YoloDet*>(handle);
}

int BuildBgrMat(const YoloDet26Image* image, cv::Mat& bgr, cv::Mat& converted)
{
	if (!image || !image->data) {
		return -1;
	}
	if (image->width <= 0 || image->height <= 0) {
		return -1;
	}

	int type = 0;
	switch (image->channels) {
	case 1:
		type = CV_8UC1;
		break;
	case 3:
		type = CV_8UC3;
		break;
	case 4:
		type = CV_8UC4;
		break;
	default:
		return -1;
	}

	cv::Mat src(image->height, image->width, type,
		const_cast<unsigned char*>(image->data));
	if (src.empty()) {
		return -1;
	}

	if (image->channels == 3) {
		bgr = src;
		return 0;
	}
	if (image->channels == 1) {
		cv::cvtColor(src, converted, cv::COLOR_GRAY2BGR);
		bgr = converted;
		return 0;
	}
	if (image->channels == 4) {
		cv::cvtColor(src, converted, cv::COLOR_BGRA2BGR);
		bgr = converted;
		return 0;
	}

	return -1;
}

int CopyResultToC(const YoloDetResultEx& src, YoloDet26Result* dst)
{
	if (!dst) {
		return -1;
	}

	const int size_classes = static_cast<int>(src.classes.size());
	const int size_scores = static_cast<int>(src.scores.size());
	const int size_boxes = static_cast<int>(src.boxes.size());
	int total = src.num;
	total = std::min(total, size_classes);
	total = std::min(total, size_scores);
	total = std::min(total, size_boxes);
	if (total < 0) {
		return -1;
	}

	const int capacity = dst->capacity;
	const int copy_count = (capacity < total) ? capacity : total;
	dst->num = copy_count;
	dst->backend = static_cast<int>(src.backend);
	dst->infer_time_ms = src.infer_time_ms;

	if (copy_count <= 0) {
		return (total > capacity) ? 1 : 0;
	}

	if (!dst->classes || !dst->scores || !dst->boxes) {
		return -1;
	}

	for (int i = 0; i < copy_count; ++i) {
		dst->classes[i] = src.classes[i];
		dst->scores[i] = src.scores[i];
		dst->boxes[i].left = src.boxes[i].left;
		dst->boxes[i].top = src.boxes[i].top;
		dst->boxes[i].right = src.boxes[i].right;
		dst->boxes[i].bottom = src.boxes[i].bottom;
	}

	return (total > capacity) ? 1 : 0;
}
}

YoloDet26Handle YOLODET26_CALL YoloDet26_Create()
{
	try {
		return new YoloDet();
	}
	catch (...) {
		return nullptr;
	}
}

void YOLODET26_CALL YoloDet26_Destroy(YoloDet26Handle handle)
{
	delete GetHandle(handle);
}

int YOLODET26_CALL YoloDet26_SetModel(YoloDet26Handle handle, const char* model_path)
{
	if (!handle || !model_path) {
		return -1;
	}
	try {
		return GetHandle(handle)->SetModel(model_path);
	}
	catch (...) {
		return -1;
	}
}

int YOLODET26_CALL YoloDet26_Inference(YoloDet26Handle handle, const YoloDet26Image* image, YoloDet26Result* result)
{
	if (!handle || !image || !result) {
		return -1;
	}
	if (result->capacity < 0) {
		return -1;
	}
	if (result->capacity > 0 && (!result->classes || !result->scores || !result->boxes)) {
		return -1;
	}

	result->num = 0;
	result->backend = static_cast<int>(YOLODET26_BACKEND_CPU);
	result->infer_time_ms = 0.0f;

	try {
		cv::Mat bgr;
		cv::Mat converted;
		if (BuildBgrMat(image, bgr, converted) != 0) {
			return -1;
		}

		YoloDetResultEx cpp_res;
		const int ret = GetHandle(handle)->Inference(bgr, cpp_res);
		if (ret != 0) {
			return ret;
		}

		return CopyResultToC(cpp_res, result);
	}
	catch (const cv::Exception&) {
		return -1;
	}
	catch (const std::exception&) {
		return -1;
	}
	catch (...) {
		return -1;
	}
}

int YOLODET26_CALL YoloDet26_IsModelLoaded(YoloDet26Handle handle)
{
	if (!handle) {
		return 0;
	}
	try {
		return GetHandle(handle)->IsModelLoaded() ? 1 : 0;
	}
	catch (...) {
		return 0;
	}
}

void YOLODET26_CALL YoloDet26_SetConfThreshold(YoloDet26Handle handle, float threshold)
{
	if (!handle) {
		return;
	}
	try {
		GetHandle(handle)->SetConfThreshold(threshold);
	}
	catch (...) {
	}
}
