#include "YoloDet26OnnxImpl.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {
#ifdef _WIN32
std::wstring ToWideString(const std::string& text)
{
	if (text.empty()) {
		return std::wstring();
	}
	const int size_needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
	if (size_needed <= 0) {
		return std::wstring();
	}
	std::wstring wide(static_cast<size_t>(size_needed), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wide[0], size_needed);
	if (!wide.empty() && wide.back() == L'\0') {
		wide.pop_back();
	}
	return wide;
}
#endif
}

YoloDetOnnxImpl::YoloDetOnnxImpl()
	: m_env(ORT_LOGGING_LEVEL_WARNING, "YoloDet26")
{
	m_input_dims = { 1, 3, YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE };
	m_h_input.resize(static_cast<size_t>(YOLO26_INPUT_SIZE) * YOLO26_INPUT_SIZE * 3);

	// 预创建预处理缓存
	m_letterbox.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_8UC3);
	m_rgb.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_8UC3);
	m_float.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_32FC3);
	m_chw.resize(3);

	m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

YoloDetOnnxImpl::~YoloDetOnnxImpl()
{
	release();
}

void YoloDetOnnxImpl::release()
{
	m_session.reset();
	m_input_names.clear();
	m_output_names.clear();
	m_model_status = false;
}

int YoloDetOnnxImpl::SetModel(std::string modelPath)
{
	try {
		release();
		if (modelPath.empty()) {
			std::cerr << "Error: Empty ONNX model path." << std::endl;
			return -1;
		}

#ifdef _WIN32
		std::wstring model_path = ToWideString(modelPath);
		if (model_path.empty()) {
			std::cerr << "Error: Failed to convert ONNX model path to wide string." << std::endl;
			return -1;
		}
		m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), m_session_options);
#else
		m_session = std::make_unique<Ort::Session>(m_env, modelPath.c_str(), m_session_options);
#endif
		if (!m_session) {
			std::cerr << "Error: Failed to create ONNX Runtime session." << std::endl;
			return -1;
		}

		const size_t input_count = m_session->GetInputCount();
		const size_t output_count = m_session->GetOutputCount();
		if (input_count < 1 || output_count < 1) {
			std::cerr << "Error: ONNX model must have at least 1 input and 1 output." << std::endl;
			return -1;
		}

		m_input_names = m_session->GetInputNames();
		m_output_names = m_session->GetOutputNames();

		if (m_input_names.empty() || m_output_names.empty()) {
			std::cerr << "Error: Failed to resolve ONNX input/output names." << std::endl;
			return -1;
		}

		auto input_info = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
		const auto input_shape = input_info.GetShape();
		if (input_shape.size() == 4) {
			bool fixed_shape = true;
			for (auto dim : input_shape) {
				if (dim <= 0) {
					fixed_shape = false;
					break;
				}
			}
			if (fixed_shape) {
				if (input_shape[1] != 3 || input_shape[2] != YOLO26_INPUT_SIZE || input_shape[3] != YOLO26_INPUT_SIZE) {
					std::cerr << "Warning: ONNX input shape does not match 1x3x"
						<< YOLO26_INPUT_SIZE << "x" << YOLO26_INPUT_SIZE << std::endl;
				}
			}
		}

		m_model_status = true;
		return 0;
	}
	catch (const Ort::Exception& e) {
		std::cerr << "Error: ONNX Runtime exception: " << e.what() << std::endl;
		release();
		return -1;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		release();
		return -1;
	}
}

void YoloDetOnnxImpl::preprocess(cv::Mat& img, float data[])
{
	int w, h, x, y;
	float r_w = YOLO26_INPUT_SIZE / (img.cols * 1.0f);
	float r_h = YOLO26_INPUT_SIZE / (img.rows * 1.0f);

	if (r_h > r_w) {
		w = YOLO26_INPUT_SIZE;
		h = static_cast<int>(r_w * img.rows);
		x = 0;
		y = (YOLO26_INPUT_SIZE - h) / 2;
	}
	else {
		w = static_cast<int>(r_h * img.cols);
		h = YOLO26_INPUT_SIZE;
		x = (YOLO26_INPUT_SIZE - w) / 2;
		y = 0;
	}

	m_resized.create(h, w, CV_8UC3);
	cv::resize(img, m_resized, m_resized.size(), 0, 0, cv::INTER_LINEAR);

	m_letterbox.setTo(cv::Scalar(114, 114, 114));
	m_resized.copyTo(m_letterbox(cv::Rect(x, y, m_resized.cols, m_resized.rows)));

	// BGR转RGB并归一化到[0,1]，直接写入NCHW缓冲
	cv::cvtColor(m_letterbox, m_rgb, cv::COLOR_BGR2RGB);
	m_rgb.convertTo(m_float, CV_32FC3, 1.0f / 255.0f);

	cv::split(m_float, m_chw);
	const size_t channel_bytes = static_cast<size_t>(YOLO26_INPUT_SIZE) * YOLO26_INPUT_SIZE * sizeof(float);
	std::memcpy(data, m_chw[0].ptr<float>(), channel_bytes);
	std::memcpy(data + YOLO26_INPUT_SIZE * YOLO26_INPUT_SIZE, m_chw[1].ptr<float>(), channel_bytes);
	std::memcpy(data + 2 * YOLO26_INPUT_SIZE * YOLO26_INPUT_SIZE, m_chw[2].ptr<float>(), channel_bytes);
}

void YoloDetOnnxImpl::postprocess(float* output, size_t det_count, int img_width, int img_height, YoloDetResult& res)
{
	// 计算缩放比例和填充
	float gain = (float)YOLO26_INPUT_SIZE / std::max(img_width, img_height);
	float pad_x = (YOLO26_INPUT_SIZE - img_width * gain) / 2.0f;
	float pad_y = (YOLO26_INPUT_SIZE - img_height * gain) / 2.0f;

	// 清空结果
	res.num = 0;
	res.classes.clear();
	res.scores.clear();
	res.boxes.clear();
	res.classes.reserve(det_count);
	res.scores.reserve(det_count);
	res.boxes.reserve(det_count);

	// 解析检测结果
	for (size_t i = 0; i < det_count; i++) {
		const float* cur = output + i * YOLO26_OUTPUT_DIM;
		float score = cur[4];

		// 置信度过滤
		if (score < m_conf_threshold) {
			continue;
		}

		// 获取边界框坐标（网络输出坐标）
		float x1 = cur[0];
		float y1 = cur[1];
		float x2 = cur[2];
		float y2 = cur[3];
		int cls = static_cast<int>(cur[5]);

		// 转换回原图坐标
		float left = (x1 - pad_x) / gain;
		float top = (y1 - pad_y) / gain;
		float right = (x2 - pad_x) / gain;
		float bottom = (y2 - pad_y) / gain;

		// 边界裁剪
		left = std::max(0.0f, std::min(left, (float)img_width));
		top = std::max(0.0f, std::min(top, (float)img_height));
		right = std::max(0.0f, std::min(right, (float)img_width));
		bottom = std::max(0.0f, std::min(bottom, (float)img_height));

		// 添加到结果
		res.boxes.push_back(DetBox(left, top, right, bottom));
		res.scores.push_back(score);
		res.classes.push_back(cls);
		res.num++;
	}
}

int YoloDetOnnxImpl::Inference(cv::Mat image, YoloDetResult& res)
{
	if (!m_model_status || !m_session) {
		std::cerr << "Error: Model not loaded, Set model please!" << std::endl;
		return -1;
	}

	// 预处理
	preprocess(image, m_h_input.data());

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info, m_h_input.data(), m_h_input.size(),
		m_input_dims.data(), m_input_dims.size());

	const char* input_names[] = { m_input_names[0].c_str() };
	const char* output_names[] = { m_output_names[0].c_str() };

	auto output_tensors = m_session->Run(Ort::RunOptions{ nullptr },
		input_names, &input_tensor, 1, output_names, 1);

	if (output_tensors.empty()) {
		std::cerr << "Error: ONNX Runtime returned empty output." << std::endl;
		return -1;
	}

	auto& output_tensor = output_tensors[0];
	float* output_ptr = output_tensor.GetTensorMutableData<float>();
	const size_t output_elems = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
	if (output_elems < YOLO26_OUTPUT_DIM) {
		std::cerr << "Error: Invalid ONNX output size." << std::endl;
		return -1;
	}

	size_t det_count = output_elems / YOLO26_OUTPUT_DIM;
	if (det_count > YOLO26_MAX_DETECTIONS) {
		det_count = YOLO26_MAX_DETECTIONS;
	}
	if (output_elems % YOLO26_OUTPUT_DIM != 0) {
		std::cerr << "Warning: ONNX output size is not a multiple of "
			<< YOLO26_OUTPUT_DIM << ", extra data will be ignored." << std::endl;
	}

	// 后处理
	postprocess(output_ptr, det_count, image.cols, image.rows, res);
	return 0;
}

int YoloDetOnnxImpl::Inference(tag_camera_data4det image, YoloDetResult& res)
{
	cv::Mat mat = createMatFromData(&image);
	if (mat.empty()) {
		return -1;
	}
	return Inference(mat, res);
}

int YoloDetOnnxImpl::InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results)
{
	if (!m_model_status) {
		std::cerr << "Error: Model not loaded, Set model please!" << std::endl;
		return -1;
	}

	// 清空结果
	results.clear();
	results.reserve(images.size());

	// 逐张图片推理（当前实现不支持真正的批量推理）
	for (auto& image : images) {
		YoloDetResult res;
		int ret = Inference(image, res);
		if (ret != 0) {
			return ret;
		}
		results.push_back(std::move(res));
	}

	return 0;
}

cv::Mat YoloDetOnnxImpl::createMatFromData(tag_camera_data4det* camera_Data)
{
	return createMatFromData(camera_Data->width, camera_Data->height, camera_Data->channels, camera_Data->data);
}

cv::Mat YoloDetOnnxImpl::createMatFromData(int width, int height, int channels, unsigned char* data)
{
	// 确定图像类型
	int type;
	if (channels == 1) {
		type = CV_8UC1; // 单通道灰度图像
	}
	else if (channels == 3) {
		type = CV_8UC3; // 三通道彩色图像
	}
	else if (channels == 4) {
		type = CV_8UC4; // 四通道图像（可能包含透明度）
	}
	else {
		std::cerr << "Unsupported channel number: " << channels << std::endl;
		return cv::Mat();
	}

	// 使用数据创建cv::Mat对象
	cv::Mat image(height, width, type, data);
	return image;
}
