#include "YoloDet26BackendImpl.h"
#include "YoloDet26Impl.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>

#if defined(YOLODET26_ENABLE_GPU)
using namespace nvinfer1;
using namespace sample;

class YoloDetTrtBackend final : public YoloDetBackendImpl
{
public:
	YoloDetTrtBackend() : m_impl(std::make_unique<YoloDetImpl>()) {}

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
		return YoloDetBackend::GPU;
	}

private:
	std::unique_ptr<YoloDetImpl> m_impl;
};

std::unique_ptr<YoloDetBackendImpl> CreateGpuBackend()
{
	return std::make_unique<YoloDetTrtBackend>();
}

// ============================================================================
// YoloDetImpl 内部实现类
// ============================================================================

YoloDetImpl::YoloDetImpl()
	: m_engine_context(nullptr)
	, m_engine(nullptr)
	, m_runtime(nullptr)
	, m_model_status(false)
	, m_conf_threshold(0.50f)
{
	// 预分配主机缓冲区
	m_input_bytes = static_cast<size_t>(YOLO26_INPUT_SIZE) * YOLO26_INPUT_SIZE * 3 * sizeof(float);
	m_output_bytes = static_cast<size_t>(YOLO26_MAX_DETECTIONS) * YOLO26_OUTPUT_DIM * sizeof(float);
	m_h_input.resize(YOLO26_INPUT_SIZE * YOLO26_INPUT_SIZE * 3);
	m_h_output.resize(YOLO26_MAX_DETECTIONS * YOLO26_OUTPUT_DIM);

	// 预创建预处理缓存
	m_letterbox.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_8UC3);
	m_rgb.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_8UC3);
	m_float.create(YOLO26_INPUT_SIZE, YOLO26_INPUT_SIZE, CV_32FC3);
	m_chw.resize(3);
}

YoloDetImpl::~YoloDetImpl()
{
	release();
}

void YoloDetImpl::release()
{
	// 释放GPU缓冲区
	if (m_buffers[0] != nullptr) {
		cudaFree(m_buffers[0]);
		m_buffers[0] = nullptr;
	}
	if (m_buffers[1] != nullptr) {
		cudaFree(m_buffers[1]);
		m_buffers[1] = nullptr;
	}

	// 释放主机锁页内存
	if (m_h_input_pinned != nullptr) {
		cudaFreeHost(m_h_input_pinned);
		m_h_input_pinned = nullptr;
	}
	if (m_h_output_pinned != nullptr) {
		cudaFreeHost(m_h_output_pinned);
		m_h_output_pinned = nullptr;
	}
	m_use_pinned = false;

	// 释放CUDA流
	if (m_stream != nullptr) {
		cudaStreamDestroy(m_stream);
		m_stream = nullptr;
	}

	// 释放TensorRT资源
	if (m_engine_context != nullptr) {
		delete m_engine_context;
		m_engine_context = nullptr;
	}
	if (m_engine != nullptr) {
		delete m_engine;
		m_engine = nullptr;
	}
	if (m_runtime != nullptr) {
		delete m_runtime;
		m_runtime = nullptr;
	}

	m_model_status = false;
}

int YoloDetImpl::SetModel(std::string modelPath)
{
	try {
		// 先释放之前的资源
		release();

		// 初始化Logger
		Logger gLogger;

		// 初始化TensorRT插件
		initLibNvInferPlugins(&gLogger, "");

		// 创建Runtime
		m_runtime = createInferRuntime(gLogger);
		if (m_runtime == nullptr) {
			std::cerr << "Error: Failed to create TensorRT runtime." << std::endl;
			return -1;
		}

		// 读取模型文件
		std::ifstream file;
		file.open(modelPath, std::ios::binary | std::ios::in);
		if (!file.is_open()) {
			std::cerr << "Error: Failed to open engine file: " << modelPath << std::endl;
			return -1;
		}

		file.seekg(0, std::ios::end);
		int length = file.tellg();
		file.seekg(0, std::ios::beg);

		std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
		file.read(data.get(), length);
		file.close();

		// 反序列化引擎
		m_engine = m_runtime->deserializeCudaEngine(data.get(), length);
		if (m_engine == nullptr) {
			std::cerr << "Error: Failed to deserialize CUDA engine." << std::endl;
			return -1;
		}

#if NV_TENSORRT_MAJOR >= 10
		// TensorRT 10+ 通过 tensor name 绑定输入/输出
		m_input_tensor_name.clear();
		m_output_tensor_name.clear();
		const int nb_io = m_engine->getNbIOTensors();
		for (int i = 0; i < nb_io; ++i) {
			const char* name = m_engine->getIOTensorName(i);
			if (name == nullptr) {
				continue;
			}
			const auto mode = m_engine->getTensorIOMode(name);
			if (mode == TensorIOMode::kINPUT && m_input_tensor_name.empty()) {
				m_input_tensor_name = name;
			}
			else if (mode == TensorIOMode::kOUTPUT && m_output_tensor_name.empty()) {
				m_output_tensor_name = name;
			}
		}
		if (m_input_tensor_name.empty() || m_output_tensor_name.empty()) {
			std::cerr << "Error: Failed to resolve input/output tensor names." << std::endl;
			return -1;
		}
#endif

		// 创建执行上下文
		m_engine_context = m_engine->createExecutionContext();
		if (m_engine_context == nullptr) {
			std::cerr << "Error: Failed to create execution context." << std::endl;
			return -1;
		}

		// 分配GPU缓冲区
	cudaError_t err = cudaMalloc(&m_buffers[0], m_input_bytes);
		if (err != cudaSuccess) {
			std::cerr << "Error: Failed to allocate GPU memory for input." << std::endl;
			return -1;
		}

	err = cudaMalloc(&m_buffers[1], m_output_bytes);
		if (err != cudaSuccess) {
			cudaFree(m_buffers[0]);
			m_buffers[0] = nullptr;
			std::cerr << "Error: Failed to allocate GPU memory for output." << std::endl;
			return -1;
		}

	// 创建CUDA流（用于异步拷贝与推理）
	err = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
	if (err != cudaSuccess) {
		m_stream = nullptr;
		std::cerr << "Warning: Failed to create CUDA stream, fallback to default stream." << std::endl;
	}

	// 申请锁页内存（提升H2D/D2H拷贝效率）
	err = cudaMallocHost(reinterpret_cast<void**>(&m_h_input_pinned), m_input_bytes);
	if (err == cudaSuccess) {
		err = cudaMallocHost(reinterpret_cast<void**>(&m_h_output_pinned), m_output_bytes);
		if (err == cudaSuccess) {
			m_use_pinned = true;
		}
		else {
			cudaFreeHost(m_h_input_pinned);
			m_h_input_pinned = nullptr;
			m_h_output_pinned = nullptr;
			m_use_pinned = false;
			std::cerr << "Warning: Failed to allocate pinned output buffer, fallback to pageable memory." << std::endl;
		}
	}
	else {
		m_h_input_pinned = nullptr;
		m_h_output_pinned = nullptr;
		m_use_pinned = false;
		std::cerr << "Warning: Failed to allocate pinned input buffer, fallback to pageable memory." << std::endl;
	}

		m_model_status = true;
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		release();
		return -1;
	}
}

void YoloDetImpl::preprocess(cv::Mat& img, float data[])
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

void YoloDetImpl::postprocess(float* output, int img_width, int img_height, YoloDetResult& res)
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
	res.classes.reserve(YOLO26_MAX_DETECTIONS);
	res.scores.reserve(YOLO26_MAX_DETECTIONS);
	res.boxes.reserve(YOLO26_MAX_DETECTIONS);

	// 解析检测结果
	for (int i = 0; i < YOLO26_MAX_DETECTIONS; i++) {
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

int YoloDetImpl::Inference(cv::Mat image, YoloDetResult& res)
{
	if (!m_model_status) {
		std::cerr << "Error: Model not loaded, Set model please!" << std::endl;
		return -1;
	}

	float* host_input = m_use_pinned ? m_h_input_pinned : m_h_input.data();
	float* host_output = m_use_pinned ? m_h_output_pinned : m_h_output.data();

	// 预处理
	preprocess(image, host_input);

	// 将输入数据拷贝到GPU（异步）
	cudaError_t err = cudaMemcpyAsync(m_buffers[0], host_input,
		m_input_bytes, cudaMemcpyHostToDevice, m_stream);
	if (err != cudaSuccess) {
		std::cerr << "Error: Failed to memcpy input to GPU." << std::endl;
		return -1;
	}

	// 执行推理
#if NV_TENSORRT_MAJOR >= 10
	if (m_input_tensor_name.empty() || m_output_tensor_name.empty()) {
		std::cerr << "Error: Tensor names not initialized." << std::endl;
		return -1;
	}
	m_engine_context->setTensorAddress(m_input_tensor_name.c_str(), m_buffers[0]);
	m_engine_context->setTensorAddress(m_output_tensor_name.c_str(), m_buffers[1]);
	if (!m_engine_context->enqueueV3(m_stream)) {
		std::cerr << "Error: Failed to enqueue inference." << std::endl;
		return -1;
	}
#else
	if (!m_engine_context->enqueueV2(m_buffers, m_stream, nullptr)) {
		std::cerr << "Error: Failed to enqueue inference." << std::endl;
		return -1;
	}
#endif

	// 将输出数据拷贝回CPU（异步）
	err = cudaMemcpyAsync(host_output, m_buffers[1],
		m_output_bytes, cudaMemcpyDeviceToHost, m_stream);
	if (err != cudaSuccess) {
		std::cerr << "Error: Failed to memcpy output from GPU." << std::endl;
		return -1;
	}

	// 等待CUDA任务完成
	err = cudaStreamSynchronize(m_stream);
	if (err != cudaSuccess) {
		std::cerr << "Error: Failed to synchronize CUDA stream." << std::endl;
		return -1;
	}

	// 后处理
	postprocess(host_output, image.cols, image.rows, res);

	return 0;
}

int YoloDetImpl::Inference(tag_camera_data4det image, YoloDetResult& res)
{
	return Inference(createMatFromData(&image), res);
}

int YoloDetImpl::InferenceBatch(std::vector<cv::Mat>& images, std::vector<YoloDetResult>& results)
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

cv::Mat YoloDetImpl::createMatFromData(tag_camera_data4det* camera_Data)
{
	return createMatFromData(camera_Data->width, camera_Data->height, camera_Data->channels, camera_Data->data);
}

cv::Mat YoloDetImpl::createMatFromData(int width, int height, int channels, unsigned char* data)
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
#endif
