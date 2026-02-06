// OpenVINO 测试：使用 yolo26n.onnx 对 bus.jpg 做一次推理并输出检测结果。
// 用法: test_openvino_bus [model_path] [image_path]
// 默认: depends/models/yolo26n.onnx, depends/models/bus.jpg
//       先按当前工作目录查找，若不存在则按可执行文件所在目录上溯到项目根再找 depends/models

#include "YoloDet26Api.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

// 返回可执行文件所在目录（末尾带 / 或 \\）
static std::string getExeDir()
{
#ifdef _WIN32
	wchar_t buf[MAX_PATH];
	DWORD n = GetModuleFileNameW(NULL, buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH) return "";
	std::string path;
	for (int i = 0; buf[i]; ++i) path += static_cast<char>(buf[i] <= 127 ? buf[i] : '?');
	std::size_t last = path.find_last_of("/\\");
	if (last != std::string::npos) return path.substr(0, last + 1);
	return path + "\\";
#else
	// Linux 等可后续用 /proc/self/exe
	return "";
#endif
}

// 若 path 不存在，尝试 exeDir/../../path（从 build/Debug 上溯到项目根）
static std::string resolveDependsPath(const std::string& path, const std::string& exeDir)
{
	if (exeDir.empty()) return path;
	std::string candidate = exeDir + "../../" + path;
	std::ifstream f(candidate);
	if (f.good()) return candidate;
	return path;
}

static bool fileExists(const std::string& path)
{
	std::ifstream f(path);
	return f.good();
}

static void drawResults(cv::Mat& image, const YoloDetResultEx& res)
{
	const int font = cv::FONT_HERSHEY_SIMPLEX;
	const double fontScale = 0.5;
	const int thickness = 1;
	for (int i = 0; i < res.num && i < static_cast<int>(res.boxes.size()); ++i) {
		const auto& box = res.boxes[i];
		cv::rectangle(image,
			cv::Point(static_cast<int>(box.left), static_cast<int>(box.top)),
			cv::Point(static_cast<int>(box.right), static_cast<int>(box.bottom)),
			cv::Scalar(0, 255, 0), 2);
		std::string label = "cls=" + std::to_string(res.classes[i]);
		if (i < static_cast<int>(res.scores.size()))
			label += " " + std::to_string(static_cast<int>(res.scores[i] * 100)) + "%";
		int baseline = 0;
		cv::Size textSize = cv::getTextSize(label, font, fontScale, thickness, &baseline);
		cv::putText(image, label,
			cv::Point(static_cast<int>(box.left), static_cast<int>(box.top) - 4),
			font, fontScale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
	}
}

static const char* backendName(YoloDetBackend b)
{
	switch (b) {
	case YoloDetBackend::CPU: return "CPU";
	case YoloDetBackend::INTEL: return "INTEL(OpenVINO)";
	default: return "GPU";
	}
}

int main(int argc, char* argv[])
{
	std::string modelPath = "depends/models/yolo26n.onnx";
	std::string imagePath = "depends/models/bus.jpg";
	if (argc >= 2)
		modelPath = argv[1];
	if (argc >= 3)
		imagePath = argv[2];

	// 未传参且默认路径不存在时，按可执行文件目录上溯到项目根再找 depends/models
	if (argc < 3) {
		std::string exeDir = getExeDir();
		if (!fileExists(imagePath))
			imagePath = resolveDependsPath("depends/models/bus.jpg", exeDir);
	}
	if (argc < 2) {
		std::string exeDir = getExeDir();
		if (!fileExists(modelPath))
			modelPath = resolveDependsPath("depends/models/yolo26n.onnx", exeDir);
	}

	cv::Mat image = cv::imread(imagePath);
	if (image.empty()) {
		std::cerr << "Error: Failed to load image: " << imagePath << std::endl;
		return 1;
	}

	YoloDet det;
	if (det.SetModel(modelPath, YoloDetBackend::INTEL) != 0) {
		std::cerr << "Error: Failed to set model: " << modelPath << std::endl;
		return 2;
	}

	YoloDetResultEx res;
	std::vector<float> infer_times;
	const int loop_count = 20;

	std::cout << "Running " << loop_count << " inference iterations (OpenVINO)..." << std::endl;
	for (int i = 0; i < loop_count; ++i) {
		YoloDetResultEx iter_res;
		if (det.Inference(image, iter_res) != 0) {
			std::cerr << "Error: Inference failed at iteration " << i << std::endl;
			return 3;
		}
		infer_times.push_back(iter_res.infer_time_ms);
		std::cout << "  Iteration [" << std::setw(2) << (i + 1) << "/" << loop_count << "]: "
			<< std::fixed << std::setprecision(3) << iter_res.infer_time_ms << " ms" << std::endl;
		// 保存最后一次的结果用于可视化
		if (i == loop_count - 1) {
			res = iter_res;
		}
	}

	// 计算平均耗时
	double avg_time = std::accumulate(infer_times.begin(), infer_times.end(), 0.0) / infer_times.size();

	std::cout << "\n=== Inference Summary ===" << std::endl;
	std::cout << "Total iterations: " << loop_count << std::endl;
	std::cout << "Average time: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
	std::cout << "Min time: " << *std::min_element(infer_times.begin(), infer_times.end()) << " ms" << std::endl;
	std::cout << "Max time: " << *std::max_element(infer_times.begin(), infer_times.end()) << " ms" << std::endl;

	std::cout << "\nDetections (last iteration): " << res.num
		<< ", backend: " << backendName(res.backend) << std::endl;
	for (int i = 0; i < res.num; ++i) {
		float score = (i < static_cast<int>(res.scores.size())) ? res.scores[i] : 0.f;
		int cls = (i < static_cast<int>(res.classes.size())) ? res.classes[i] : -1;
		const auto& box = res.boxes[i];
		std::cout << "  [" << i << "] class=" << cls
			<< " score=" << score
			<< " box=(" << box.left << "," << box.top << "," << box.right << "," << box.bottom << ")"
			<< std::endl;
	}

	cv::Mat outImage = image.clone();
	drawResults(outImage, res);
	const std::string outPath = "bus_out_openvino.jpg";
	if (cv::imwrite(outPath, outImage))
		std::cout << "Saved visualization to " << outPath << std::endl;
	else
		std::cerr << "Warning: Failed to save " << outPath << std::endl;

	return 0;
}
