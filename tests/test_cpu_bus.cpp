// CPU 测试：使用 yolo26n.onnx 对 bus.jpg 做一次推理并输出检测结果。
// 用法: test_cpu_bus [model_path] [image_path]
// 默认: depends/models/yolo26n.onnx, depends/models/bus.jpg
//       先按当前工作目录查找，若不存在则按可执行文件所在目录上溯到项目根再找 depends/models

#include "YoloDet26Api.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>

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
	if (det.SetModel(modelPath) != 0) {
		std::cerr << "Error: Failed to set model: " << modelPath << std::endl;
		return 2;
	}

	YoloDetResultEx res;
	if (det.Inference(image, res) != 0) {
		std::cerr << "Error: Inference failed." << std::endl;
		return 3;
	}

	std::cout << "Detections: " << res.num
		<< ", infer_time_ms: " << res.infer_time_ms << " ms" << std::endl;
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
	const std::string outPath = "bus_out.jpg";
	if (cv::imwrite(outPath, outImage))
		std::cout << "Saved visualization to " << outPath << std::endl;
	else
		std::cerr << "Warning: Failed to save " << outPath << std::endl;

	return 0;
}
