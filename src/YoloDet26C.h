#pragma once
// C ABI for YoloDet26, convenient for C#/PInvoke.

#include <stddef.h>

#if defined(_WIN32)
#ifdef YOLODET26_EXPORTS
#define YOLODET26_API __declspec(dllexport)
#else
#define YOLODET26_API __declspec(dllimport)
#endif
#define YOLODET26_CALL __cdecl
#else
#define YOLODET26_API
#define YOLODET26_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define YOLODET26_MAX_DETECTIONS 300

typedef void* YoloDet26Handle;

typedef struct YoloDet26Box {
	float left;
	float top;
	float right;
	float bottom;
} YoloDet26Box;

typedef struct YoloDet26Image {
	int width;
	int height;
	int channels;
	const unsigned char* data;
} YoloDet26Image;

typedef enum YoloDet26Backend {
	YOLODET26_BACKEND_CPU = 0,
	YOLODET26_BACKEND_GPU = 1
} YoloDet26Backend;

typedef struct YoloDet26Result {
	int capacity;
	int num;
	int* classes;
	float* scores;
	YoloDet26Box* boxes;
	int backend;
	float infer_time_ms;
} YoloDet26Result;

// Return: 0 success, 1 truncated, <0 error.
YOLODET26_API YoloDet26Handle YOLODET26_CALL YoloDet26_Create();
YOLODET26_API void YOLODET26_CALL YoloDet26_Destroy(YoloDet26Handle handle);
YOLODET26_API int YOLODET26_CALL YoloDet26_SetModel(YoloDet26Handle handle, const char* model_path);
YOLODET26_API int YOLODET26_CALL YoloDet26_Inference(YoloDet26Handle handle, const YoloDet26Image* image, YoloDet26Result* result);
YOLODET26_API int YOLODET26_CALL YoloDet26_IsModelLoaded(YoloDet26Handle handle);
YOLODET26_API void YOLODET26_CALL YoloDet26_SetConfThreshold(YoloDet26Handle handle, float threshold);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "YoloDet26Config.h"
static_assert(YOLODET26_MAX_DETECTIONS == YOLO26_MAX_DETECTIONS,
	"YOLODET26_MAX_DETECTIONS must match YOLO26_MAX_DETECTIONS.");
#endif
