cmake_minimum_required(VERSION 3.17)

project(YoloDet26 LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(YOLODET26_BACKEND "GPU" CACHE STRING "Backend: CPU or GPU")
set_property(CACHE YOLODET26_BACKEND PROPERTY STRINGS CPU GPU)
string(TOUPPER "${YOLODET26_BACKEND}" YOLODET26_BACKEND_UPPER)
if (YOLODET26_BACKEND_UPPER STREQUAL "CPU")
	set(YOLODET26_ENABLE_CPU ON)
elseif (YOLODET26_BACKEND_UPPER STREQUAL "GPU")
	set(YOLODET26_ENABLE_GPU ON)
else()
	message(FATAL_ERROR "YOLODET26_BACKEND must be CPU or GPU.")
endif()

set(TENSORRT_DIR "" CACHE PATH "TensorRT root directory")
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root directory")
# 尝试从环境变量自动填充路径，失败则使用 depends
if (NOT TENSORRT_DIR)
	if (DEFINED ENV{TENSORRT_DIR})
		set(TENSORRT_DIR "$ENV{TENSORRT_DIR}" CACHE PATH "TensorRT root directory" FORCE)
	elseif (DEFINED ENV{TENSORRT_ROOT})
		set(TENSORRT_DIR "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root directory" FORCE)
	endif()
endif()

if (NOT ONNXRUNTIME_DIR)
	if (DEFINED ENV{ONNXRUNTIME_DIR})
		set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}" CACHE PATH "ONNX Runtime root directory" FORCE)
	elseif (DEFINED ENV{ONNXRUNTIME_ROOT})
		set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root directory" FORCE)
	else()
		set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/depends/onnxruntime" CACHE PATH "ONNX Runtime root directory" FORCE)
	endif()
endif()

# OpenCV：优先使用 depends 中的预构建（depends/opencv/build/x64/vc16）
set(OPENCV_DEPENDS_ROOT "${CMAKE_SOURCE_DIR}/depends/opencv")
set(OPENCV_VC16 "${OPENCV_DEPENDS_ROOT}/build/x64/vc16")
find_library(OpenCV_LIB_RELEASE opencv_world4100 PATHS "${OPENCV_VC16}/lib" NO_DEFAULT_PATH)
find_library(OpenCV_LIB_DEBUG   opencv_world4100d PATHS "${OPENCV_VC16}/lib" NO_DEFAULT_PATH)
if (OpenCV_LIB_RELEASE)
	set(OpenCV_INCLUDE_DIRS "${OPENCV_DEPENDS_ROOT}/build/include")
	if (NOT OpenCV_LIB_DEBUG)
		set(OpenCV_LIB_DEBUG "${OpenCV_LIB_RELEASE}")
	endif()
	set(OpenCV_LIBS optimized "${OpenCV_LIB_RELEASE}" debug "${OpenCV_LIB_DEBUG}")
	set(OpenCV_DEPENDS_BIN "${OPENCV_VC16}/bin")
else()
	find_package(OpenCV REQUIRED)
	set(OpenCV_DEPENDS_BIN "")
endif()

if (YOLODET26_ENABLE_GPU)
	find_package(CUDAToolkit QUIET)
	if (NOT CUDAToolkit_FOUND)
		if (DEFINED ENV{CUDA_PATH})
			set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}" CACHE PATH "CUDA Toolkit root directory" FORCE)
		elseif (DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
			set(CUDAToolkit_ROOT "$ENV{CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA Toolkit root directory" FORCE)
		endif()
		find_package(CUDAToolkit QUIET)
	endif()

	set(CUDART_LIB "")
	if (TARGET CUDAToolkit::cudart)
		set(CUDART_LIB CUDAToolkit::cudart)
	else()
		set(CUDA_ROOT_DIR "${CUDAToolkit_ROOT}")
		if (NOT CUDA_ROOT_DIR AND DEFINED ENV{CUDA_PATH})
			set(CUDA_ROOT_DIR "$ENV{CUDA_PATH}")
		endif()
		if (NOT CUDA_ROOT_DIR AND DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
			set(CUDA_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
		endif()
		unset(CUDART_LIB CACHE)
		find_library(CUDART_LIB cudart
			PATHS
				"${CUDA_ROOT_DIR}/lib/x64"
				"${CUDA_ROOT_DIR}/lib"
				"${CUDA_ROOT_DIR}/lib64"
		)
		if (NOT CUDART_LIB AND DEFINED CUDA_CUDART)
			set(CUDART_LIB "${CUDA_CUDART}")
		endif()
		if (NOT CUDART_LIB AND DEFINED CUDA_CUDART_LIBRARY)
			set(CUDART_LIB "${CUDA_CUDART_LIBRARY}")
		endif()
	endif()

	if (NOT CUDART_LIB)
		message(FATAL_ERROR "CUDA cudart not found. 请手动设置 CUDAToolkit_ROOT 或 CUDA_PATH。若仅需CPU构建，请设置 YOLODET26_BACKEND=CPU。")
	endif()

	set(CUDA_INCLUDE_DIR "")
	if (DEFINED CUDAToolkit_INCLUDE_DIRS AND CUDAToolkit_INCLUDE_DIRS)
		set(CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
	elseif (CUDA_ROOT_DIR)
		set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
	endif()
	if (NOT CUDA_INCLUDE_DIR)
		message(FATAL_ERROR "CUDA include not found. 请确认 CUDA_PATH 或 CUDAToolkit_ROOT。若仅需CPU构建，请设置 YOLODET26_BACKEND=CPU。")
	endif()

	find_path(TENSORRT_INCLUDE_DIR NvInfer.h
		PATHS "${TENSORRT_DIR}/include"
	)
	find_library(TENSORRT_LIB
		NAMES nvinfer nvinfer_10
		PATHS "${TENSORRT_DIR}/lib" "${TENSORRT_DIR}/lib64"
	)
	find_library(TENSORRT_PLUGIN_LIB
		NAMES nvinfer_plugin nvinfer_plugin_10
		PATHS "${TENSORRT_DIR}/lib" "${TENSORRT_DIR}/lib64"
	)

	if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB OR NOT TENSORRT_PLUGIN_LIB)
		message(FATAL_ERROR "TensorRT not found. 请手动设置 TENSORRT_DIR（或环境变量 TENSORRT_DIR/TENSORRT_ROOT）。若仅需CPU构建，请设置 YOLODET26_BACKEND=CPU。")
	endif()
endif()

if (YOLODET26_ENABLE_CPU)
	find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
		PATHS "${ONNXRUNTIME_DIR}/include"
	)
	find_library(ONNXRUNTIME_LIB onnxruntime
		PATHS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/lib64"
	)

	if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
		message(FATAL_ERROR "ONNX Runtime not found. 请手动设置 ONNXRUNTIME_DIR（或环境变量 ONNXRUNTIME_DIR/ONNXRUNTIME_ROOT）。若仅需GPU构建，请设置 YOLODET26_BACKEND=GPU。")
	endif()
endif()

set(SRC
	src/YoloDet26.cpp
)
if (YOLODET26_ENABLE_CPU)
	list(APPEND SRC
		src/cpu/YoloDet26OnnxImpl.cpp
		src/cpu/YoloDet26CpuBackend.cpp
	)
endif()
if (YOLODET26_ENABLE_GPU)
	list(APPEND SRC
		src/trt/YoloDet26TrtBackend.cpp
	)
endif()

add_library(YoloDet26 SHARED ${SRC})
target_compile_definitions(YoloDet26 PRIVATE YOLODET26_EXPORTS)
if (YOLODET26_ENABLE_CPU)
	target_compile_definitions(YoloDet26 PRIVATE YOLODET26_ENABLE_CPU)
endif()
if (YOLODET26_ENABLE_GPU)
	target_compile_definitions(YoloDet26 PRIVATE YOLODET26_ENABLE_GPU)
endif()
if (MSVC)
	target_compile_options(YoloDet26 PRIVATE /utf-8)
endif()

set(YOLODET26_PRIVATE_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	${OpenCV_INCLUDE_DIRS}
)
set(YOLODET26_PRIVATE_LIBS
	${OpenCV_LIBS}
)
if (YOLODET26_ENABLE_GPU)
	list(APPEND YOLODET26_PRIVATE_INCLUDE_DIRS
		"${CMAKE_CURRENT_SOURCE_DIR}/src/trt"
		${CUDA_INCLUDE_DIR}
		${TENSORRT_INCLUDE_DIR}
	)
	list(APPEND YOLODET26_PRIVATE_LIBS
		${TENSORRT_LIB}
		${TENSORRT_PLUGIN_LIB}
		${CUDART_LIB}
	)
endif()
if (YOLODET26_ENABLE_CPU)
	list(APPEND YOLODET26_PRIVATE_INCLUDE_DIRS
		"${CMAKE_CURRENT_SOURCE_DIR}/src/cpu"
		${ONNXRUNTIME_INCLUDE_DIR}
	)
	list(APPEND YOLODET26_PRIVATE_LIBS
		${ONNXRUNTIME_LIB}
	)
endif()

target_include_directories(YoloDet26
	PRIVATE
		${YOLODET26_PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(YoloDet26
	PRIVATE
		${YOLODET26_PRIVATE_LIBS}
)

# CPU 测试可执行文件：yolo26n.onnx 识别 bus.jpg；运行需能加载 depends 中的 OpenCV/ONNX DLL
if (YOLODET26_ENABLE_CPU)
	add_executable(test_cpu_bus tests/test_cpu_bus.cpp)
	target_include_directories(test_cpu_bus PRIVATE
		${CMAKE_SOURCE_DIR}/src
		${OpenCV_INCLUDE_DIRS}
	)
	target_link_libraries(test_cpu_bus PRIVATE YoloDet26 ${OpenCV_LIBS})
	if (MSVC)
		target_compile_options(test_cpu_bus PRIVATE /utf-8)
	endif()
	# 构建后复制 OpenCV/ONNX Runtime DLL 到 exe 同目录，便于直接运行
	add_custom_command(TARGET test_cpu_bus POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:test_cpu_bus>"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll" "$<TARGET_FILE_DIR:test_cpu_bus>"
		COMMENT "Copy ONNX Runtime DLLs to test_cpu_bus output dir"
	)
	if (OpenCV_DEPENDS_BIN)
		add_custom_command(TARGET test_cpu_bus POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenCV_DEPENDS_BIN}/opencv_world4100.dll" "$<TARGET_FILE_DIR:test_cpu_bus>"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenCV_DEPENDS_BIN}/opencv_world4100d.dll" "$<TARGET_FILE_DIR:test_cpu_bus>"
			COMMENT "Copy OpenCV DLLs to test_cpu_bus output dir"
		)
	endif()
	add_test(NAME test_cpu_bus COMMAND test_cpu_bus WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	# 运行测试时 PATH 包含 depends/opencv/build/x64/vc16/bin 与 depends/onnxruntime/lib
	set(TEST_CPU_BUS_PATH "${ONNXRUNTIME_DIR}/lib")
	if (OpenCV_DEPENDS_BIN)
		set(TEST_CPU_BUS_PATH "${OpenCV_DEPENDS_BIN};${TEST_CPU_BUS_PATH}")
	endif()
	set_tests_properties(test_cpu_bus PROPERTIES
		ENVIRONMENT "PATH=${TEST_CPU_BUS_PATH};$ENV{PATH}"
	)
endif()
