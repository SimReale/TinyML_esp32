#include <cmath>
#include <cstdint>
#include <cstdio>

#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "sin_wave_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kWarmupRuns = 10;
constexpr int kBenchmarkRuns = 1000;
constexpr float kSineRangeEnd = 2.0f * 3.14159265359f;
constexpr size_t kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

bool FillInputTensor(float value)
{
	if (input == nullptr) {
		return false;
	}

	const int element_count = input->bytes;

	switch (input->type) {
	case kTfLiteFloat32: {
		const int count = element_count / static_cast<int>(sizeof(float));
		for (int i = 0; i < count; ++i) {
			input->data.f[i] = value;
		}
		return true;
	}
	case kTfLiteInt8: {
		const int count = element_count / static_cast<int>(sizeof(int8_t));
		const int32_t zero_point = input->params.zero_point;
		const float scale = input->params.scale;
		for (int i = 0; i < count; ++i) {
			int32_t quantized = static_cast<int32_t>(std::lround(value / scale)) + zero_point;
			if (quantized < -128) {
				quantized = -128;
			} else if (quantized > 127) {
				quantized = 127;
			}
			input->data.int8[i] = static_cast<int8_t>(quantized);
		}
		return true;
	}
	case kTfLiteUInt8: {
		const int count = element_count / static_cast<int>(sizeof(uint8_t));
		const int32_t zero_point = input->params.zero_point;
		const float scale = input->params.scale;
		for (int i = 0; i < count; ++i) {
			int32_t quantized = static_cast<int32_t>(std::lround(value / scale)) + zero_point;
			if (quantized < 0) {
				quantized = 0;
			} else if (quantized > 255) {
				quantized = 255;
			}
			input->data.uint8[i] = static_cast<uint8_t>(quantized);
		}
		return true;
	}
	case kTfLiteInt16: {
		const int count = element_count / static_cast<int>(sizeof(int16_t));
		const int32_t zero_point = input->params.zero_point;
		const float scale = input->params.scale;
		for (int i = 0; i < count; ++i) {
			int32_t quantized = static_cast<int32_t>(std::lround(value / scale)) + zero_point;
			if (quantized < -32768) {
				quantized = -32768;
			} else if (quantized > 32767) {
				quantized = 32767;
			}
			input->data.i16[i] = static_cast<int16_t>(quantized);
		}
		return true;
	}
	default:
		MicroPrintf("Unsupported input tensor type: %d", input->type);
		return false;
	}
}

// float ReadOutputTensor()
// {
// 	if (output == nullptr) {
// 		return 0.0f;
// 	}

// 	switch (output->type) {
// 	case kTfLiteFloat32:
// 		return output->data.f[0];
// 	case kTfLiteInt8:
// 		return (static_cast<float>(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
// 	case kTfLiteUInt8:
// 		return (static_cast<float>(output->data.uint8[0]) - output->params.zero_point) * output->params.scale;
// 	case kTfLiteInt16:
// 		return (static_cast<float>(output->data.i16[0]) - output->params.zero_point) * output->params.scale;
// 	default:
// 		return 0.0f;
// 	}
// }

bool InitializeInterpreter()
{
	model = tflite::GetModel(sin_wave_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf("Model schema %d is not supported by TFLM schema %d", model->version(), TFLITE_SCHEMA_VERSION);
		return false;
	}

	static tflite::MicroMutableOpResolver<3> resolver;
	if (resolver.AddFullyConnected() != kTfLiteOk) {
		MicroPrintf("AddFullyConnected failed");
		return false;
	}
	if (resolver.AddTanh() != kTfLiteOk) {
		MicroPrintf("AddTanh failed");
		return false;
	}
	if (resolver.AddReshape() != kTfLiteOk) {
		MicroPrintf("AddReshape failed");
		return false;
	}

	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		MicroPrintf("AllocateTensors failed");
		return false;
	}

	input = interpreter->input(0);
	output = interpreter->output(0);
	if (input == nullptr || output == nullptr) {
		MicroPrintf("Failed to obtain input or output tensor");
		return false;
	}

	return true;
}

void MeasureInferenceLatency()
{
	const int total_runs = kWarmupRuns + kBenchmarkRuns;
	const int run_count = total_runs > 1 ? total_runs - 1 : 1;

	for (int i = 0; i < kWarmupRuns; ++i) {
		const float position = static_cast<float>(i) / static_cast<float>(run_count);
		const float x = position * kSineRangeEnd;

		if (!FillInputTensor(x)) {
			return;
		}

		if (interpreter->Invoke() != kTfLiteOk) {
			MicroPrintf("Warmup invoke failed");
			return;
		}
	}

	const int64_t start_us = esp_timer_get_time();
	for (int i = 0; i < kBenchmarkRuns; ++i) {
		const float position = static_cast<float>(kWarmupRuns + i) / static_cast<float>(run_count);
		const float x = position * kSineRangeEnd;

		if (!FillInputTensor(x)) {
			return;
		}

		if (interpreter->Invoke() != kTfLiteOk) {
			MicroPrintf("Benchmark invoke failed");
			return;
		}
	}
	const int64_t total_us = esp_timer_get_time() - start_us;
	const float average_us = static_cast<float>(total_us) / static_cast<float>(kBenchmarkRuns);

	std::printf("Sin model latency: total=%lld us, average=%.2f us over %d runs\n",
				static_cast<long long>(total_us),
				static_cast<double>(average_us),
				kBenchmarkRuns);
}

}  // namespace

extern "C" void app_main(void)
{
	if (!InitializeInterpreter()) {
		while (true) {
			vTaskDelay(pdMS_TO_TICKS(1000));
		}
	}

	while (true) {
		MeasureInferenceLatency();
		vTaskDelay(pdMS_TO_TICKS(5000));
	}
}
