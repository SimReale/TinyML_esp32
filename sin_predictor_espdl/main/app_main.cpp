#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint8_t model_espdl[] asm("_binary_sin_wave_model_espdl_start");

namespace {
constexpr const char *TAG = "sin_benchmark";
constexpr int kWarmupRuns = 100;
constexpr int kBenchmarkRuns = 1000;
constexpr float kTwoPi = 2.0f * 3.14159265359f;
} // namespace

extern "C" void app_main(void)
{
    dl::Model *model = new dl::Model((const char *)model_espdl,
                                    fbs::MODEL_LOCATION_IN_FLASH_RODATA);
                                    // 0,
                                    // dl::MEMORY_MANAGER_GREEDY,
                                    // nullptr,
                                    // false);

    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    
    if (model_input == nullptr) {
        ESP_LOGE(TAG, "Model input tensor is missing");
        return;
    }

    if (model_input->get_size() < 1) {
        ESP_LOGE(TAG, "Unexpected tensor size: input=%d", model_input->get_size());
        return;
    }

    if (model_input->get_dtype() != dl::DATA_TYPE_INT8) {
        ESP_LOGE(TAG, "Unexpected input dtype: %s", model_input->get_dtype_string());
        return;
    }

    const int input_exponent = model_input->get_exponent();
    const float input_inv_scale = DL_RESCALE(input_exponent);

    model->run();

    while (true) {

        for (int i = 1; i < kWarmupRuns; ++i) {
            const float x = kTwoPi * static_cast<float>(i) / static_cast<float>(kWarmupRuns);
            const int8_t quant_x = dl::quantize<int8_t>(x, input_inv_scale);
            dl::TensorBase input_sample({1, 1}, &quant_x, input_exponent, dl::DATA_TYPE_INT8, false);
            if (!model_input->assign(&input_sample)) {
                ESP_LOGE(TAG, "Failed to write warmup input into the model tensor");
                return;
            }
            model->run();
        }

        int64_t total_elapsed_us = 0;
        for (int i = 0; i < kBenchmarkRuns; ++i) {
            const float x = (kBenchmarkRuns == 1)
                                ? 0.0f
                                : (kTwoPi * static_cast<float>(i) / static_cast<float>(kBenchmarkRuns - 1));
            const int8_t quant_x = dl::quantize<int8_t>(x, input_inv_scale);
            dl::TensorBase input_sample({1, 1}, &quant_x, input_exponent, dl::DATA_TYPE_INT8, false);
            if (!model_input->assign(&input_sample)) {
                ESP_LOGE(TAG, "Failed to write benchmark input into the model tensor");
                return;
            }

            const int64_t start_us = esp_timer_get_time();
            model->run();
            const int64_t elapsed_us = esp_timer_get_time() - start_us;
            total_elapsed_us += elapsed_us;
        }

        ESP_LOGI(TAG,
                 "Average latency over %d runs: total=%lld us avg=%.2f us",
                 kBenchmarkRuns,
                 static_cast<long long>(total_elapsed_us),
                 static_cast<double>(total_elapsed_us) / kBenchmarkRuns);
    }
}