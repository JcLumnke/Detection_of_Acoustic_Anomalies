#include "microphone.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

static const char* TAG = "MICROPHONE";

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 480

static i2s_chan_handle_t rx_chan = NULL;

void microphone_init() {

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0,
        I2S_ROLE_MASTER
    );

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &rx_chan)
    );

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = GPIO_NUM_4,
            .ws = GPIO_NUM_5,
            .dout = I2S_GPIO_UNUSED,
            .din = GPIO_NUM_6,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(rx_chan, &std_cfg)
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_chan)
    );

    ESP_LOGI(TAG, "Microfone INMP441 inicializado");
}

bool microphone_read(float* buffer, int samples) {

    int32_t raw_buffer[BUFFER_SIZE];

    size_t bytes_read = 0;

    esp_err_t err = i2s_channel_read(
        rx_chan,
        raw_buffer,
        sizeof(raw_buffer),
        &bytes_read,
        portMAX_DELAY
    );

    if (err != ESP_OK) {
        return false;
    }

    int samples_read = bytes_read / sizeof(int32_t);

    if (samples_read > samples) {
        samples_read = samples;
    }

    for (int i = 0; i < samples_read; i++) {

        buffer[i] = raw_buffer[i] / 2147483648.0f;
    }

    return true;
}