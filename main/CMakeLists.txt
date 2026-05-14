#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS
        main.cc
        main_functions.cc
        constants.cc
        output_handler.cc
        model_data.cc
        microphone.cc

    PRIV_REQUIRES
        spi_flash
        esp_driver_i2s
        esp_driver_gpio

    INCLUDE_DIRS
        ""
)