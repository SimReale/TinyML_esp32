# _ESP-DL_ vs _ESP-TFLite-Micro_: Neural Networks Inference on ESP32 Devices

In order to run the model with ESP-TFLite-micro on the ESP32, it must be converted into a C-style byte array (`const unsigned char[]`) so it can be compiled directly into the firmware's Read-Only Memory (Flash). Given the quantized TFLite model file (`model_quantized.tflite`), there are two options:

1. **Linux/MacOS**: Use the `xxd` command-line tool to convert the TFLite file into a C header file. This tool is typically pre-installed on Linux and MacOS systems.

    ```bash
    # Run the xxd conversion
    xxd -i model_quantized.tflite > model_data.cc
    ```

2. **Windows**: Use the `xxd` command-line tool from the Git for Windows package. As an alternative, use the WSL (Windows Subsystem for Linux) to run the same tool.

    ```bash
    # Navigate to your Windows Desktop from the WSL terminal
    cd /mnt/c/Users/YourWindowsUsername/Desktop

    # Run the xxd conversion in WSL
    xxd -i model_quantized.tflite > model_data.cc
    ```

After running the command, you will find a new file named `model_data.cc` in the same directory. This file contains the TFLite model as a C-style byte array, which can be included in your ESP32 firmware project.