/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
==============================================================================*/

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"

// Globals, usados para compatibilidade com estilo Arduino.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Ajuste 1: Aumentamos a arena porque o modelo agora tem mais camadas
constexpr int kTensorArenaSize = 10240; 
uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d != supported %d.", 
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Ajuste 2: Aumentado para 3 operações para garantir suporte total
  static tflite::MicroMutableOpResolver<3> resolver; 
  resolver.AddFullyConnected();
  resolver.AddRelu();
  // Às vezes o modelo quantizado usa a operação Reshape ou Quantize internamente
  // Se continuar zerado, tente usar tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  inference_count = 0;
}

void loop() {
  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // A lógica de quantização abaixo está correta, DESDE QUE o modelo seja INT8
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  input->data.int8[0] = x_quantized;

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
    return;
  }

  int8_t y_quantized = output->data.int8[0];
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  HandleOutput(x, y);

  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}