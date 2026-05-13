#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "main_functions.h"
#include "model.h"
#include "test_data.h"

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

void setup() {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Modelo incompatível!");
    return;
  }

  static tflite::MicroMutableOpResolver<2> resolver;

  resolver.AddFullyConnected();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Erro ao alocar tensores");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Modelo de anomalia acústica carregado!");
}

void loop() {

  static int sample_index = 0;

  // envia as 5 features para o modelo
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = test_samples[sample_index][i];
  }

  // executa inferência
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // saída do modelo
  float probability = output->data.f[0];

  int predicted = probability >= 0.5f ? 1 : 0;
  int expected  = expected_labels[sample_index];

  MicroPrintf("=================================");
  MicroPrintf("Sample %d", sample_index);

  MicroPrintf("Probabilidade: %.4f", probability);

  MicroPrintf("Esperado : %s",
               expected ? "ANOMALIA" : "NORMAL");

  MicroPrintf("Inferido : %s",
               predicted ? "ANOMALIA" : "NORMAL");

  sample_index++;

  if (sample_index >= NUM_TEST_SAMPLES) {
    sample_index = 0;
  }
}