#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "main_functions.h"
#include "model.h"
#include "test_data.h"
#include "microphone.h"
#include <math.h>

#define SIMULATION_MODE true

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

float audio_buffer[480];

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

#if SIMULATION_MODE
  MicroPrintf("Modo: SIMULACAO (dataset de teste)");
#else
  MicroPrintf("Modo: SENSOR REAL");
  microphone_init();
#endif
}

void loop() {

#if SIMULATION_MODE
  static int sample_index = 0;

  for (int i = 0; i < 5; i++) {
    input->data.f[i] = test_samples[sample_index][i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  float probability = output->data.f[0];
  int predicted = probability >= 0.5f ? 1 : 0;
  int expected  = expected_labels[sample_index];

  MicroPrintf("=================================");
  MicroPrintf("Sample %d", sample_index);
  MicroPrintf("Probabilidade: %.4f", probability);
  MicroPrintf("Esperado : %s", expected ? "ANOMALIA" : "NORMAL");
  MicroPrintf("Inferido : %s", predicted ? "ANOMALIA" : "NORMAL");

  sample_index++;
  if (sample_index >= NUM_TEST_SAMPLES) {
    sample_index = 0;
  }

#else

  if (microphone_read(audio_buffer, 480)) {

    const int N = 480;
    float mean = 0.0f;
    float rms = 0.0f;
    float peak = 0.0f;

    // 1. Primeiro Passo: Calcular apenas a MÉDIA
    for (int i = 0; i < N; i++) {
        mean += audio_buffer[i];
    }
    mean /= N;

    // 2. Segundo Passo: Calcular RMS LIMPO (subtraindo a média) e o PICO
    for (int i = 0; i < N; i++) {
        float x_limpo = audio_buffer[i] - mean;
        rms += x_limpo * x_limpo;

        if (fabs(audio_buffer[i]) > peak) {
            peak = fabs(audio_buffer[i]);
        }
    }
    rms = sqrtf(rms / N);

    MicroPrintf("RMS atual: %.5f", rms);

    if (rms < 0.40f) {   // alterar aqui para detectar mais ou menos ruido ambiente
      // Aumentando o valor ele detecta menos  o ruido do ambiente.
        MicroPrintf("Silencio detectado");
        return;
    }

    // variância
    float variance = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = audio_buffer[i] - mean;
        variance += d * d;
    }
    variance /= N;

    float stddev = sqrtf(variance + 1e-8f);

    // skewness + kurtosis
    float skewness = 0.0f;
    float kurtosis = 0.0f;

    for (int i = 0; i < N; i++) {
        float d = audio_buffer[i] - mean;
        skewness += d * d * d;
        kurtosis += d * d * d * d;
    }

    skewness /= (N * stddev * stddev * stddev + 1e-8f);
    kurtosis /= (N * variance * variance + 1e-8f);

    // crest factor
    float crest_factor = peak / (rms + 1e-6f);

    // envia para o modelo
    input->data.f[0] = rms;
    input->data.f[1] = peak;
    input->data.f[2] = kurtosis;
    input->data.f[3] = skewness;
    input->data.f[4] = crest_factor;

    // inferência
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
    }

    float probability = output->data.f[0];

    int predicted = probability >= 0.33f ? 1 : 0; //alterar aqui para ser mais ou menos conservador
    // Ao diminuir o valor a IA fica mais sensivel e detecta as anomalias mais cedo.

    MicroPrintf("=================================");
    MicroPrintf("RMS          : %.5f", rms);
    MicroPrintf("Peak         : %.5f", peak);
    MicroPrintf("Kurtosis     : %.5f", kurtosis);
    MicroPrintf("Skewness     : %.5f", skewness);
    MicroPrintf("Crest Factor : %.5f", crest_factor);
    MicroPrintf("Probabilidade: %.5f", probability);
    MicroPrintf("Inferido     : %s", predicted ? "ANOMALIA" : "NORMAL");
  }

#endif
}