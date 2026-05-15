#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "main_functions.h"
#include "model.h"
#include "test_data.h"
#include "microphone.h"
#include "feature_scaler.h"
#include "mfcc.h"
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SIMULATION_MODE false

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

float audio_buffer[MFCC_FRAME_SIZE];

// Suavização temporal: média móvel das últimas 5 inferências (~5 segundos).
// Só classifica ANOMALIA se a média for >= 0.70 — reduz falsos positivos
// de frames isolados fora da distribuição de treino.
static float prob_history[5] = {};
static int   prob_idx = 0;

static const char* kMfccNames[13] = {
    "Energia     ", "Inclinacao  ", "Curvatura   ",
    "Forma-3     ", "Forma-4     ", "Forma-5     ",
    "Forma-6     ", "Forma-7     ", "Forma-8     ",
    "Forma-9     ", "Forma-10    ", "Forma-11    ",
    "Forma-12    "
};

}  // namespace

void setup() {
  tflite::InitializeTarget();
  mfcc_init();

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Modelo incompatível!");
    return;
  }

  static tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Erro ao alocar tensores");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Modelo MFCC+MLP carregado!");

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

  for (int i = 0; i < MFCC_N_COEFFS; i++) {
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
  MicroPrintf("Esperado : %s", expected  ? "ANOMALIA" : "NORMAL");
  MicroPrintf("Inferido : %s", predicted ? "ANOMALIA" : "NORMAL");

  sample_index++;
  if (sample_index >= NUM_TEST_SAMPLES) sample_index = 0;

  vTaskDelay(pdMS_TO_TICKS(1000));

#else

  if (microphone_read(audio_buffer, MFCC_FRAME_SIZE)) {

    // Remove DC offset antes de calcular RMS — o INMP441 tem bias DC
    // que mantinha o RMS em ~0.5 mesmo sem som, mascarando variações reais
    float mean = 0.0f;
    for (int i = 0; i < MFCC_FRAME_SIZE; i++) mean += audio_buffer[i];
    mean /= MFCC_FRAME_SIZE;

    float rms = 0.0f;
    for (int i = 0; i < MFCC_FRAME_SIZE; i++) {
      float ac = audio_buffer[i] - mean;
      rms += ac * ac;
    }
    rms = sqrtf(rms / MFCC_FRAME_SIZE);

    if (rms < 0.02f) {
      MicroPrintf("Silencio (RMS_AC=%.4f)", rms);
      return;
    }

    float mfcc_features[MFCC_N_COEFFS];
    mfcc_compute(audio_buffer, mfcc_features);

    for (int i = 0; i < MFCC_N_COEFFS; i++) {
      input->data.f[i] = (mfcc_features[i] - kScalerMean[i]) / kScalerStd[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("Invoke failed");
      return;
    }

    float probability = output->data.f[0];

    prob_history[prob_idx] = probability;
    prob_idx = (prob_idx + 1) % 5;
    float avg_prob = 0.0f;
    for (int i = 0; i < 5; i++) avg_prob += prob_history[i];
    avg_prob /= 5.0f;
    int predicted = avg_prob >= 0.70f ? 1 : 0;

    char bar[21];
    int filled = (int)(rms * 100.0f);  // escala: 0.20 RMS_AC = barra cheia
    if (filled > 20) filled = 20;
    for (int i = 0; i < 20; i++) bar[i] = (i < filled) ? '#' : '.';
    bar[20] = '\0';

    MicroPrintf("=================================");
    MicroPrintf("Audio [%s] %.3f", bar, rms);
    for (int i = 0; i < MFCC_N_COEFFS; i++) {
      MicroPrintf("MFCC[%02d] %s: %.4f", i, kMfccNames[i], mfcc_features[i]);
    }
    MicroPrintf("Probabilidade        : %.4f", probability);
    MicroPrintf("Media (5 frames)     : %.4f", avg_prob);
    MicroPrintf("Inferido             : %s", predicted ? "*** ANOMALIA ***" : "NORMAL");

    vTaskDelay(pdMS_TO_TICKS(1000));
  }

#endif
}
