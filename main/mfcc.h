#ifndef MFCC_H_
#define MFCC_H_

#define MFCC_FRAME_SIZE  1024
#define MFCC_N_MELS      26
#define MFCC_N_COEFFS    13
#define MFCC_SAMPLE_RATE 16000
#define MFCC_F_MIN       20.0f
#define MFCC_F_MAX       8000.0f
#define MFCC_PREEMPH     0.97f

// Inicializa o banco de filtros Mel (chamar uma vez no setup)
void mfcc_init();

// Calcula 13 coeficientes MFCC a partir de um frame de MFCC_FRAME_SIZE amostras
// frame_in : sinal de áudio float normalizado [-1, 1]
// mfcc_out : saída com MFCC_N_COEFFS coeficientes
void mfcc_compute(const float* frame_in, float* mfcc_out);

#endif  // MFCC_H_
