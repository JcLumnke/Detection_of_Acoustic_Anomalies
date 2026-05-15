#include "mfcc.h"
#include <math.h>
#include <string.h>

#define FFT_SIZE MFCC_FRAME_SIZE
#define N_BINS   (FFT_SIZE / 2 + 1)

static float mel_fb[MFCC_N_MELS][N_BINS];
static bool  initialized = false;

// ---------------------------------------------------------------------------
// Utilitários Mel
// ---------------------------------------------------------------------------
static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ---------------------------------------------------------------------------
// Inicialização do banco de filtros Mel
// ---------------------------------------------------------------------------
void mfcc_init() {
    if (initialized) return;

    float mel_min = hz_to_mel(MFCC_F_MIN);
    float mel_max = hz_to_mel(MFCC_F_MAX);

    // MFCC_N_MELS + 2 pontos igualmente espaçados na escala Mel
    float mel_points[MFCC_N_MELS + 2];
    for (int i = 0; i < MFCC_N_MELS + 2; i++) {
        mel_points[i] = mel_min + i * (mel_max - mel_min) / (MFCC_N_MELS + 1);
    }

    // Converte para bins FFT
    float bin_points[MFCC_N_MELS + 2];
    for (int i = 0; i < MFCC_N_MELS + 2; i++) {
        bin_points[i] = floorf((FFT_SIZE + 1) * mel_to_hz(mel_points[i]) / MFCC_SAMPLE_RATE);
    }

    // Preenche os filtros triangulares
    for (int m = 0; m < MFCC_N_MELS; m++) {
        int f_left   = (int)bin_points[m];
        int f_center = (int)bin_points[m + 1];
        int f_right  = (int)bin_points[m + 2];

        for (int k = 0; k < N_BINS; k++) {
            if (k < f_left || k > f_right) {
                mel_fb[m][k] = 0.0f;
            } else if (k <= f_center) {
                mel_fb[m][k] = (float)(k - f_left) / (f_center - f_left + 1e-6f);
            } else {
                mel_fb[m][k] = (float)(f_right - k) / (f_right - f_center + 1e-6f);
            }
        }
    }

    initialized = true;
}

// ---------------------------------------------------------------------------
// FFT radix-2 Cooley-Tukey in-place
// ---------------------------------------------------------------------------
static void fft_inplace(float* re, float* im, int n) {
    // Bit-reversal
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }

    // Butterfly
    for (int len = 2; len <= n; len <<= 1) {
        float ang  = -2.0f * (float)M_PI / len;
        float wre  = cosf(ang);
        float wim  = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float u_re = re[i + k];
                float u_im = im[i + k];
                float v_re = re[i + k + len/2] * cur_re - im[i + k + len/2] * cur_im;
                float v_im = re[i + k + len/2] * cur_im + im[i + k + len/2] * cur_re;
                re[i + k]         = u_re + v_re;
                im[i + k]         = u_im + v_im;
                re[i + k + len/2] = u_re - v_re;
                im[i + k + len/2] = u_im - v_im;
                float new_re = cur_re * wre - cur_im * wim;
                cur_im       = cur_re * wim + cur_im * wre;
                cur_re       = new_re;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cálculo dos MFCCs
// ---------------------------------------------------------------------------
// Todos os buffers estáticos para não ocupar a stack da task
static float fft_re[FFT_SIZE];
static float fft_im[FFT_SIZE];
static float s_power[N_BINS];
static float s_log_mel[MFCC_N_MELS];

void mfcc_compute(const float* frame_in, float* mfcc_out) {
    if (!initialized) mfcc_init();

    // 1. Pré-ênfase
    fft_re[0] = frame_in[0];
    for (int i = 1; i < FFT_SIZE; i++) {
        fft_re[i] = frame_in[i] - MFCC_PREEMPH * frame_in[i - 1];
    }

    // 2. Janela Hamming
    for (int i = 0; i < FFT_SIZE; i++) {
        float w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (FFT_SIZE - 1));
        fft_re[i] *= w;
        fft_im[i]  = 0.0f;
    }

    // 3. FFT
    fft_inplace(fft_re, fft_im, FFT_SIZE);

    // 4. Espectro de potência
    for (int k = 0; k < N_BINS; k++) {
        s_power[k] = fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k];
    }

    // 5. Banco de filtros Mel + log
    for (int m = 0; m < MFCC_N_MELS; m++) {
        float energy = 0.0f;
        for (int k = 0; k < N_BINS; k++) {
            energy += mel_fb[m][k] * s_power[k];
        }
        s_log_mel[m] = logf(energy + 1e-10f);
    }

    // 6. DCT-II → 13 coeficientes MFCC
    for (int c = 0; c < MFCC_N_COEFFS; c++) {
        float sum = 0.0f;
        for (int m = 0; m < MFCC_N_MELS; m++) {
            sum += s_log_mel[m] * cosf((float)M_PI * c * (m + 0.5f) / MFCC_N_MELS);
        }
        mfcc_out[c] = sum;
    }
}
