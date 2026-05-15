"""
Treina MLP para detecção de anomalias acústicas usando features MFCC.

Pipeline:
  Áudio real (WAV 44100Hz) → resample 16kHz → frames 1024 → MFCC (13 coefs)  [NORMAL]
  Frames reais + impulsos sintéticos → MFCC                                   [ANOMALIA]
  MFCCs → StandardScaler → MLP(13→16→32→1)

Se a pasta audios_proprios/ não existir, usa geração sintética calibrada.

Gera:
  main/model_data.cc      <- modelo float32
  main/feature_scaler.h   <- constantes de normalização (13 features MFCC)
  main/test_data.h        <- amostras normalizadas para modo simulação

Uso:
  pip install tensorflow numpy scipy
  python train_models.py
"""

import os
import wave
import struct
import numpy as np
import tensorflow as tf
from scipy.fft import fft as scipy_fft
from scipy.signal import resample_poly

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Parâmetros MFCC — devem ser IDÊNTICOS ao mfcc.cc do firmware
# ---------------------------------------------------------------------------
SAMPLE_RATE  = 16000
FRAME_SIZE   = 1024
N_MELS       = 26
N_COEFFS     = 13
F_MIN        = 20.0
F_MAX        = 8000.0
PREEMPH      = 0.97

AUDIO_DIR    = "audios_proprios"

# ---------------------------------------------------------------------------
# MFCC em Python (espelha exatamente o mfcc.cc)
# ---------------------------------------------------------------------------
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def build_mel_filterbank():
    n_bins = FRAME_SIZE // 2 + 1
    mel_min = hz_to_mel(F_MIN)
    mel_max = hz_to_mel(F_MAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    bin_points = np.floor((FRAME_SIZE + 1) * mel_to_hz(mel_points) / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        fl, fc, fr = bin_points[m], bin_points[m+1], bin_points[m+2]
        for k in range(n_bins):
            if fl <= k <= fc:
                fb[m, k] = (k - fl) / (fc - fl + 1e-6)
            elif fc < k <= fr:
                fb[m, k] = (fr - k) / (fr - fc + 1e-6)
    return fb

MEL_FB = build_mel_filterbank()

def compute_mfcc(signal):
    emphasized = np.concatenate([[signal[0]], signal[1:] - PREEMPH * signal[:-1]])
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_SIZE) / (FRAME_SIZE - 1))
    windowed = emphasized * window
    spectrum = np.abs(scipy_fft(windowed, n=FRAME_SIZE)[:FRAME_SIZE//2+1]) ** 2
    mel_energy = MEL_FB @ spectrum
    log_mel = np.log(mel_energy + 1e-10)
    n = np.arange(N_MELS)
    mfcc = np.array([
        np.sum(log_mel * np.cos(np.pi * c * (n + 0.5) / N_MELS))
        for c in range(N_COEFFS)
    ], dtype=np.float32)
    return mfcc

# ---------------------------------------------------------------------------
# Carrega WAV mono, reamostrado para SAMPLE_RATE
# ---------------------------------------------------------------------------
def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr    = w.getframerate()
        ch    = w.getnchannels()
        sw    = w.getsampwidth()
        nf    = w.getnframes()
        raw   = w.readframes(nf)

    fmt = {1: 'b', 2: 'h', 4: 'i'}[sw]
    samples = np.array(struct.unpack(f'{nf * ch}{fmt}', raw), dtype=np.float32)

    # Mix down para mono se necessário
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)

    # Normaliza para [-1, 1]
    max_val = float(2 ** (sw * 8 - 1))
    samples /= max_val

    # Resample para 16kHz usando fração racional
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        samples = resample_poly(samples, SAMPLE_RATE // g, sr // g).astype(np.float32)

    return samples

# ---------------------------------------------------------------------------
# Extrai frames de FRAME_SIZE sem overlap
# ---------------------------------------------------------------------------
def extract_frames(signal):
    frames = []
    n_frames = len(signal) // FRAME_SIZE
    for i in range(n_frames):
        frame = signal[i * FRAME_SIZE : (i + 1) * FRAME_SIZE].copy()
        # Remove DC do frame (mesmo que o firmware faz antes do RMS)
        frame -= frame.mean()
        rms = np.sqrt(np.mean(frame ** 2))
        if rms >= 0.02:  # mesmo threshold do firmware
            frames.append(frame)
    return frames

# ---------------------------------------------------------------------------
# Gera anomalia adicionando impulsos sobre um frame real
# ---------------------------------------------------------------------------
def make_anomaly_frame(frame):
    sig = frame.copy()
    n_impulses = np.random.randint(3, 10)
    for _ in range(n_impulses):
        pos   = np.random.randint(0, FRAME_SIZE)
        width = np.random.randint(2, 8)
        amp   = np.random.uniform(1.5, 3.5)
        start = max(0, pos - width)
        end   = min(FRAME_SIZE, pos + width)
        sig[start:end] += amp * np.random.choice([-1, 1])
    sig /= (np.max(np.abs(sig)) + 1e-6)
    return sig.astype(np.float32)

# ---------------------------------------------------------------------------
# Geração sintética (fallback se não houver áudios reais)
# ---------------------------------------------------------------------------
def generate_base_signal():
    t = np.linspace(0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE)
    fundamental    = np.random.uniform(80, 200)
    harmonic_scale = np.random.uniform(0.5, 2.0)
    harmonics = np.zeros(FRAME_SIZE)
    for h in range(1, 6):
        harmonics += (1.0 / h) * np.sin(2 * np.pi * fundamental * h * t)
    noise = np.random.normal(0, 1.0, FRAME_SIZE)
    return (noise + harmonics * harmonic_scale).astype(np.float32)

def generate_normal_signal():
    signal = generate_base_signal()
    signal /= (np.max(np.abs(signal)) + 1e-6)
    signal *= np.random.uniform(0.45, 0.70)
    return signal

def generate_anomaly_signal():
    signal = generate_base_signal()
    n_impulses = np.random.randint(3, 10)
    for _ in range(n_impulses):
        pos   = np.random.randint(0, FRAME_SIZE)
        width = np.random.randint(2, 8)
        amp   = np.random.uniform(1.5, 3.5)
        start = max(0, pos - width)
        end   = min(FRAME_SIZE, pos + width)
        signal[start:end] += amp * np.random.choice([-1, 1])
    signal /= (np.max(np.abs(signal)) + 1e-6)
    signal *= np.random.uniform(0.65, 0.90)
    return signal

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def split(X, y, val_ratio=0.1, test_ratio=0.2):
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (
        X[n_test + n_val:], y[n_test + n_val:],
        X[n_test:n_test + n_val], y[n_test:n_test + n_val],
        X[:n_test], y[:n_test],
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 55)

real_frames = []
if os.path.isdir(AUDIO_DIR):
    wav_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"  Carregando {len(wav_files)} arquivos de {AUDIO_DIR}/")
    print("  (todos tratados como NORMAL — anomalias geradas via impulsos)")
    for fname in wav_files:
        path = os.path.join(AUDIO_DIR, fname)
        sig  = load_wav(path)
        frames = extract_frames(sig)
        real_frames.extend(frames)
        print(f"    {fname}: {len(sig)/SAMPLE_RATE:.1f}s → {len(frames)} frames")
    print(f"  Total frames reais: {len(real_frames)}")

if real_frames:
    # NORMAL: frames reais
    # ANOMALIA: impulsos sobre frames reais (preserva distribuição espectral)
    print("\n  Calculando MFCCs dos frames normais...")
    X_normal_frames = [compute_mfcc(f) for f in real_frames]

    print("  Gerando anomalias a partir dos frames reais...")
    # Gera anomalia para cada frame normal (dataset balanceado)
    anomaly_frames = [make_anomaly_frame(f) for f in real_frames]
    X_anomaly_frames = [compute_mfcc(f) for f in anomaly_frames]

    X_normal  = np.array(X_normal_frames,  dtype=np.float32)
    X_anomaly = np.array(X_anomaly_frames, dtype=np.float32)
    source = "áudios reais"
else:
    print("  audios_proprios/ não encontrada — usando geração sintética")
    n = 1000
    print("  Gerando sinais normais...")
    X_normal  = np.array([compute_mfcc(generate_normal_signal())  for _ in range(n)], dtype=np.float32)
    print("  Gerando sinais de anomalia...")
    X_anomaly = np.array([compute_mfcc(generate_anomaly_signal()) for _ in range(n)], dtype=np.float32)
    source = "síntese calibrada"

X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))]).astype(np.float32)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"\n  Dataset ({source}): {len(X_normal)} NORMAL + {len(X_anomaly)} ANOMALIA = {len(X)} total")

X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split(X, y)
print(f"  Treino: {len(X_train_raw)} | Val: {len(X_val_raw)} | Teste: {len(X_test_raw)}")

# Normalização StandardScaler
scaler_mean = X_train_raw.mean(axis=0)
scaler_std  = X_train_raw.std(axis=0) + 1e-8

X_train = (X_train_raw - scaler_mean) / scaler_std
X_val   = (X_val_raw   - scaler_mean) / scaler_std
X_test  = (X_test_raw  - scaler_mean) / scaler_std

# feature_scaler.h
mean_vals = ", ".join(f"{v:.6f}f" for v in scaler_mean)
std_vals  = ", ".join(f"{v:.6f}f" for v in scaler_std)
scaler_lines = [
    "// Gerado por train_models.py — StandardScaler para 13 MFCCs",
    "// Aplicar no firmware: x_scaled = (x - mean) / std",
    "",
    "#ifndef FEATURE_SCALER_H_",
    "#define FEATURE_SCALER_H_",
    "",
    "// Ordem: MFCC[0] .. MFCC[12]",
    f"static const float kScalerMean[13] = {{{mean_vals}}};",
    f"static const float kScalerStd[13]  = {{{std_vals}}};",
    "",
    "#endif  // FEATURE_SCALER_H_",
    "",
]
with open("main/feature_scaler.h", "w") as f:
    f.write("\n".join(scaler_lines))
print("  Salvo: main/feature_scaler.h")

# Treinamento
print("\n  Treinando MLP(13→16→32→1)...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_COEFFS,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=100, batch_size=32, verbose=0)

_, acc_train = model.evaluate(X_train, y_train, verbose=0)
_, acc_val   = model.evaluate(X_val,   y_val,   verbose=0)
_, acc_test  = model.evaluate(X_test,  y_test,  verbose=0)
print(f"  Treino: {acc_train*100:.1f}%  Val: {acc_val*100:.1f}%  Teste: {acc_test*100:.1f}%")

# Converte para TFLite float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()

with open("model_mfcc_mlp.tflite", "wb") as f:
    f.write(tflite_bytes)

def to_c_array(data, acc_train, acc_val, acc_test):
    lines = [
        "// Gerado por train_models.py — Detection of Acoustic Anomalies",
        "// Modelo MFCC+MLP: Dense(13->16,relu)->Dense(16->32,relu)->Dense(32->1,sigmoid)",
        "// Features: 13 coeficientes MFCC normalizados (StandardScaler)",
        f"// Acuracia treino: {acc_train:.1f}%  val: {acc_val:.1f}%  teste: {acc_test:.1f}%",
        f"// float32 | {len(data)} bytes",
        "",
        '#include "model.h"',
        "",
        "alignas(8) const unsigned char g_model[] = {",
    ]
    row = []
    for i, b in enumerate(data):
        row.append(f"0x{b:02x}")
        if len(row) == 16 or i == len(data) - 1:
            lines.append("  " + ", ".join(row) + ("," if i < len(data) - 1 else ""))
            row = []
    lines += ["};", "", f"const int g_model_len = {len(data)};", ""]
    return "\n".join(lines)

with open("main/model_data.cc", "w") as f:
    f.write(to_c_array(tflite_bytes, acc_train*100, acc_val*100, acc_test*100))
print("  Salvo: main/model_data.cc")

# test_data.h com 20 amostras (10 normal + 10 anomalia)
idx_normal  = np.where(y_test == 0)[0][:10]
idx_anomaly = np.where(y_test == 1)[0][:10]
idx_sel = np.concatenate([idx_normal, idx_anomaly])
np.random.shuffle(idx_sel)
samples = X_test[idx_sel]
labels  = y_test[idx_sel].astype(int)

td_lines = [
    "#ifndef TEST_DATA_H_",
    "#define TEST_DATA_H_",
    "",
    "// Amostras MFCC normalizadas — compatíveis com modelo MFCC+MLP",
    f"#define NUM_TEST_SAMPLES {len(samples)}",
    "",
    f"static const float test_samples[{len(samples)}][{N_COEFFS}] = {{",
]
for row in samples:
    vals = ", ".join(f"{v:.6f}f" for v in row)
    td_lines.append(f"    {{{vals}}},")
td_lines += [
    "};",
    "",
    f"static const int expected_labels[{len(labels)}] = {{",
    "    " + ", ".join(str(l) for l in labels),
    "};",
    "",
    "#endif  // TEST_DATA_H_",
    "",
]
with open("main/test_data.h", "w") as f:
    f.write("\n".join(td_lines))
print("  Salvo: main/test_data.h")

print("\n" + "=" * 55)
print("  TFLite: model_mfcc_mlp.tflite")
print("Para gravar no ESP32-S3:")
print("  idf.py build && idf.py flash monitor")
print("=" * 55)
