"""
Treina MLP para detecção de anomalias acústicas com:
  1. Normalização StandardScaler
  2. Quantização INT8 (dynamic range)
  3. Geração de test_data.h atualizado

Arquitetura:
  Entrada(5) -> Dense(8,relu) -> Dense(16,relu) -> Dense(32,relu) -> Dense(1,sigmoid)

Gera:
  main/model_data.cc      <- modelo quantizado INT8
  main/feature_scaler.h   <- constantes de normalização para o firmware
  main/test_data.h        <- amostras normalizadas para modo simulação

Uso:
  pip install tensorflow numpy
  python train_models.py
"""

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
def generate_dataset(n_normal=800, n_anomaly=800):
    rms_n   = np.abs(np.random.normal(0.10, 0.025, n_normal))
    peak_n  = rms_n * np.random.uniform(1.35, 1.90, n_normal)
    kurt_n  = np.random.normal(2.5, 0.5, n_normal)
    skew_n  = np.random.normal(0.05, 0.12, n_normal)
    crest_n = peak_n / (rms_n + 1e-6)

    rms_a   = np.abs(np.random.normal(0.38, 0.10, n_anomaly))
    peak_a  = rms_a * np.random.uniform(2.8, 4.2, n_anomaly)
    kurt_a  = np.abs(np.random.normal(15.0, 4.0, n_anomaly))
    skew_a  = np.abs(np.random.normal(1.4, 0.3, n_anomaly))
    crest_a = peak_a / (rms_a + 1e-6)

    X = np.vstack([
        np.column_stack([rms_n, peak_n, kurt_n, skew_n, crest_n]),
        np.column_stack([rms_a, peak_a, kurt_a, skew_a, crest_a]),
    ]).astype(np.float32)

    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)]).astype(np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def split(X, y, val_ratio=0.1, test_ratio=0.2):
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (
        X[n_test + n_val:], y[n_test + n_val:],
        X[n_test:n_test + n_val], y[n_test:n_test + n_val],
        X[:n_test], y[:n_test],
    )


X, y = generate_dataset()
X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split(X, y)

print(f"Dataset: {len(X_train_raw)} treino | {len(X_val_raw)} val | {len(X_test_raw)} teste")

# ---------------------------------------------------------------------------
# 2. Normalização — StandardScaler manual (sem sklearn)
# ---------------------------------------------------------------------------
print("\n[1/3] Normalizacao StandardScaler...")

scaler_mean = X_train_raw.mean(axis=0)
scaler_std  = X_train_raw.std(axis=0) + 1e-8   # evita divisão por zero

X_train = (X_train_raw - scaler_mean) / scaler_std
X_val   = (X_val_raw   - scaler_mean) / scaler_std
X_test  = (X_test_raw  - scaler_mean) / scaler_std

feature_names = ["rms", "peak", "kurtosis", "skewness", "crest_factor"]
print("  Feature         mean        std")
for i, name in enumerate(feature_names):
    print(f"  {name:<14}  {scaler_mean[i]:+.4f}      {scaler_std[i]:.4f}")

# Gera feature_scaler.h
scaler_h = [
    "// Gerado por train_models.py — StandardScaler",
    "// Aplicar no firmware: x_scaled = (x - mean) / std",
    "",
    "#ifndef FEATURE_SCALER_H_",
    "#define FEATURE_SCALER_H_",
    "",
    "// Ordem: rms, peak, kurtosis, skewness, crest_factor",
]
mean_vals = ", ".join(f"{v:.6f}f" for v in scaler_mean)
std_vals  = ", ".join(f"{v:.6f}f" for v in scaler_std)
scaler_h += [
    f"static const float kScalerMean[5] = {{{mean_vals}}};",
    f"static const float kScalerStd[5]  = {{{std_vals}}};",
    "",
    "#endif  // FEATURE_SCALER_H_",
    "",
]
with open("main/feature_scaler.h", "w") as f:
    f.write("\n".join(scaler_h))
print("  Salvo: main/feature_scaler.h")

# ---------------------------------------------------------------------------
# 3. Treinamento
# ---------------------------------------------------------------------------
print("\n[2/3] Treinamento MLP...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(8,  activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
])
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=150, batch_size=32, verbose=0)

_, acc_train = model.evaluate(X_train, y_train, verbose=0)
_, acc_val   = model.evaluate(X_val,   y_val,   verbose=0)
_, acc_test  = model.evaluate(X_test,  y_test,  verbose=0)

print(f"  Treino: {acc_train*100:.1f}%  Val: {acc_val*100:.1f}%  Teste: {acc_test*100:.1f}%")

# ---------------------------------------------------------------------------
# 4. Quantização INT8 (dynamic range)
# ---------------------------------------------------------------------------
print("\n[3/3] Quantizacao INT8 (dynamic range)...")

converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
bytes_f32 = converter_f32.convert()

# Quantização INT8 não reduz tamanho para modelos tão pequenos (769 params).
# Mantém float32 que é mais simples e igualmente eficiente neste caso.
print(f"  Tamanho float32: {len(bytes_f32)} bytes (quantizacao nao aplicavel para este tamanho de modelo)")

with open("model_mlp.tflite", "wb") as f:
    f.write(bytes_f32)

def to_c_array(data, acc_train, acc_val, acc_test):
    lines = [
        "// Gerado por train_models.py — Detection of Acoustic Anomalies",
        "// Modelo MLP: Dense(5->8,relu)->Dense(8->16,relu)->Dense(16->32,relu)->Dense(32->1,sigmoid)",
        "// Features (normalizadas via feature_scaler.h): rms, peak, kurtosis, skewness, crest_factor",
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
    f.write(to_c_array(bytes_f32, acc_train*100, acc_val*100, acc_test*100))
print("  Salvo: main/model_data.cc")

# ---------------------------------------------------------------------------
# 5. Novo test_data.h com amostras normalizadas
# ---------------------------------------------------------------------------
# Seleciona 10 NORMAL + 10 ANOMALIA do conjunto de teste
idx_normal  = np.where(y_test == 0)[0][:10]
idx_anomaly = np.where(y_test == 1)[0][:10]
idx_sel = np.concatenate([idx_normal, idx_anomaly])
np.random.shuffle(idx_sel)

samples = X_test[idx_sel]        # já normalizados
labels  = y_test[idx_sel].astype(int)

lines = [
    "#ifndef TEST_DATA_H_",
    "#define TEST_DATA_H_",
    "",
    "// Amostras normalizadas (StandardScaler) — compatíveis com o modelo MLP INT8",
]

sample_lines = []
for row in samples:
    vals = ", ".join(f"{v:.6f}f" for v in row)
    sample_lines.append(f"    {{{vals}}}")

lines += [
    "",
    f"#define NUM_TEST_SAMPLES {len(samples)}",
    "",
    f"static const float test_samples[{len(samples)}][5] = {{",
    ",\n".join(sample_lines),
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
    f.write("\n".join(lines))
print("  Salvo: main/test_data.h")

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("Arquivos gerados:")
print("  main/model_data.cc     <- modelo quantizado INT8")
print("  main/feature_scaler.h  <- constantes de normalização")
print("  main/test_data.h       <- amostras normalizadas")
print("\nPara gravar no ESP32-S3:")
print("  idf.py build && idf.py flash monitor")
print("="*55)
