// Gerado por train_models.py — StandardScaler
// Aplicar no firmware: x_scaled = (x - mean) / std

#ifndef FEATURE_SCALER_H_
#define FEATURE_SCALER_H_

// Ordem: rms, peak, kurtosis, skewness, crest_factor
static const float kScalerMean[5] = {0.241633f, 0.749500f, 8.826342f, 0.725575f, 2.563149f};
static const float kScalerStd[5]  = {0.158919f, 0.643905f, 6.857750f, 0.706485f, 0.984179f};

#endif  // FEATURE_SCALER_H_
