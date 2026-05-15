// Gerado por train_models.py — StandardScaler para 13 MFCCs
// Aplicar no firmware: x_scaled = (x - mean) / std

#ifndef FEATURE_SCALER_H_
#define FEATURE_SCALER_H_

// Ordem: MFCC[0] .. MFCC[12]
static const float kScalerMean[13] = {104.144119f, -35.171471f, 1.633677f, -2.129519f, -1.297641f, -2.869311f, -1.936405f, -1.692654f, -0.951284f, -1.255010f, -1.275250f, -1.355234f, -1.014330f};
static const float kScalerStd[13]  = {9.640083f, 3.936101f, 2.860840f, 2.860904f, 2.847424f, 2.442340f, 1.859309f, 1.537129f, 1.612931f, 1.917992f, 2.110828f, 1.944760f, 1.775904f};

#endif  // FEATURE_SCALER_H_
