// Gerado por train_models.py — StandardScaler para 13 MFCCs
// Aplicar no firmware: x_scaled = (x - mean) / std

#ifndef FEATURE_SCALER_H_
#define FEATURE_SCALER_H_

// Ordem: MFCC[0] .. MFCC[12]
static const float kScalerMean[13] = {78.263756f, -20.169058f, -8.801491f, -6.274077f, -5.203150f, -2.749129f, -3.593590f, -0.746911f, -0.469227f, -1.431739f, -0.611803f, 0.031752f, -0.283743f};
static const float kScalerStd[13]  = {27.401705f, 12.689322f, 6.596492f, 4.272541f, 3.708165f, 3.706525f, 2.913760f, 2.795519f, 2.810439f, 2.631772f, 2.495861f, 2.167337f, 1.807310f};

#endif  // FEATURE_SCALER_H_
