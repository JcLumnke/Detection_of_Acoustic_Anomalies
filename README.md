# ⚙️ Detection of Acoustic Anomalies with TinyML on ESP32-S3

Sistema embarcado de detecção de anomalias acústicas utilizando **TensorFlow Lite Micro**, **ESP32-S3** e técnicas de **TinyML** para inferência em tempo real.

O projeto executa um modelo de Machine Learning embarcado capaz de identificar padrões acústicos anormais a partir de features extraídas de sinais de vibração/áudio industrial.

---

# 📸 Simulação no Wokwi

![Simulação Wokwi](simulation.png)

---

# 🎯 Objetivo do Projeto

Este projeto demonstra a implementação de um pipeline completo de IA embarcada:

- Captura de áudio via microfone I2S
- Extração de features acústicas (MFCC)
- Normalização estatística embarcada
- Inferência TinyML em microcontrolador
- Classificação de anomalias em tempo real
- Simulação no Wokwi e deploy em hardware físico

O sistema foi projetado para aplicações de:

- Manutenção preditiva
- Monitoramento industrial
- Detecção de falhas mecânicas
- Edge AI / TinyML

---

# 📜 Histórico de Versões

## V1.0 — Regressão Logística (baseline)

Primeira versão do modelo embarcado. Arquitetura mínima para validar o pipeline completo de ponta a ponta no ESP32-S3.

**Arquitetura:**

```
Entrada (5 features) → FullyConnected(5→1) → Logistic → Probabilidade
```

**Características:**
- 6 parâmetros treináveis
- 680 bytes em flash
- Features: RMS, Peak, Kurtosis, Skewness, Crest Factor (brutas, sem normalização)
- Acurácia: 99.1% treino / 99.0% val / 99.3% teste (dataset sintético)

**Pesos aprendidos:**

| Feature | Peso (W) |
|---|---|
| RMS | +0.2233 |
| Peak | +0.5838 |
| Kurtosis | +0.5933 |
| Skewness | +2.2004 |
| Crest Factor | −1.0163 |
| Bias (b) | −2.7500 |

**Limitações identificadas:**
- Features com escalas muito diferentes sem normalização
- Modelo linear sem capacidade de aprender padrões complexos
- `test_data.h` com amostras brutas não normalizadas

---

## V1.2 — MLP com Normalização

Evolução para rede neural multicamada com pré-processamento de normalização embarcado.

**Arquitetura:**

```
Entrada (5 features)
    ↓  [StandardScaler]
Dense(8,  relu)   →  48 parâmetros
    ↓
Dense(16, relu)   → 144 parâmetros
    ↓
Dense(32, relu)   → 544 parâmetros
    ↓
Dense(1, sigmoid) →  33 parâmetros
```

**Características:**
- 769 parâmetros treináveis
- 5.664 bytes em flash
- Normalização StandardScaler embarcada em `feature_scaler.h`
- Acurácia: 100% treino / 100% val / 100% teste

**O que mudou em relação à V1.0:**

| Aspecto | V1.0 | V1.2 |
|---|---|---|
| Tipo de modelo | Regressão logística | MLP (3 camadas ocultas) |
| Parâmetros | 6 | 769 |
| Tamanho em flash | 680 bytes | 5.664 bytes |
| Normalização | Não | StandardScaler embarcado |
| Camadas ocultas | 0 | 3 (8 → 16 → 32 neurônios) |
| Ops TFLite Micro | FC + Logistic | FC + ReLU + Logistic |

**Nota sobre quantização INT8:**
Avaliada e descartada — com apenas 769 parâmetros o overhead do formato flatbuffer domina o tamanho total. Quantização INT8 só é vantajosa em modelos com centenas de milhares de parâmetros.

---

## Teste com Hardware Real — Problema de Calibração (entre V1.2 e V2.0)

Durante os testes com a placa física, identificou-se incompatibilidade entre o dataset sintético e o sensor real:

**Problema:** o dataset sintético treinava NORMAL com RMS ~0.10, mas o sensor em ambiente real apresentava RMS ~0.42–0.43. O modelo classificava tudo como ANOMALIA.

**Correção aplicada na V1.2:**

| Parâmetro | Antes | Depois |
|---|---|---|
| RMS NORMAL (treino) | ~0.10 | ~0.50 |
| RMS ANOMALIA (treino) | ~0.38 | ~0.75 |
| Threshold de silêncio | 0.40 | 0.45 |

**Lição aprendida:** features estatísticas como RMS, Kurtosis e Skewness não são suficientes para diferenciar motor saudável de ruído ambiente — ambos podem ter distribuições estatísticas similares. A solução foi migrar para features espectrais (MFCC).

---

## V2.0 — MFCC + MLP (versão atual)

Substituição completa do pipeline de features: as 5 features estatísticas foram trocadas por **13 coeficientes MFCC**, implementados diretamente no firmware em C.

**Motivação:** MFCC captura o timbre sonoro do motor (distribuição de energia nas frequências), que é muito mais discriminativo do que estatísticas de amplitude. Um motor saudável e um com falha têm assinaturas espectrais distintas mesmo com RMS similar.

**Pipeline completo:**

```
Microfone INMP441 (I2S @ 16kHz)
    ↓
Frame de 1024 amostras (64ms)
    ↓
Remoção de offset DC (subtração da média do frame)
    ↓
Filtro de silêncio (RMS_AC < 0.02 → descarta)
    ↓
Pré-ênfase (α=0.97)
    ↓
Janela Hamming
    ↓
FFT 1024 pontos
    ↓
Banco de 26 filtros Mel (20Hz – 8kHz)
    ↓
Log + DCT
    ↓
13 coeficientes MFCC
    ↓
StandardScaler
    ↓
MLP(13→16→32→1)
    ↓
Probabilidade instantânea [0.0 – 1.0]
    ↓
Média móvel (5 frames ≈ 5s)
    ↓
Decisão: média ≥ 0.70 → ANOMALIA
```

**Arquitetura do modelo:**

```
Entrada (13 MFCCs)
    ↓  [StandardScaler]
Dense(16, relu)   → 224 parâmetros
    ↓
Dense(32, relu)   → 544 parâmetros
    ↓
Dense(1, sigmoid) →  33 parâmetros
```

**Características:**
- 801 parâmetros treináveis
- Features: 13 coeficientes MFCC normalizados
- Frame: 1024 amostras @ 16kHz (64ms)
- FFT implementada em C (radix-2 Cooley-Tukey) no próprio firmware
- Suavização temporal: média móvel de 5 frames (~5 segundos) com threshold 0.70
- Acurácia: 91.9% treino / 86.5% val / 81.0% teste

> A acurácia menor que a V1.2 (sintética) é esperada e intencional: o dataset V2.0 foi
> calibrado para reproduzir a distribuição MFCC do sensor real, criando sobreposição
> genuína entre classes — o modelo é mais honesto e robusto em hardware real.

**O que mudou em relação à V1.2:**

| Aspecto | V1.2 | V2.0 |
|---|---|---|
| Features | 5 estatísticas (RMS, Peak...) | 13 MFCCs |
| Extração | Loop simples sobre o buffer | FFT → Mel → log → DCT |
| Buffer de áudio | 480 amostras (30ms) | 1024 amostras (64ms) |
| Discriminação | Amplitude / forma do sinal | Timbre / espectro de frequências |
| Remoção de DC | Não | Sim (média do frame subtraída antes do RMS) |
| Threshold de silêncio | RMS bruto ≥ 0.45 | RMS_AC < 0.02 |
| Decisão | P ≥ 0.50 (1 frame) | Média(5 frames) ≥ 0.70 |
| Arquivos novos | — | `mfcc.cc`, `mfcc.h` |
| Parâmetros do modelo | 769 | 801 |
| Robustez em hardware real | Baixa | Alta |

**Dataset de treinamento V2.0:**

Gerado com síntese de áudio calibrada empiricamente para reproduzir a distribuição MFCC do sensor real (INMP441):
- **Sinal base**: ruído branco + harmônicos (fundamental 80–200 Hz, 5 parciais) com escala U[0.5, 2.0]
  - Garante MFCC[1] na faixa −44..−31 (sensor real: −38 ± 5–8)
- **NORMAL**: sinal base normalizado × amplitude U[0.45, 0.70]
- **ANOMALIA**: sinal base + 3–10 impulsos aleatórios (amplitude 1.5–3.5×), normalizado × U[0.65, 0.90]

---

# 🧠 Arquitetura do Modelo (V2.0 — atual)

## Features: 13 coeficientes MFCC

| Coeficiente | Nome | Representa |
|---|---|---|
| MFCC[0] | Energia | Energia global (espectro médio) |
| MFCC[1] | Inclinação | Inclinação do envelope espectral |
| MFCC[2] | Curvatura | Curvatura do envelope espectral |
| MFCC[3–12] | Forma-3..12 | Detalhes finos do timbre sonoro |

## Parâmetros MFCC

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem | 16.000 Hz |
| Tamanho do frame | 1024 amostras (64ms) |
| Filtros Mel | 26 (20Hz – 8kHz) |
| Coeficientes DCT | 13 |
| Pré-ênfase | α = 0.97 |
| Janela | Hamming |

---

# 🔬 Pipeline de Processamento (V2.0)

**1. Captura:** 1024 amostras via I2S (INMP441 @ 16kHz)

**2. Remoção de offset DC:** subtrai a média do frame do sinal antes do RMS
- O INMP441 apresenta bias DC que mantinha o RMS bruto em ~0.5 mesmo sem som

**3. Filtro de silêncio:** descarta frame se RMS_AC < 0.02

**4. MFCC (`mfcc.cc`):**
- Pré-ênfase → Hamming → FFT 1024pts → Espectro de potência → 26 filtros Mel → log → DCT → 13 coeficientes

**5. Normalização:** StandardScaler com parâmetros de `feature_scaler.h`

**6. Inferência:** MLP(13→16→32→1) via TFLite Micro

**7. Suavização temporal:** média móvel das últimas 5 probabilidades (~5 segundos)

**8. Decisão:** média ≥ 0.70 → ANOMALIA (reduz falsos positivos de frames isolados)

---

# 📦 Estrutura do Projeto

```text
detection_of_acoustic_anomalies/
│
├── main/
│   ├── main.cc                # Ponto de entrada
│   ├── main_functions.cc      # Lógica principal (Setup/Loop)
│   ├── mfcc.cc                # Implementação MFCC (FFT + Mel + DCT)
│   ├── mfcc.h                 # Parâmetros e interface MFCC
│   ├── microphone.cc          # Driver I2S (1024 samples @ 16kHz)
│   ├── microphone.h           # Interface do microfone
│   ├── model_data.cc          # Array do modelo TFLite (MFCC+MLP V2.0)
│   ├── model.h                # Cabeçalho do modelo
│   ├── feature_scaler.h       # StandardScaler para 13 MFCCs
│   ├── test_data.h            # Amostras MFCC normalizadas (simulação)
│   └── output_handler.cc      # Logs e saídas
│
├── train_models.py            # Treinamento: síntese de áudio → MFCC → MLP
├── model_mfcc_mlp.tflite      # Modelo TFLite V2.0
├── build/                     # Gerado automaticamente
├── simulation.png             # Simulação Wokwi
├── README.md
└── CMakeLists.txt
```

---

# 🧪 Saída Esperada

## Modo Simulação

```text
=================================
Sample 0
Probabilidade: 0.9981
Esperado : ANOMALIA
Inferido : ANOMALIA
```

## Modo Sensor Real (V2.0)

```text
=================================
Audio [#####...............] 0.052
MFCC[00] Energia     : 98.3214
MFCC[01] Inclinacao  : -37.8412
MFCC[02] Curvatura   : 2.1034
MFCC[03] Forma-3     : -1.8762
...
MFCC[12] Forma-12    : -0.9421
Probabilidade        : 0.8214
Media (5 frames)     : 0.7341
Inferido             : *** ANOMALIA ***
```

A barra `[###...]` representa o nível de áudio AC (20 `#` = RMS_AC ≥ 0.20).

---

# ⚙️ Modos de Operação

```cpp
#define SIMULATION_MODE true  // dataset de teste pré-computado
#define SIMULATION_MODE false // microfone real com extração MFCC
```

| Modo | Fonte | MFCC |
|---|---|---|
| Simulação | `test_data.h` (pré-normalizado) | Pré-calculado no Python |
| Sensor Real | INMP441 via I2S | Calculado em tempo real no ESP32-S3 |

---

# 🔁 Retreinando o Modelo

```bash
pip install tensorflow numpy scipy
python train_models.py
```

Gera automaticamente:
- `main/model_data.cc` — novo modelo
- `main/feature_scaler.h` — parâmetros de normalização dos MFCCs
- `main/test_data.h` — amostras de simulação atualizadas

---

# 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| ESP-IDF | Framework embarcado |
| ESP32-S3 | Microcontrolador |
| TensorFlow Lite Micro | Inferência TinyML |
| Wokwi | Simulação |
| Python + TensorFlow/Keras | Treinamento |
| NumPy + SciPy | Processamento de sinais e MFCC |
| TinyML | IA embarcada |

---

# 🔌 Conexões de Hardware (INMP441)

| Pino do Sensor | Pino ESP32-S3 | Função |
|:---|:---|:---|
| **VDD** | 3.3V | Alimentação |
| **GND** | GND | Aterramento |
| **L/R** | GND | Canal esquerdo |
| **SCK** | **GPIO 14** | Serial Clock (I2S) |
| **WS** | **GPIO 15** | Word Select (I2S) |
| **SD** | **GPIO 13** | Serial Data |

---

# 🚀 Como Compilar

```bash
git clone <repositorio>
cd detection_of_acoustic_anomalies
get_idf                          # ativa o ambiente ESP-IDF
idf.py build && idf.py flash monitor
```

---

# 📊 Dataset

Gerado sinteticamente com distribuições calibradas para o sensor real:

- **NORMAL**: ruído branco + harmônicos (80–200Hz, escala U[0.5, 2.0]) × amplitude U[0.45, 0.70]
  - MFCC[1] resultante: −44..−31 (sensor real: −38 ± 5–8) ✓
- **ANOMALIA**: mesmo sinal com 3–10 impulsos mecânicos aleatórios sobrepostos × amplitude U[0.65, 0.90]

Dataset: 2.000 amostras (1.000 NORMAL + 1.000 ANOMALIA), split 70/10/20%.

Inspirado em: DCASE Task 2, MIMII Dataset, sinais industriais sintéticos.

---

# 🔮 Próximos Passos

- Coleta de dados reais de motor para fine-tuning do modelo
- Delta-MFCC e Delta-Delta para capturar dinâmica temporal
- Janela deslizante com overlap para detecção contínua
- Dashboard serial para visualização em tempo real
- Quantização INT8 com dataset real (atualmente float32)

---

# 👨‍🏫 Orientação Acadêmica

Professor orientador: Rodrigo Kobashikawa Rosa

---

# 👨‍💻 Autores

- Julio Cesar Lumke
- Emanoel Spanhol

Áreas: TinyML · Edge AI · Sistemas embarcados inteligentes · Detecção de anomalias acústicas

---

# 📜 Licença

Projeto acadêmico desenvolvido para fins de estudo, pesquisa e experimentação em IA embarcada.
