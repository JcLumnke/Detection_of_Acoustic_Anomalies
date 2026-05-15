# ⚙️ Detection of Acoustic Anomalies with TinyML on ESP32-S3

Sistema embarcado de detecção de anomalias acústicas utilizando **TensorFlow Lite Micro**, **ESP32-S3** e técnicas de **TinyML** para inferência em tempo real.

O projeto executa um modelo de Machine Learning embarcado capaz de identificar padrões acústicos anormais a partir de features extraídas de sinais de vibração/áudio industrial.

---

# 📸 Simulação no Wokwi

```text
simulation.png
```

![Simulação Wokwi](simulation.png)

---

# 🎯 Objetivo do Projeto

Este projeto demonstra a implementação de um pipeline completo de IA embarcada:

- Pré-processamento de sinais
- Extração de features acústicas
- Normalização estatística das features
- Inferência TinyML em microcontrolador
- Classificação de anomalias em tempo real
- Simulação no Wokwi
- Compatibilidade com hardware físico ESP32-S3

O sistema foi projetado para aplicações de:

- Manutenção preditiva
- Monitoramento industrial
- Detecção de falhas mecânicas
- Análise de vibração
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
- Inferência float32
- Acurácia: 99.1% treino / 99.0% val / 99.3% teste
- Features brutas sem normalização

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
- Features com escalas muito diferentes (RMS ~0.1, kurtosis ~15) sem normalização
- Capacidade expressiva limitada — modelo linear sem camadas ocultas
- `test_data.h` com amostras brutas não normalizadas

---

## V1.2 — MLP com Normalização (versão atual)

Evolução para uma rede neural multicamada (MLP) com pré-processamento de normalização embarcado.

**Arquitetura:**

```
Entrada (5 features)
    ↓  [normalização StandardScaler]
Dense(8,  relu)   →  48 parâmetros
    ↓
Dense(16, relu)   → 144 parâmetros
    ↓
Dense(32, relu)   → 544 parâmetros
    ↓
Dense(1, sigmoid) →  33 parâmetros  ← regressão logística na saída
```

**Características:**
- 769 parâmetros treináveis
- 5.664 bytes em flash
- Inferência float32
- Acurácia: 100% treino / 100% val / 100% teste
- Features normalizadas via StandardScaler antes da inferência

**O que mudou em relação à V1.0:**

| Aspecto | V1.0 | V1.2 |
|---|---|---|
| Tipo de modelo | Regressão logística | MLP (3 camadas ocultas) |
| Parâmetros | 6 | 769 |
| Tamanho em flash | 680 bytes | 5.664 bytes |
| Normalização | Não | StandardScaler embarcado |
| Camadas ocultas | 0 | 3 (8 → 16 → 32 neurônios) |
| Ops TFLite Micro | FC + Logistic | FC + ReLU + Logistic |
| `test_data.h` | Amostras brutas | Amostras normalizadas |

**Por que MLP em vez de CNN?**

As 5 features estatísticas (RMS, Peak, Kurtosis, Skewness, Crest Factor) já são extraídas antes da inferência. Uma CNN seria adequada se o modelo recebesse o sinal bruto de áudio (os 480 samples) para detectar padrões locais. Como o pré-processamento já transforma o sinal em um vetor compacto de features, a MLP é a escolha correta — mais leve e igualmente eficaz.

**Nota sobre quantização INT8:**
A quantização foi avaliada para reduzir o tamanho do modelo, mas não trouxe benefício mensurável: o modelo possui apenas 769 parâmetros (~3KB de pesos), e o overhead do formato flatbuffer do TFLite domina o tamanho total. Quantização INT8 só é vantajosa em modelos com centenas de milhares de parâmetros.

---

# 🧠 Arquitetura do Modelo (V1.2)

## Features utilizadas

| Feature | Descrição |
|---|---|
| RMS | Energia média do sinal (DC removido) |
| Peak | Pico máximo absoluto |
| Kurtosis | Detecção de impulsos mecânicos |
| Skewness | Assimetria estatística do sinal |
| Crest Factor | Relação pico/RMS |

## Normalização StandardScaler

Antes de entrar no modelo, cada feature é normalizada com os parâmetros calculados no treinamento e embarcados em `feature_scaler.h`:

```
x_normalizado = (x - média) / desvio_padrão
```

| Feature | Média | Desvio Padrão |
|---|---|---|
| RMS | 0.2416 | 0.1589 |
| Peak | 0.7495 | 0.6439 |
| Kurtosis | 8.8263 | 6.8577 |
| Skewness | 0.7256 | 0.7065 |
| Crest Factor | 2.5631 | 0.9842 |

---

# 🔬 Pipeline de Processamento

## 1. Captura do sinal

O sistema recebe frames de 480 amostras via I2S (microfone INMP441) ou dataset simulado contendo:

- ruído normal
- transientes
- impulsos mecânicos
- assinaturas acústicas

## 2. Pré-processamento HHT + UKF

O pipeline aplica:

- EMA (Exponential Moving Average)
- HHT-inspired filtering
- suavização adaptativa tipo UKF

Objetivos:

- remover drift
- reduzir ruído
- preservar transientes de falha

## 3. Extração de Features

As 5 features estatísticas são calculadas para cada frame de áudio.

## 4. Normalização

As features são normalizadas com os parâmetros do StandardScaler embarcados antes de entrar na rede.

## 5. Inferência TinyML

O vetor normalizado de 5 features percorre as 3 camadas ocultas do MLP e produz uma probabilidade de anomalia entre 0.0 e 1.0.

Classificação:

- `NORMAL` — probabilidade < threshold
- `ANOMALIA` — probabilidade ≥ threshold

---

# 📦 Estrutura do Projeto

```text
detection_of_acoustic_anomalies/
│
├── main/
│   ├── main.cc                # Ponto de entrada da aplicação
│   ├── main_functions.cc      # Lógica principal (Setup/Loop)
│   ├── microphone.cc          # Driver e captura do microfone I2S
│   ├── microphone.h           # Cabeçalho e pinagem do microfone
│   ├── model_data.cc          # Array do modelo TFLite (MLP V1.2)
│   ├── model.h                # Cabeçalho do modelo
│   ├── feature_scaler.h       # Constantes StandardScaler para normalização
│   ├── test_data.h            # Amostras normalizadas para modo simulação
│   ├── constants.h            # Definições de thresholds e parâmetros
│   └── output_handler.cc      # Gerenciamento de logs e saídas
│
├── train_models.py            # Script de treinamento (gera model_data.cc, feature_scaler.h, test_data.h)
├── model_mlp.tflite           # Modelo TFLite gerado pelo treinamento
├── build/                     # Arquivos de compilação (gerados automaticamente)
├── simulation.png             # Imagem da simulação Wokwi
├── README.md                  # Documentação do projeto
└── CMakeLists.txt             # Configuração do Build System (ESP-IDF)
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

## Modo Sensor Real

```text
=================================
RMS          : 0.38412
Peak         : 0.91034
Kurtosis     : 14.23100
Skewness     : 1.41200
Crest Factor : 3.76500
Probabilidade: 0.96700
Inferido     : ANOMALIA
```

---

# ⚙️ Modos de Operação

O projeto possui dois modos de funcionamento controlados por uma constante no arquivo `main_functions.cc`:

```cpp
#define SIMULATION_MODE true  // Habilita Modo Simulação
// ou
#define SIMULATION_MODE false // Habilita Modo Sensor Real
```

| Modo | Fonte de dados | Normalização |
|---|---|---|
| Simulação | `test_data.h` (amostras pré-normalizadas) | Já aplicada nas amostras |
| Sensor Real | Microfone INMP441 via I2S | Aplicada em tempo real via `feature_scaler.h` |

---

# 🔁 Retreinando o Modelo

Para regenerar o modelo (ex.: com novos dados ou nova arquitetura):

```bash
pip install tensorflow numpy
python train_models.py
```

O script gera automaticamente:
- `main/model_data.cc` — modelo atualizado
- `main/feature_scaler.h` — novos parâmetros de normalização
- `main/test_data.h` — novas amostras de simulação

---

# 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| ESP-IDF | Framework embarcado |
| ESP32-S3 | Microcontrolador |
| TensorFlow Lite Micro | Inferência TinyML |
| Wokwi | Simulação |
| Python + TensorFlow/Keras | Treinamento do modelo |
| NumPy | Processamento numérico e geração do dataset |
| TinyML | IA embarcada |

---

# 🔌 Conexões de Hardware (INMP441)

| Pino do Sensor | Pino ESP32-S3 | Função | Descrição |
|:---|:---|:---|:---|
| **VDD** | 3.3V | VCC | Alimentação |
| **GND** | GND | GND | Aterramento |
| **L/R** | GND | — | Canal esquerdo |
| **SCK** | **GPIO 14** | `bclk` | Serial Clock (I2S) |
| **WS** | **GPIO 15** | `ws` | Word Select (I2S) |
| **SD** | **GPIO 13** | `din` | Serial Data |

---

# 🚀 Como Compilar

```bash
git clone <repositorio>
cd detection_of_acoustic_anomalies
idf.py build
idf.py flash monitor
```

---

# 📊 Dataset Utilizado

O projeto utiliza dados alinhados com distribuições estatísticas inspiradas em:

- DCASE Task 2
- MIMII Dataset
- sinais industriais sintéticos calibrados

As amostras incluem condições normais, falhas impulsivas, assinaturas de rolamentos e eventos transientes.

---

# 🔮 Próximos Passos

- Streaming serial de features para dashboard externo
- Coleta de dados reais para refinamento do modelo
- Features espectrais via FFT (Spectral Centroid, bandas de frequência)
- Threshold adaptativo por média móvel

---

# 👨‍🏫 Orientação Acadêmica

Professor orientador: Rodrigo Kobashikawa Rosa

---

# 👨‍💻 Autores

Projeto desenvolvido por:

- Julio Cesar Lumke
- Emanoel Spanhol

Áreas de pesquisa:

- TinyML
- Edge AI
- Sistemas embarcados inteligentes
- Detecção de anomalias acústicas
- Inteligência Artificial embarcada

---

# 📜 Licença

Projeto acadêmico e educacional desenvolvido para fins de estudo, pesquisa e experimentação em IA embarcada.
