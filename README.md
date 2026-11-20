# GPT — Grading Prediction of Tumors

Aplicação desktop de **Diagnóstico Auxiliado por Computador (CAD)** que utiliza uma Rede Neural Convolucional (CNN) customizada para auxiliar na detecção de tumores cerebrais em exames de Ressonância Magnética (RM) do crânio.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_%26_Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras Tuner](https://img.shields.io/badge/Keras_Tuner-D00000?style=flat-square)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-2B5B84?style=flat-square)
![Windows](https://img.shields.io/badge/Deploy-Windows_.exe-0078D6?style=flat-square&logo=windows&logoColor=white)

---

## 📋 Sobre o Projeto

O **GPT (Grading Prediction of Tumors)** foi projetado para oferecer uma triagem rápida e acessível de imagens médicas em poucos cliques, integrando um modelo de *Deep Learning* a uma interface gráfica desktop sem exigir conhecimentos técnicos de programação do usuário final.

> **⚠️ AVISO IMPORTANTE:** Esta ferramenta é um **projeto acadêmico** e **não substitui um diagnóstico médico profissional**. Os resultados servem exclusivamente como apoio e segunda opinião, devendo ser interpretados por um especialista. Além disso, o modelo foi treinado apenas com **ressonâncias magnéticas do crânio** e não deve ser aplicado a outras modalidades de exame.

---

## 🧠 Arquitetura da CNN e Otimização

O modelo foi construído e otimizado em duas etapas utilizando **Keras Tuner (Hyperband)** ao longo de 30 *trials*, priorizando tanto a capacidade de generalização quanto a redução do tamanho do arquivo para distribuição em desktop:

* **Base de Dados:** *Br35H :: Brain Tumor Detection 2020* (3.000 imagens de RM divididas em **2.400 treino**, **300 validação** e **300 teste**).
* **Pré-processamento:** Redimensionamento automático para **224×224×3 px**, normalização dos pixels para o intervalo `[0, 1]` e treinamento por 100 épocas com `batch_size = 32`.
* **Compressão de Parâmetros:** Substituição da camada `Flatten` por **`GlobalAveragePooling2D`**, reduzindo drasticamente o volume de parâmetros treináveis, mitigando *overfitting* e tornando o modelo leve para empacotamento.
* **Configuração Final Selecionada:**
  * **Extração de Características:** 4 camadas `Conv2D` (64, 96, 64 e 96 filtros de tamanho 3×3, `padding='same'`, ativação `ReLU`) intercaladas com `MaxPooling2D (2×2)`.
  * **Classificador:** `GlobalAveragePooling2D` → 1 camada densa oculta (256 neurônios, `ReLU`) → `Dropout (0.2)` → saída `Sigmoid`.
  * **Treinamento e Decisão:** Função de perda `binary_crossentropy`, otimizador `Adam` (`lr = 0.001`) e **limiar de confiança ≥ 0,95** na interface gráfica para classificar uma amostra como positiva (*"Tumor detectado"*).

---

## 📊 Resultados no Conjunto de Teste (300 imagens)

Avaliado em 300 imagens inéditas (150 com tumor e 150 sem tumor), o classificador atingiu **90,67% de acurácia global**, registrando **150 Verdadeiros Negativos**, **122 Verdadeiros Positivos**, **28 Falsos Negativos** e **0 Falsos Positivos** (100% de precisão nas predições positivas):

| Classe | Precisão | Revocação (Recall) | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Sem Tumor (`no`)** | 0,84 | **1,00** | 0,91 | 150 |
| **Com Tumor (`yes`)** | **1,00** | 0,81 | 0,90 | 150 |
| **Média Ponderada** | **0,92** | **0,91** | **0,91** | **300** |

---

## 💻 Pré-requisitos e Instalação

### Opção 1 — Executável Standalone (Windows)
* **Sistema Operacional:** Windows 10 ou superior
* **Processador:** Intel Core i5 (4ª geração) | AMD Ryzen 3 ou equivalente
* **Memória RAM:** 8 GB ou mais
* **Armazenamento:** 500 MB de espaço livre em disco

1. Baixe o arquivo `gpt.exe` disponível na raiz deste repositório (ou na aba *Releases*).
2. Salve o executável em uma pasta local e dê um duplo clique para iniciar *(caso o Windows SmartScreen exiba um aviso de segurança, clique em **"Mais informações"** → **"Executar assim mesmo"**)*.

### Opção 2 — Execução via Código-Fonte (Python)
```bash
git clone https://github.com/paulorbr/GPT---Grading-Prediction-of-Tumors-.git
cd GPT---Grading-Prediction-of-Tumors-
pip install -r requirements.txt
python interface/app.py
