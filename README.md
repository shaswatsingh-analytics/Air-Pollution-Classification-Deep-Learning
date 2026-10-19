# 🌫️ Deep Learning-Based Classification of Air Pollution Levels from Sky Images

[Python](https://www.google.com/search?q=https%3A%2F%2Fwww.python.org%2F)
[TensorFlow](https://www.google.com/search?q=https%3A%2F%2Fwww.tensorflow.org%2F)
[Keras](https://www.google.com/search?q=https%3A%2F%2Fkeras.io%2F)
[Dataset](https://www.google.com/search?q=https%3A%2F%2Fwww.kaggle.com%2F)
[License](https://www.google.com/search?q=LICENSE)
[](#)

> An end-to-end computer vision and multimodal deep learning framework to classify ambient air pollution levels into six standard categories using outdoor sky imagery, transfer learning architectures, explainable AI (Grad-CAM), and environmental sensor fusion.

---

## 📌 Executive Summary

Air pollution is among the most pressing global public health and environmental challenges, directly contributing to cardiopulmonary diseases and atmospheric degradation. While conventional assessment relies on municipal ground-monitoring stations equipped with specialized chemical sensors, these stations are geographically sparse and capital-intensive to maintain.

This project investigates whether visual cues in ambient outdoor sky and urban imagery (such as haze, atmospheric turbidity, color attenuation, and horizon contrast) can reliably indicate air pollution severity. Leveraging the **Air Pollution Image Dataset from India and Nepal (APIN)**, we benchmark custom Convolutional Neural Networks (CNNs) against pretrained deep transfer learning architectures (**ResNet50**, **EfficientNetB0**), evaluate explainability via **Grad-CAM**, and explore multimodal integration with six core pollutant metrics ($PM_{2.5}$, $PM_{10}$, $O_3$, $CO$, $SO_2$, $NO_2$).

> **Disclaimer**: This system is a research prototype developed for academic exploration, comparative benchmarking, and public awareness. It is not intended as a certified regulatory monitoring system or a direct substitute for official Air Quality Index (AQI) monitoring stations.

---

## 🚀 Key Highlights & Capabilities

- **Six-Tier AQI Category Classification**: Maps real-world outdoor images to six standardized environmental health risk tiers (`Good` to `Severe`).
- **Comprehensive Architecture Benchmarking**: Systematic comparison of a custom baseline CNN, ResNet50, and EfficientNetB0 (initial transfer learning vs. fine-tuned).
- **High-Accuracy Vision Baseline**: Fine-tuned EfficientNetB0 achieves **94.69% test accuracy** and **95.07% Macro F1-score** on image-only classification.
- **Explainable AI via Grad-CAM**: Generates class-activation heatmaps to visualize the salient spatial features (sky gradients, skyline contrasts, haze dispersion) guiding model inferences.
- **Multimodal Sensor-Image Fusion**: Combines visual feature representations with environmental sensor readings, yielding **99.43% test accuracy** on usable paired records.
- **Reproducible Academic Pipeline**: Fully modularized data splits (80:10:10), pre-processing workflows, and standardized evaluation protocols.

---

## 📂 Dataset Specification

The project utilizes the **Air Pollution Image Dataset from India and Nepal (APIN)**, curated to capture diverse South Asian meteorological and urban atmospheric conditions.

### Dataset Parameters

| Property | Value / Specification |
| :--- | :--- |
| **Dataset Source** | Air Pollution Image Dataset from India and Nepal (APIN) — Kaggle |
| **Total Images** | 12,240 images |
| **Resolution & Format** | $224 \times 224$ pixels, 3-channel RGB |
| **Target Classes** | 6 discrete pollution categories |
| **Data Partitioning** | 80% Train : 10% Validation : 10% Test (Stratified) |
| **Training Set Size** | 9,792 images |
| **Validation Set Size** | 1,224 images |
| **Test Set Size** | 1,224 images |

### Target Classification Categories

1. **Good**: Clear sky, minimal atmospheric scattering, high visual range.
2. **Moderate**: Subtle haze, acceptable air quality with slight horizon blurring.
3. **Unhealthy for Sensitive Groups**: Noticeable particulate accumulation, mild contrast reduction.
4. **Unhealthy**: Prominent smog/haze layer, substantial color shifting and sky graying.
5. **Very Unhealthy**: Heavy atmospheric turbidity, high obscurity of distant structures.
6. **Severe**: Extreme particulate density, severe visual horizon cutoff and low contrast.

---

## 🛠️ Technology Stack & Tools

| Component | Technology | Rationale & Application |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Primary language for pipeline scripting, data loading, and modeling. |
| **Deep Learning** | TensorFlow 2.x & Keras | Model architecture building, distributed GPU training, and evaluation. |
| **Pretrained Models** | EfficientNetB0, ResNet50 | ImageNet-pretrained feature extractors for transfer learning and fine-tuning. |
| **Data Science** | NumPy, Pandas | Array manipulations, dataset slicing, and tabular pollutant processing. |
| **Visualization** | Matplotlib, Seaborn | Training curves, confusion matrices, and benchmark performance charts. |
| **Explainable AI** | Grad-CAM (Keras) | Gradient-weighted Class Activation Mapping for visual feature localization. |
| **Environment** | JupyterLab / Colab | Interactive development, experimental checkpointing, and documentation. |

---

## 🧠 Methodology & Experimental Architecture

```
[ Input: 224x224 RGB Sky Image ]
              │
       ┌──────┴──────┐
       ▼             ▼
[ Baseline CNN ]  [ Pretrained Backbones ]
 (From Scratch)    ├── ResNet50 (Transfer Learning)
                   ├── EfficientNetB0 (Feature Extractor)
                   └── EfficientNetB0 (Fine-Tuning Champion)
                             │
                             ▼
              [ 6-Class Softmax Classification ]
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                                           ▼
[ Explainability: Grad-CAM ]          [ Multimodal Fusion Option ]
 • Spatial attention heatmaps          • Merges visual embeddings with
 • Feature sanity checks                 6 pollutant readings (PM2.5, PM10, etc.)
 • Region of interest audit            • Dense classification head (99.43% Acc)
```

### 1. Custom Baseline CNN
A foundational multi-layer convolutional network built from scratch (alternating Conv2D, Batch Normalization, ReLU activation, MaxPooling, and Dropout layers) to determine raw convolutional performance without prior visual weights.

### 2. ResNet50 Transfer Learning
Leverages residual skip connections pretrained on ImageNet. The convolutional base is frozen while custom dense pooling and classification heads are trained to adapt representations to haze and atmospheric textures.

### 3. EfficientNetB0 (Transfer Learning & Fine-Tuning)
Employs compound scaling balancing network depth, width, and resolution.
- **Stage 1 (Feature Extraction)**: Base layers frozen; top classification head trained with categorical cross-entropy.
- **Stage 2 (Fine-Tuning)**: Unfreezing the top convolutional blocks with a reduced learning rate to adapt high-level feature maps to sky visual dynamics.

### 4. Explainable AI (Grad-CAM)
Gradient-weighted Class Activation Mapping computes the gradient of the target class score with respect to the feature map activations of the final convolutional layer. This produces a coarse localization heatmap highlighting the exact image regions (e.g., upper sky gradient vs. ground horizon) guiding the network's output.

### 5. Multimodal Sensor Fusion
An auxiliary architecture combining image embeddings with six localized environmental pollutant readings:
- **Particulate Matter**: $PM_{2.5}$, $PM_{10}$
- **Gaseous Pollutants**: Ozone ($O_3$), Carbon Monoxide ($CO$), Sulfur Dioxide ($SO_2$), Nitrogen Dioxide ($NO_2$)
- *Note*: Target AQI is excluded from the input feature vector to prevent direct identity leakage.

---

## 📊 Experimental Results & Benchmark Analysis

### Image-Only Model Comparison (Test Set: $N = 1,224$)

| Model Architecture | Training Strategy | Test Accuracy | Macro F1-Score | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **Custom CNN Baseline** | Trained from Scratch | 78.10% | — | — |
| **EfficientNetB0** | Initial Transfer Learning (Frozen Base) | 90.77% | — | — |
| **ResNet50** | Transfer Learning | 92.32% | 92.49% | 92.30% |
| **EfficientNetB0 (Champion)** | **Transfer Learning + Fine-Tuning** | **94.69%** | **95.07%** | **94.69%** |

The fine-tuned **EfficientNetB0** model outperformed all other image-only architectures across all metrics, proving that compound feature representations adapt exceptionally well to subtle variations in sky color degradation and atmospheric scattering.

---

### Multimodal Sensor-Assisted Experiment

| Evaluation Metric | Reported Multimodal Value |
| :--- | :--- |
| **Test Accuracy** | **99.43%** |
| **Macro F1-Score** | **99.36%** |
| **Weighted F1-Score** | **99.43%** |
| **Test Subset Size** | 1,061 paired records |
| **Environmental Features** | 6 Pollutants ($PM_{2.5}, PM_{10}, O_3, CO, SO_2, NO_2$) |

> **Critical Interpretation Note**: The multimodal experiment is conducted on a filtered subset of 1,061 paired image-sensor records using an independent split. Because chemical pollutant concentrations are intrinsic determinants of regulatory AQI definitions, sensor inputs make category separation significantly easier. This result should not be compared directly against image-only benchmarks or interpreted as a pure vision performance metric.

---

## 📈 Visualizations & Artifacts

All visual outputs and diagnostic evaluation charts are archived in the `outputs/` directory:

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Model Comparison** | `outputs/model_comparison.png` | Bar chart benchmarking test accuracies across Custom CNN, ResNet50, and EfficientNetB0. |
| **Confusion Matrix** | `outputs/confusion_matrix.png` | Normalized 6-class confusion matrix for the fine-tuned EfficientNetB0 model. |
| **Grad-CAM Explanations** | `outputs/gradcam_examples.png` | Superimposed heatmaps highlighting atmospheric and horizon features across classes. |

*(Ensure generated PNG files are placed in `outputs/` with these exact filenames for repository rendering).*

---

## 📁 Repository Directory Structure

```text
Air-Pollution-Classification-Deep-Learning/
│
├── README.md                               # Comprehensive project documentation
├── requirements.txt                        # Python library dependencies
├── .gitignore                              # Git exclusion rules (data, checkpoints, caches)
│
├── data/
│   └── README.md                           # Dataset download links & directory setup guide
│
├── notebooks/
│   └── Air_Pollution_Classification_Final.ipynb  # Full end-to-end experimental notebook
│
├── outputs/
│   ├── confusion_matrix.png                # Final model confusion matrix
│   ├── model_comparison.png               # Performance comparison visualization
│   └── gradcam_examples.png                # Grad-CAM interpretability visualizations
│
└── reports/
    └── research_paper.pdf                  # Academic paper / final project report
```

---

## 🚀 Getting Started & Execution Guide

### Prerequisites
- Python 3.8, 3.9, or 3.10
- GPU acceleration recommended (NVIDIA CUDA-compatible GPU, Google Colab, or Kaggle Notebooks)
- Git installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/Air-Pollution-Classification-Deep-Learning.git
cd Air-Pollution-Classification-Deep-Learning
```

### 2. Set Up Virtual Environment
On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (Command Prompt / PowerShell):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dataset Configuration
1. Download the **Air Pollution Image Dataset from India and Nepal (APIN)** from Kaggle.
2. Extract the dataset into the local directory structure specified in `data/README.md`.
3. Verify that images are mapped to their respective class folders (`Good`, `Moderate`, etc.).

### 5. Run the Project
Launch Jupyter Notebook to inspect and run the end-to-end pipeline:
```bash
jupyter notebook notebooks/Air_Pollution_Classification_Final.ipynb
```

---

## ⚠️ Limitations & Research Scope

1. **Environmental & Optical Variability**: Photographic features vary significantly under differing solar angles, cloud formations, seasonal fog, shadows, and white balance settings.
2. **Geographical Generalization**: Models trained on South Asian urban sky profiles require regional fine-tuning before deployment in dissimilar microclimates.
3. **Interpretability Constraints**: Grad-CAM heatmaps offer qualitative visual correlation and feature localization; they do not establish causal atmospheric reasoning.
4. **Sensor Collinearity**: In multimodal settings, strong correlation between chemical pollutants and AQI formulaic definitions can introduce dominant signal bias.
5. **Operational Suitability**: This model functions as a complementary situational awareness tool, not a certified environmental safety sensor.

---

## 👥 Research Team & Academic Context

### Project Investigators
- **Shaswat Singh** — Enrollment No: `S25MCAG0105`
- **Deepak Kumar** — Enrollment No: `S25MCAG0088`

### Academic Supervision & Affiliation
- **Faculty Guide**: Dr. Anurag Shrivastava
- **Institution**: Bennett University, Greater Noida, Uttar Pradesh, India
- **Program**: Master of Computer Applications (MCA), 2nd Year — 3rd Semester
- **Course**: Deep Learning & Computer Vision (`2025CSET703`)

---

## 📄 License & Acknowledgments

- **License**: This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
- **Acknowledgements**: We express our gratitude to Bennett University and the Department of Computer Science & Engineering for institutional support and computing resources, as well as the curators of the APIN dataset on Kaggle for facilitating open environmental AI research.