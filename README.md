<div align="center">

# Deep Learning-Based Classification of Air Pollution Levels from Sky Images

**An MCA capstone research project that classifies air-pollution levels from sky images using CNN and transfer-learning models, with Grad-CAM interpretability.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/doc/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/docs/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)](https://matplotlib.org/stable/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-008080?style=for-the-badge)](https://www.tensorflow.org/tutorials/images)

[![Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/shaswatsingh-analytics/Air-Pollution-Classification-Deep-Learning)
[![Notebook](https://img.shields.io/badge/Open-Project_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](./Air_Pollution_Classification_Final.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal)
[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0A66C2?style=flat-square)](https://www.linkedin.com/in/shaswatsinghda27)

</div>

---

## Quick Navigation

[Highlights](#highlights) · [Overview](#overview) · [Dataset](#dataset) · [Tools](#tools-and-technologies) · [Methodology](#methodology) · [Results](#results-and-performance) · [How to Run](#how-to-run) · [Milestone 1](#milestone-1--current-progress) · [Limitations](#limitations) · [Team](#team-and-academic-information)

---

## Highlights

| | |
|---|---|
| **Task** | Six-class classification of air-pollution level from sky images |
| **Best image-only model** | Fine-tuned EfficientNetB0, **94.69%** test accuracy |
| **Macro / Weighted F1** | 95.07% / 94.69% |
| **Models compared** | Custom CNN, EfficientNetB0, ResNet50 |
| **Interpretability** | Grad-CAM visualisation of influential image regions |
| **Extra experiment** | Separate multimodal run (images + pollutant features), see [caveats](#separate-multimodal-experiment) |
| **Status** | Academic research prototype, Milestone 1 |

---

## Overview

Air pollution is a major environmental concern, and the visible appearance of the sky changes with atmospheric conditions. This project investigates whether deep learning and computer vision models can classify sky images into predefined air-pollution categories.

Using the **Air Pollution Image Dataset from India and Nepal (APIN)**, the project compares a custom Convolutional Neural Network (CNN) with transfer-learning models (EfficientNetB0 and ResNet50). Grad-CAM is used to visualise the image regions that influence predictions.

> **Note:** This is an academic research prototype. It is **not** an official air-quality monitoring system and does not measure PM2.5 or produce a validated AQI reading.

### Problem Statement

Visual patterns in sky images may correlate with pollution levels, but image appearance is also affected by lighting, weather, location and camera settings. This project explores how well deep learning can learn pollution-level categories from such images, and how far the results can be trusted.

### Objectives

- Classify sky images into six dataset-defined pollution categories.
- Train and compare a custom CNN, EfficientNetB0 and ResNet50.
- Evaluate with test accuracy and F1-scores.
- Use Grad-CAM to inspect image regions linked to predictions.
- Explore multimodal learning (images + environmental features) as a separate experiment.
- Document results, limitations and next steps transparently.

---

## Dataset

**Air Pollution Image Dataset from India and Nepal (APIN)**

| Detail | Description |
|---|---|
| Total images | 12,240 |
| Training | 9,792 |
| Validation | 1,224 |
| Test | 1,224 |
| Classes | 6 |

**Classes:** Good · Moderate · Unhealthy for Sensitive Groups · Unhealthy · Very Unhealthy · Severe

**Links:** [Kaggle dataset](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal) · [Source repository](https://github.com/ICCC-Platform/Air-Pollution-Image-Dataset-From-India-and-Nepal) · [DOI](https://doi.org/10.34740/KAGGLE/DS/3152196)

**Citation:** Rouniyar et al. (2023). *Air Pollution Image Dataset from India and Nepal* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3152196

The dataset is an external research resource and is not claimed as original work of this team.

---

## Tools and Technologies

| Technology | Purpose in this project | Docs |
|---|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core language for data preparation, training and evaluation | [Python docs](https://docs.python.org/3/) |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | Deep-learning framework | [TensorFlow docs](https://www.tensorflow.org/api_docs) |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) | Building and training neural networks | [Keras docs](https://keras.io/api/) |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) | Notebook-based experimentation | [Jupyter docs](https://docs.jupyter.org/) |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical operations and arrays | [NumPy docs](https://numpy.org/doc/) |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Tabular data for the multimodal experiment | [Pandas docs](https://pandas.pydata.org/docs/) |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Plots and result visualisation | [Matplotlib docs](https://matplotlib.org/stable/) |

---

## Methodology

1. **Data loading and exploration:** load the APIN dataset and inspect the six classes.
2. **Preprocessing and splitting:** prepare images and use the 9,792 / 1,224 / 1,224 train / validation / test split.
3. **Model training:** train a custom CNN and transfer-learning models.
4. **Evaluation:** measure test accuracy and F1-scores.
5. **Model comparison:** compare all image-only models on the same test split.
6. **Interpretability:** apply Grad-CAM to selected predictions.
7. **Multimodal experiment:** separately combine images with environmental features.

Exact preprocessing settings and hyperparameters are in the [project notebook](./Air_Pollution_Classification_Final.ipynb).

### Models Implemented

| Model | Role |
|---|---|
| Custom CNN | Baseline convolutional model for six-class classification |
| EfficientNetB0 | Transfer-learning model; fine-tuned version gave the best image-only result |
| ResNet50 | Residual network evaluated for comparison |
| Grad-CAM | Visual explanation of image regions behind predictions |

---

## Results and Performance

### Image-only classification

| Model | Test Accuracy |
|---|---|
| Custom CNN | 78.10% |
| **EfficientNetB0 (fine-tuned)** | **94.69%** |
| ResNet50 | 92.32% |

**Best model: fine-tuned EfficientNetB0**

| Metric | Score |
|---|---|
| Test accuracy | 94.69% |
| Macro F1-score | 95.07% |
| Weighted F1-score | 94.69% |

These numbers describe performance on the reported test split only. They do not establish performance on new locations, cameras, seasons or weather conditions.


### Separate multimodal experiment

A separate experiment combined sky images with environmental features (PM2.5, PM10, O3, CO, SO2, NO2).

| Metric | Reported result |
|---|---|
| Test accuracy | 99.43% |
| Macro F1-score | 99.36% |
| Weighted F1-score | 99.43% |
| Test samples | 1,061 |

> **Important:** AQI was excluded as an input to avoid direct target leakage. However, pollutant measurements are closely related to AQI and may still carry target-related information. This result may be optimistic and is **not directly comparable** with the image-only results. It needs further leakage analysis and independent validation.

---

## Repository Structure

```text
Air-Pollution-Classification-Deep-Learning/
├── README.md
├── Air_Pollution_Classification_Final.ipynb
└── outputs/
    └── model_comparison.png
```

---

## How to Run

### Option 1: Google Colab

1. Open [the notebook](./Air_Pollution_Classification_Final.ipynb) on GitHub and download it, or upload it to Google Colab.
2. Make the [dataset](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal) available in the runtime and update the dataset paths if needed.
3. Run the cells in order.

### Option 2: Run locally

```bash
git clone https://github.com/shaswatsingh-analytics/Air-Pollution-Classification-Deep-Learning.git
cd Air-Pollution-Classification-Deep-Learning

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install jupyter tensorflow numpy pandas matplotlib

jupyter notebook
```

Open `Air_Pollution_Classification_Final.ipynb`, download the dataset from Kaggle and set the dataset path before running the cells.

---

## Milestone 1 — Current Progress

**Completed**

- [x] Prepared image-only train / validation / test splits
- [x] Trained and compared Custom CNN, EfficientNetB0 and ResNet50
- [x] Reached 94.69% test accuracy with fine-tuned EfficientNetB0
- [x] Reported macro and weighted F1-scores for the best model
- [x] Implemented Grad-CAM visualisation
- [x] Ran a separate multimodal experiment

**Planned**

- [ ] Validate on independent images from different locations
- [ ] Test robustness to lighting, weather and image quality
- [ ] Analyse possible leakage in the multimodal experiment
- [ ] Expand per-class evaluation and interpretability analysis
- [ ] Consider a demo or deployment only after further validation

---

## Limitations

- **Not a pollutant measurement:** the image-only model predicts dataset labels; it does not measure PM2.5, PM10 or official AQI.
- **Generalisation:** results on this test split may not carry over to new locations, cameras or seasons.
- **Image conditions:** lighting, cloud cover, visibility and camera quality affect visual features.
- **Multimodal results:** pollutant features are related to AQI, so the high score needs leakage checks.
- **Research prototype:** not validated as a replacement for official air-quality monitoring.

---

## Team and Academic Information

| Detail | Information |
|---|---|
| Project | Deep Learning-Based Classification of Air Pollution Levels from Sky Images |
| University | Bennett University, Greater Noida |
| Course code | 2025CSET703 |
| Mentor | Dr. Anurag Shrivastava |
| Team member | Shaswat Singh (S25MCAG0105) - [LinkedIn](https://www.linkedin.com/in/shaswatsinghda27) |
| Team member | Deepak Kumar (S25MCAG0088) |

---

## References

- Rouniyar et al. (2023). *Air Pollution Image Dataset from India and Nepal* [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DS/3152196](https://doi.org/10.34740/KAGGLE/DS/3152196)
- [APIN dataset on Kaggle](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal) · [Dataset source repository](https://github.com/ICCC-Platform/Air-Pollution-Image-Dataset-From-India-and-Nepal)
- [TensorFlow](https://www.tensorflow.org/) · [Keras](https://keras.io/) · [Python](https://docs.python.org/3/)

---

<div align="center">

**MCA Capstone Project · Milestone 1 · Bennett University**

[Repository](https://github.com/shaswatsingh-analytics/Air-Pollution-Classification-Deep-Learning) · [Notebook](./Air_Pollution_Classification_Final.ipynb) · [Dataset](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal) · [LinkedIn](https://www.linkedin.com/in/shaswatsinghda27)

</div>
