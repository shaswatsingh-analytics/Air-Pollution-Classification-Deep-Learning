# 🌫️ Deep Learning-Based Classification of Air Pollution Levels from Sky Images

> A deep learning project that looks at a sky image and tells you how polluted the air is, across six AQI levels. 🌤️➡️😷

---

## 🛠️ Tools & Technologies

| Tool | How we used it |
|------|----------------|
| 🐍 **Python** | Built the full workflow: data loading, preprocessing, training, and evaluation. |
| 🧠 **TensorFlow / Keras** | Built, trained, and evaluated the custom CNN and the transfer-learning models (EfficientNetB0, ResNet50). |
| 📓 **Jupyter Notebook** | Used for step-by-step experiments, training, evaluation, and result plots. |
| 🔢 **NumPy & Pandas** | Used for numerical work and for handling image paths, labels, and the AQI metadata table. |
| 📊 **Matplotlib & Seaborn** | Used for class distribution charts, training curves, confusion matrices, and model comparison plots. |
| 🧪 **scikit-learn** | Used for train/validation/test splits, class weights, feature scaling, and classification reports. |
| 👁️ **OpenCV** | Used to create and overlay Grad-CAM heatmaps on images. |
| 🐙 **GitHub** | Used to version, document, and share the project code and results. |

---

## 📌 Overview

Air pollution is a serious health problem, but air quality sensors are expensive and not available everywhere. A camera, on the other hand, is cheap and almost everywhere.

This project asks a simple question: **can a deep learning model estimate air pollution levels just by looking at the sky?** 🤔

We trained and compared three image models, then built one extra model that also uses pollutant readings. We also used **Grad-CAM** to see which parts of an image the model focuses on when it makes a decision.

---

## ❗ Problem Statement

Given a sky or urban image, classify the air quality into one of **six AQI classes**:

1. 🟢 Good
2. 🟡 Moderate
3. 🟠 Unhealthy for Sensitive Groups
4. 🔴 Unhealthy
5. 🟣 Very Unhealthy
6. 🟤 Severe

---

## 🗂️ Dataset

- **Name:** Air Pollution Image Dataset (APIN), India and Nepal subset (`IND_and_NEP`)
- **Size:** 12,240 images
- **Classes:** 6 (listed above)
- **Image size used for training:** 224 × 224
- **Metadata file:** `IND_and_Nep_AQI_Dataset.csv`, with location, date and time, AQI, PM2.5, PM10, O3, CO, SO2, NO2, and AQI class for every image

| Class | Images |
|-------|-------:|
| Good | 1,541 |
| Moderate | 1,573 |
| Unhealthy for Sensitive Groups | 2,863 |
| Unhealthy | 2,622 |
| Very Unhealthy | 2,194 |
| Severe | 1,447 |

**Data split (image-only models):** 80% train (9,792) · 10% validation (1,224) · 10% test (1,224), stratified so every class is represented fairly. ⚖️

The classes are not perfectly balanced, so we used **balanced class weights** during training.

---

## 🧩 Methods

1. **🧱 Custom CNN (baseline):** 4 convolution blocks with Batch Normalization, Max Pooling, and Dropout, followed by a dense classification head.
2. **⚡ EfficientNetB0 (transfer learning):** Pretrained on ImageNet. First we trained only the new classification head (base model frozen).
3. **🎯 Fine-tuning:** We then unfroze the **last 30 layers** of EfficientNetB0 and trained them with a very small learning rate (`1e-5`).
4. **🏗️ ResNet50 (transfer learning):** Trained with a frozen base as a second pretrained model for comparison.
5. **🔥 Grad-CAM:** Heatmaps that show which parts of the image influenced the prediction.
6. **🔗 Multimodal EfficientNetB0 (extension):** One branch reads the image, another branch reads six pollutant values (PM2.5, PM10, O3, CO, SO2, NO2). Both are joined to make the final prediction. AQI itself was **not** used as an input, to avoid direct target leakage.

**Training setup:** Adam optimizer, data augmentation (rotation, shift, zoom, shear, flip, brightness), Early Stopping, Reduce LR on Plateau, and Model Checkpoint.

---

## 🏆 Key Results

### 🖼️ Image-only models (test set: 1,224 images)

| Model | Test Accuracy | Macro F1 |
|-------|:-------------:|:--------:|
| Custom CNN | 78.10% | 79.50% |
| EfficientNetB0 (frozen base) | 90.77% | – |
| ResNet50 | 92.32% | 92.49% |
| **EfficientNetB0 (fine-tuned)** ⭐ | **94.69%** | **95.07%** |

✅ **Best image-only model: fine-tuned EfficientNetB0** with 94.69% test accuracy and a test loss of 0.1581.

### 🔗 Multimodal model (image + pollutant readings)

| Model | Test Accuracy | Macro F1 | Weighted F1 | Test Samples |
|-------|:-------------:|:--------:|:-----------:|:------------:|
| Multimodal EfficientNetB0 | 99.43% | 99.36% | 99.43% | 1,061 |

> ⚠️ **Please read before comparing the numbers:** The multimodal model uses a smaller dataset. Records with missing pollutant readings were removed, which left **10,610 of 12,240** records (split 8,488 / 1,061 / 1,061). The test sets are different, and the multimodal model also gets pollutant readings that are closely related to AQI. So its score **should not be directly compared** with the image-only models. We report the two sets of results separately on purpose.

---

## 🖼️ Model Outputs

Export your plots from the notebook into an `images/` folder so they show up here.

| Output | File |
|--------|------|
| 📈 Class distribution | `images/class_distribution.png` |
| 🔲 Confusion matrix (fine-tuned EfficientNetB0) | `images/confusion_matrix_efficientnet.png` |
| 📊 Model comparison chart | `images/model_comparison.png` |
| 🔥 Grad-CAM examples | `images/gradcam_examples.png` |

Example of how to show an image in this file:

```markdown
![Confusion Matrix](images/confusion_matrix_efficientnet.png)
```

**Saved model files** (created by the notebook):
- `Air_Pollution_EfficientNetB0_Final.keras`
- `ResNet50_AirPollution_Best.keras`
- `Air_Pollution_Multimodal_EfficientNetB0_Final.keras`
- `Air_Pollution_Environmental_Feature_Scaler.pkl`
- `Air_Pollution_AQI_Class_LabelEncoder.pkl`
- `Air_Pollution_Multimodal_Config.json`

---

## 🚀 How to Run

**1️⃣ Clone the repository**
```bash
git clone <your-repository-link>
cd <your-repository-folder>
```

**2️⃣ Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

**3️⃣ Install the libraries**
```bash
pip install -r requirements.txt
```

**4️⃣ Add the dataset**
- Place the dataset zip file in the project folder and name it `capstone project archive.zip`.
- The notebook will extract it into a folder called `air_pollution_dataset` on the first run.

**5️⃣ Open and run the notebook**
```bash
jupyter notebook Air_Pollution_Classification_Final.ipynb
```
Run the cells from top to bottom. ▶️

> 💡 **Tip:** Training on a CPU is slow. In our runs, one epoch took roughly 4 to 9 minutes. A GPU (or Google Colab) will make it much faster.

---

## 🔭 Limitations & Future Work

**Limitations**
- 📍 The images are split randomly, so photos from the same place and time can appear in both training and test sets. The model has not been tested on **completely unseen locations**.
- 🌦️ The model has not been tested on very different weather, such as heavy fog, rain, or unusual lighting. These can look like pollution.
- 🌇 Only the India and Nepal subset was used, so results may not carry over to other regions or camera types.
- 🔗 The multimodal model needs pollutant readings at prediction time, which means it needs sensors again.
- 🧱 Neighbouring classes (for example, Moderate vs. Unhealthy) are still confused in some cases, as one Grad-CAM example showed.


---

## 👥 Authors & Acknowledgements

**MCA Capstone Project-I**

| Name | Enrollment No. |
|------|----------------|
| Shaswat Singh | S25MCAG0105 |
| Deepak Kumar | S25MCAG0088 |

 **Mentor:** Dr. Anurag Shrivastava, thank you for your guidance and support throughout this project.

📚 We also thank the creators of the **APIN air pollution image dataset** for making their data available.

---

⭐ If you found this project useful, please consider giving it a star on GitHub!
