🌫️ Deep Learning-Based Classification of Air Pollution Levels from Sky Images

Python
TensorFlow
Keras
Computer Vision
Jupyter Notebook
NumPy
Pandas
Matplotlib
Grad-CAM


A deep learning and computer vision project that classifies air pollution levels from sky and urban images into six categories using CNNs, transfer learning, and explainable AI.


Kaggle Dataset
Project Notebook
LinkedIn
GitHub Repository



📑 Quick Navigation


Overview

Problem Statement

Dataset

Tools and Technologies

Methods and Models

Results

Model Outputs

How to Run

Repository Structure

Limitations

Future Work

Team and Acknowledgements


📌 Overview

Air pollution is a major environmental and public-health concern. This project investigates whether visual features in outdoor sky and urban images can help classify air pollution levels using deep learning.


The project uses the Air Pollution Image Dataset from India and Nepal (APIN) and compares a custom Convolutional Neural Network (CNN) with pretrained architectures. It also explores a multimodal approach that combines image features with environmental pollutant measurements.


This is a research prototype intended to support exploration and awareness. It is not a replacement for official air-quality monitoring stations or certified AQI measurements.


🎯 Problem Statement

Air-quality assessment commonly relies on measurements collected by sensors and monitoring stations. This project explores image-based classification as a complementary research approach by training deep learning models to assign sky and urban images to one of six pollution categories.


The project evaluates model performance using accuracy, precision, recall, F1-score, and confusion matrices, and uses Grad-CAM to inspect image regions influencing model predictions.


📂 Dataset

Dataset: Air Pollution Image Dataset from India and Nepal (APIN)



Total images: 12,240

Image dimensions: 224 × 224 pixels

Image type: RGB

Number of classes: 6

Image-only training split: 9,792 images

Image-only validation split: 1,224 images

Image-only test split: 1,224 images


Classification Categories


Good

Moderate

Unhealthy for Sensitive Groups

Unhealthy

Very Unhealthy

Severe


The image-only experiment uses an 80:10:10 train-validation-test split. Dataset source: Air Pollution Image Dataset from India and Nepal — Kaggle


The dataset files themselves are not included in this repository.


🛠️ Tools and Technologies

Tool / Technology	Description
Python	Main programming language used for model development, data processing, and evaluation.
TensorFlow	Deep learning framework used to train and evaluate neural networks.
Keras	High-level API used to build CNN models and implement transfer learning.
NumPy	Numerical operations and array processing.
Pandas	Processing tabular environmental metadata and preparing multimodal inputs.
Matplotlib	Plotting and visualising experimental results.
Jupyter Notebook	Interactive environment for experiments, code, evaluation, and documentation.
EfficientNetB0	Pretrained image-classification architecture used for transfer learning and fine-tuning.
ResNet50	Pretrained architecture used for comparison with other image-classification models.
Grad-CAM	Visual explanation technique used to highlight image regions influencing a model prediction.
GitHub	Platform for sharing source code, documentation, and project outputs.


The final requirements.txt should be generated from the libraries actually used in the notebook. Do not add dependencies solely because they are listed in this table.



🧠 Methods and Models

1. Custom CNN

A convolutional neural network trained from scratch to establish a baseline for six-class image classification.


2. EfficientNetB0 Transfer Learning

An ImageNet-pretrained EfficientNetB0 architecture adapted to the project's six pollution categories.


3. EfficientNetB0 Fine-Tuning

The pretrained model was fine-tuned for the target classification task. The final fine-tuned model produced the strongest reported result among the image-only experiments.


4. ResNet50 Transfer Learning

A pretrained ResNet50 architecture was evaluated as a comparison model.


5. Grad-CAM Explainability

Grad-CAM heatmaps were generated to inspect which image regions contributed to selected model predictions. These heatmaps provide qualitative explanations; they do not prove causal reasoning.


6. Multimodal Image + Environmental Features

An additional experiment combined image inputs with six environmental measurements:



PM2.5

PM10

O3

CO

SO2

NO2


AQI itself was excluded as a direct model input. However, pollutant measurements are closely related to AQI category definitions, so high sensor-assisted performance should be interpreted carefully and independently validated.


📊 Results

Image-Only Model Comparison

Model	Reported Test Accuracy	Macro F1-score	Weighted F1-score
Custom CNN	78.10%	—	—
EfficientNetB0 — Initial Transfer Learning	90.77%	—	—
ResNet50	92.32%	92.49%	92.30%
EfficientNetB0 — Fine-Tuned	94.69%	95.07%	94.69%

The fine-tuned EfficientNetB0 achieved the strongest reported image-only performance, with 94.69% test accuracy.


Multimodal Experiment

The notebook reports the following results for the sensor-assisted multimodal experiment:



Test accuracy: 99.43%

Macro F1-score: 99.36%

Weighted F1-score: 99.43%

Test samples: 1,061

Environmental features: Six pollutant measurements


Important interpretation: The multimodal experiment uses a separate usable-record subset and a separate train-validation-test split from the image-only experiment. Its result must not be presented as image-only accuracy or as a directly controlled comparison with the image-only models. Because pollutant measurements are related to the AQI category target, further independent validation and leakage checks are important before drawing real-world conclusions.


📈 Model Outputs and Visualizations

Image-Only Model Comparison

The chart below is taken from the project notebook and compares the reported test accuracy of the custom CNN, fine-tuned EfficientNetB0, and ResNet50 models.


Image-only model accuracy comparison


Upload this image to the repository at outputs/model_comparison.png for it to render here.


Upload the original output images to the outputs/ folder and keep these filenames consistent for the links below to work:



Image-only confusion matrix

Model performance comparison

Grad-CAM examples


These links are repository-relative and will become clickable once the corresponding files are uploaded. Do not use placeholder or recreated figures as if they were original experimental outputs.


🔗 Quick Access Links


Open the project notebook

Download the dataset on Kaggle

Connect with me on LinkedIn

View this GitHub repository


🚀 How to Run the Project

Prerequisites


Python installed on your computer.

Jupyter Notebook or JupyterLab.

The project notebook.

Access to the dataset files, arranged according to the paths expected by the notebook.


1. Clone the Repository

Replace YOUR-GITHUB-USERNAME with the actual GitHub username after creating the repository.


git clone https://github.com/YOUR-GITHUB-USERNAME/Air-Pollution-Classification-Deep-Learning.git
cd Air-Pollution-Classification-Deep-Learning

2. Create a Virtual Environment

python -m venv .venv

Activate it on Windows:


.venv\\Scripts\\activate

3. Install Dependencies

After the project requirements.txt has been verified and uploaded, run:


pip install -r requirements.txt

4. Launch Jupyter Notebook

jupyter notebook

Open the notebook under notebooks/ and follow its dataset-path and execution instructions.


Compute note: Deep learning training may require substantial computing resources. A compatible GPU or a cloud notebook environment such as Google Colab may be useful.


📁 Repository Structure

The repository is intended to follow this layout. Only files that have actually been uploaded will appear in GitHub.


Air-Pollution-Classification-Deep-Learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── Air_Pollution_Classification_Final.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── gradcam_examples.png
│
└── reports/
    └── research_paper.pdf

The full image dataset is not included. data/README.md should explain where to obtain the dataset and how to arrange it locally.


⚠️ Limitations


Image appearance can change with lighting, weather, camera settings, location, and season.

Generalisation to unseen cities and environmental conditions requires further validation.

Grad-CAM is a qualitative visualisation technique and does not establish causality.

Multimodal pollutant features are closely related to AQI categories and may make classification easier than image-only prediction.

The image-only and multimodal experiments use different evaluation subsets; their scores are not a controlled, apples-to-apples comparison.

The project is a research prototype and should not be used as a certified AQI measurement system.



👥 Team and Acknowledgements

Project Team



Shaswat Singh — S25MCAG0105

Deepak Kumar — S25MCAG0088


Faculty Guide: Dr. Anurag Shrivastava
Institution: Bennett University, Greater Noida, Uttar Pradesh, India
Programme: Master of Computer Applications (MCA), Second Year — Third Semester
Course Code: 2025CSET703



⭐ If you find this project useful for learning about deep learning, computer vision, or environmental applications, feel free to explore the repository.

