# 🧠 MURA X-Ray Classification using EfficientNet

This project focuses on detecting abnormalities in musculoskeletal X-ray images using deep learning, specifically through the **EfficientNetB0** architecture. It utilizes the **MURA (Musculoskeletal Radiographs)** dataset — one of the largest publicly available medical imaging datasets, released by the **Stanford ML Group**.

The MURA dataset contains **over 40,000 images from nearly 15,000 studies**, across **seven upper extremity body parts**:

- Elbow  
- Finger  
- Forearm  
- Hand  
- Humerus  
- Shoulder  
- Wrist  

Each study is labeled as either **normal** or **abnormal** based on expert radiologist interpretation. This rich dataset offers a valuable foundation for developing machine learning models to support clinical diagnostics.

In this project, I built a classification pipeline using **EfficientNetB0**, enhanced through **transfer learning**. The goal is to enable reliable identification of abnormal X-rays, potentially assisting radiologists in real-world diagnostic tasks.

---

### ✅ Key Project Highlights

- Leveraged **EfficientNetB0** pre-trained on ImageNet for transfer learning  
- Handled class imbalance using **class weight adjustments**
- Ensured **patient-level splitting** for more realistic model evaluation  
- Used **data augmentation** with `ImageDataGenerator`  
- Tracked performance via training history and prediction visualization  

---

## 📌 Overview

- **Task:** Binary classification of X-rays (Normal vs Abnormal)
- **Dataset:** [MURA – Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
- **Model Architecture:** EfficientNetB0
- **Libraries Used:** TensorFlow, Keras, Pandas, Scikit-learn

---

## 🧰 Core Features

- Clean data pipeline with stratified patient-wise splitting
- Transfer learning setup using EfficientNetB0
- Balanced training through `compute_class_weight`
- Real-time data augmentation
- Classification reporting and performance charts

---

## 📊 Model Performance

| Metric      | Model 1 (Baseline) | Model 2 (EfficientNetB0) |
|-------------|--------------------|---------------------------|
| Accuracy    | 59%                | **85%**                   |
| Precision   | 64% (macro avg)    | **85%** (macro avg)       |
| Recall      | 63% (macro avg)    | **84%** (macro avg)       |
| F1-Score    | 59% (macro avg)    | **84%** (macro avg)       |

🧠 The second model—powered by EfficientNetB0—delivered a significant performance jump, reaching **85% accuracy** and showing balanced results across both classes.

---

## 🏁 Future Enhancements

- Test other EfficientNet variants (B1–B7) for potential accuracy gains  
- Integrate **Grad-CAM** for visual model explainability  
- Apply **hyperparameter tuning** to further improve generalization  

---
