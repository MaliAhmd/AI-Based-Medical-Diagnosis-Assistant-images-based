# 🏥 AI-Based Medical Diagnosis Assistant (Image-Based)

This project is a lightweight **AI-powered medical diagnosis assistant** that analyzes **Chest X-ray images** to detect **Pneumonia** using Deep Learning.

It is designed to run on a **CPU-based system** using Transfer Learning (MobileNetV2).

---

## 🚀 Features

- 📷 Upload Chest X-ray image
- 🤖 AI predicts:
  - Normal
  - Pneumonia
- 📊 Displays prediction confidence
- 🌐 Simple web interface using Streamlit
- ⚡ CPU-friendly (no GPU required)

---

## 📂 Project Structure
medical_ai_diagnosis/
│
├── dataset/ # (ignored in git)
├── models/ # (ignored in git)
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│
├── app.py # Streamlit app
├── requirements.txt
├── README.md
└── .gitignore



---

## 📊 Dataset

Dataset used:

Chest X-Ray Images (Pneumonia) from Kaggle

Download using Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

```
dataset/
    train/
    val/
    test/


pip install -r requirements.txt
