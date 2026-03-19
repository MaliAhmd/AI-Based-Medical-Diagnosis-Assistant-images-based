# 🩺 AI-Based Medical Diagnosis Assistant (Image-Based)

## 📌 Project Overview

This project is a **Deep Learning-based Medical Diagnosis Assistant** that analyzes **Chest X-ray images** to detect **Pneumonia**.

It helps doctors by:

* Providing quick predictions
* Highlighting disease probability
* Assisting in early diagnosis

> ⚠️ Note: This system is for **educational purposes only** and should not replace professional medical advice.

---

## 🚀 Features

* Image-based disease detection (Chest X-rays)
* Lightweight model (runs on CPU)
* Transfer Learning using MobileNetV2
* Streamlit Web Interface
* Easy to train and deploy

---

## 🧠 Model Details

* Architecture: MobileNetV2 (Pretrained on ImageNet)
* Input Size: 224x224
* Output: Binary Classification (Normal / Pneumonia)

---

## 📂 Project Structure

```
medical_ai_diagnosis/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── pneumonia_model.h5
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset used:
Chest X-Ray Images (Pneumonia) – Kaggle

Download using:

```
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

After downloading, extract and place inside:

```
dataset/
```

---

## ⚙️ Installation

Clone the repository:

```
git clone <your-repo-link>
cd medical_ai_diagnosis
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```
cd src
python train.py
```

* Model will be saved in:

```
models/pneumonia_model.h5
```

---

## 📈 Evaluate Model

```
python evaluate.py
```

---

## 🌐 Run Web App

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

Upload an X-ray image and get prediction.

---

## 🧪 Example Output

```
Prediction: Pneumonia
Confidence: 92%
```

---

## 💻 Requirements

* Python 3.8+
* TensorFlow
* OpenCV
* NumPy
* Streamlit

---

## ⚡ Performance

* Accuracy: ~90–95%
* Works on CPU (no GPU required)

---

## 🔮 Future Improvements

* Grad-CAM (Explainable AI)
* Multi-disease detection
* Medical report generation
* Integration with hospital systems

---

## 👨‍💻 Author

Muhammad Ali Ahmad
BS Software Engineering

---

## 📜 License

This project is for academic and educational use.
