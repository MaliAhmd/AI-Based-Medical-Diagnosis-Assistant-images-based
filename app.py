import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

model = tf.keras.models.load_model("models/pneumonia_model.h5")

st.title("AI Medical Diagnosis Assistant")
st.write("Upload Chest X-ray Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

# ✅ FIXED PREPROCESS FUNCTION
def preprocess(image):
    image = image.convert("RGB")              # 🔥 VERY IMPORTANT (adds 3 channels)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    image = np.array(image) / 255.0           # normalize
    
    image = np.expand_dims(image, axis=0)     # shape → (1, 224, 224, 3)
    
    return image

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)

    # 🔍 DEBUG (optional - remove later)
    print("Shape:", img.shape)

    prediction = model.predict(img)

    # ✅ FIX: access correct value
    pred_value = prediction[0][0]

    if pred_value > 0.5:
        st.error("Prediction: Pneumonia")
        st.write("Confidence:", float(pred_value))
    else:
        st.success("Prediction: Normal")
        st.write("Confidence:", float(1 - pred_value))