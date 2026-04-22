import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fish Counter AI", layout="centered")

MODEL_PATH = "model_fixed.h5"

# =========================
# DOWNLOAD MODEL DARI DRIVE
# =========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        url = "https://drive.google.com/uc?id=1yg7jK-C_I-HtnoLY9lZ3-BTiJ3NdMI2g"
        gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL (SUDAH NORMAL)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

with st.spinner("Memuat model..."):
    model = load_model()

# =========================
# UI
# =========================
st.title("🐟 Fish Counter AI")
st.markdown("Prediksi jumlah benih ikan dari citra secara otomatis")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar", use_container_width=True)

    # PREPROCESSING
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICTION
    if st.button("Hitung Jumlah Ikan"):
        with st.spinner("Memproses..."):
            prediction = model.predict(img_array)
            fish_count = max(0, int(np.round(prediction[0][0])))

        st.success(f"Jumlah ikan terdeteksi: {fish_count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
