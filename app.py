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

MODEL_PATH = "model.h5"

# =========================
# DOWNLOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        url = "https://drive.google.com/uc?id=1Y_K4r-TfEqnCEJ0uhrnTB9n7i299QyGS"
        gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL (ANTI CRASH)
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        return str(e)

model = load_model()

# =========================
# UI
# =========================
st.title("🐟 Fish Counter AI")
st.markdown("Prediksi jumlah benih ikan dari citra secara otomatis")

# =========================
# HANDLE ERROR MODEL
# =========================
if isinstance(model, str):
    st.error("❌ Model gagal dimuat")
    st.code(model)
    st.stop()

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar", use_container_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION
    # =========================
    if st.button("Hitung Jumlah Ikan"):
        with st.spinner("Memproses gambar..."):
            prediction = model.predict(img_array)
            fish_count = max(0, int(np.round(prediction[0][0])))

        st.success(f"Jumlah ikan terdeteksi: {fish_count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
