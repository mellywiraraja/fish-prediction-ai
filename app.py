import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os
import zipfile

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fish Counter AI", layout="centered")

MODEL_DIR = "model_saved"
ZIP_PATH = "model.zip"

# =========================
# DOWNLOAD & EXTRACT MODEL
# =========================
if not os.path.exists(MODEL_DIR):
    with st.spinner("Mengunduh model..."):
        url = "https://drive.google.com/uc?id=1FKXLHjrdc77PZu63mhFhaIanlj-gcwP0"
        gdown.download(url, ZIP_PATH, quiet=False)

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(ZIP_PATH)

# =========================
# LOAD MODEL (SAVEDMODEL)
# =========================
@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_DIR)
    return model.signatures["serving_default"]

with st.spinner("Memuat model..."):
    model = load_model()

# =========================
# UI
# =========================
st.title("🐟 Fish Counter AI")
st.markdown("Prediksi jumlah benih ikan dari citra secara otomatis")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar", use_container_width=True)

    # =========================
    # PREPROCESSING (WAJIB SAMA DENGAN TRAINING)
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICTION (FIX SAVEDMODEL)
    # =========================
    if st.button("Hitung Jumlah Ikan"):
        with st.spinner("Memproses..."):
            try:
                input_tensor = tf.convert_to_tensor(img_array)

                output = model(input_tensor)

                # ambil output pertama
                prediction = list(output.values())[0].numpy()

                fish_count = int(np.round(prediction[0][0]))
                fish_count = max(0, fish_count)

                st.success(f"Jumlah ikan terdeteksi: {fish_count}")

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
