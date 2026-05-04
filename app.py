import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Anime Detector AI", layout="centered")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model():
    # Memuat model MobileNetV2 yang sudah dibuat[cite: 4, 5]
    return tf.keras.models.load_model('model_anime_v2.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model tidak ditemukan! Pastikan 'model_anime_v2.h5' ada di folder yang sama.")
    st.stop()

# --- TAMPILAN ANTARMUKA ---
st.title("🎨 Anime Classifier: Manusia vs AI")
st.markdown("Deteksi keaslian gambar anime menggunakan arsitektur **MobileNetV2**.")
st.divider()

# Opsi Input: Galeri atau Kamera[cite: 5]
source_option = st.radio("Pilih Sumber Gambar:", ("Galeri (Upload)", "Kamera (Live)"))

input_image = None
if source_option == "Galeri (Upload)":
    input_image = st.file_uploader("Pilih gambar dari galeri Anda...", type=["jpg", "jpeg", "png"])
else:
    input_image = st.camera_input("Ambil foto gambar anime")

# --- PROSES KLASIFIKASI ---
if input_image is not None:
    # 1. Tampilkan Gambar
    image = Image.open(input_image)
    st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
    
    with st.spinner('Menganalisis pola gambar...'):
        # 2. Preprocessing (Resize ke $224 \times 224$)[cite: 5]
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Inferensi (Prediksi)[cite: 4, 5]
        prediction = model.predict(img_array)
        score = prediction[0][0]

        # 4. Tampilkan Hasil (Threshold 0.5)
        st.divider()
        if score > 0.5:
            st.success(f"### HASIL: **KARYA MANUSIA**")
            st.info(f"Tingkat Keyakinan: {score * 100:.2f}%")
        else:
            st.error(f"### HASIL: **TERDETEKSI AI**")
            st.info(f"Tingkat Keyakinan: {(1 - score) * 100:.2f}%")

st.divider()
st.caption("Aplikasi ini menggunakan MobileNetV2 untuk efisiensi perangkat mobile.")
