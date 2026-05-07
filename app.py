import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Anime Detector AI", layout="centered", page_icon="🎨")

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model():
    # Pastikan file 'model_anime_v2.h5' ada di folder yang sama
return tf.keras.models.load_model('model_anime_v2_asli.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model tidak ditemukan! Pastikan 'model_anime_v2.h5' tersedia.")
    st.stop()

# --- 3. TAMPILAN ANTARMUKA ---
st.title("🎨 Anime Classifier : Manusia vs AI")
st.markdown("Sistem ini mendeteksi keaslian seni anime menggunakan arsitektur **MobileNetV2**.")
st.divider()

# Opsi Input: Galeri atau Kamera
source_option = st.radio("Pilih Sumber Gambar:", ("Galeri (Upload)", "Kamera (Live)"))

input_image = None
if source_option == "Galeri (Upload)":
    input_image = st.file_uploader("Pilih gambar dari galeri...", type=["jpg", "jpeg", "png"])
else:
    input_image = st.camera_input("Ambil foto gambar anime")

# --- 4. PROSES KLASIFIKASI & ELIMINASI ---
if input_image is not None:
    # 1. Tampilkan Gambar
    image = Image.open(input_image)
    st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
    
    with st.spinner('Menganalisis pola visual...'):
        # 2. Preprocessing
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Inferensi (Prediksi)
        prediction = model.predict(img_array)
        score = float(prediction[0][0]) # Paksa ke float standar Python

        # 4. Logika Eliminasi (Confidence Check)
        # Menghitung seberapa jauh skor dari titik ragu (0.5)
        # Nilai 'certainty' akan berada di rentang 0.0 (sangat ragu) sampai 1.0 (sangat yakin)
        certainty = abs(score - 0.5) * 2

        st.divider()

        # THRESHOLD ELIMINASI: 
        # Jika certainty < 0.3, sistem menganggap gambar bukan anime atau tidak valid.
        if certainty < 0.3:
            st.warning("⚠️ **Gambar Tidak Dikenali/Bukan Anime**")
            st.write("Sistem mendeteksi bahwa input ini mungkin bukan karakter anime atau gambar terlalu ambigu bagi model.")
            st.info(f"Tingkat Keyakinan Model: {certainty * 100:.2f}% (Terlalu Rendah)")
        
        else:
            # Jika lolos eliminasi, tampilkan hasil klasifikasi
            if score > 0.5:
                st.success(f"### HASIL: KARYA MANUSIA")
                st.progress(score)
                st.info(f"Tingkat Keyakinan: {score * 100:.2f}%")
            else:
                st.error(f"### HASIL: TERDETEKSI AI")
                val_ai = 1.0 - score
                st.progress(val_ai)
                st.info(f"Tingkat Keyakinan: {val_ai * 100:.2f}%")

st.divider()
st.caption("Sistem ini menggunakan MobileNetV2 untuk efisiensi perangkat mobile.")
