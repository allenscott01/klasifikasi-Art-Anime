import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Anime Detector AI", layout="centered", page_icon="🎨")

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model():
    # Menggunakan model asli hasil training dataset anime Anda
    return tf.keras.models.load_model('model_anime_v2_asli.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model 'model_anime_v2_asli.h5' tidak ditemukan! Pastikan file tersedia di folder yang sama.")
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
    
    with st.spinner('Menganalisis pola visual dan karakteristik anime...'):
        # 2. Preprocessing
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Inferensi (Prediksi)
        prediction = model.predict(img_array)
        # Ambil skor dan pastikan dalam format float murni
        score = float(np.clip(prediction[0][0], 0.0, 1.0))

        # 4. Logika Eliminasi (OOD - Out of Distribution Detection)
        # Certainty dihitung berdasarkan jarak dari titik ambang ragu (0.5)
        certainty = abs(score - 0.5) * 2

        st.divider()

        # TENTUKAN THRESHOLD KETAT (Misalnya 0.4 atau 40%)
        # Semakin tinggi angka ini, semakin "pilih-pilih" sistem dalam menerima gambar.
        rejection_threshold = 0.4 

        if certainty < rejection_threshold:
            # JIKA TIDAK LOLOS: Gambar ditolak dan tidak diklasifikasi
            st.warning("### ❌ Gambar Ditolak")
            st.write("Sistem tidak mendeteksi karakteristik pola anime yang cukup kuat pada gambar ini.")
            st.info(f"Keyakinan Model: {certainty * 100:.2f}% (Di bawah batas minimal {rejection_threshold * 100:.0f}%)")
            st.info("Saran: Gunakan gambar karakter anime dengan garis/outline yang jelas.")
        
        else:
            # JIKA LOLOS: Lanjutkan ke klasifikasi
            st.subheader("✅ Hasil Analisis:")
            
            if score > 0.5:
                st.success(f"### KATEGORI: KARYA MANUSIA")
                st.progress(score)
                st.write(f"Sistem yakin **{score * 100:.2f}%** bahwa ini adalah guratan tangan seniman.")
            else:
                st.error(f"### KATEGORI: TERDETEKSI AI")
                val_ai = 1.0 - score
                st.progress(val_ai)
                st.write(f"Sistem yakin **{val_ai * 100:.2f}%** bahwa ini adalah hasil generator AI.")

st.divider()
st.caption("Sistem ini menggunakan MobileNetV2 untuk efisiensi perangkat mobile.")
