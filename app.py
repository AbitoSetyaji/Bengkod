import streamlit as st
import joblib
import pandas as pd
import sklearn

# ===============================
# CONFIG & LOAD MODEL
# ===============================
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

@st.cache_resource
def load_model():
    try:
        # Menambahkan pengecekan versi di log
        return joblib.load("model_churn.pkl")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# ===============================
# UI STREAMLIT
# ===============================
st.title("📊 Prediksi Churn Pelanggan Telco")
st.write(f"Sistem menggunakan Scikit-Learn versi: {sklearn.__version__}")

with st.expander("ℹ️ Informasi Aplikasi"):
    st.write("Aplikasi ini memprediksi apakah seorang pelanggan berpotensi Churn atau Tetap.")

# Input Form
st.header("Masukkan Data Pelanggan")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)

with col2:
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# ===============================
# PROSES PREDIKSI
# ===============================
if st.button("🔍 Jalankan Prediksi"):
    if model is not None:
        # Membuat DataFrame input (Pastikan nama kolom sama dengan saat training)
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'PaperlessBilling': [paperless],
            'Contract': [contract],
            'InternetService': [internet]
        })

        try:
            prediction = model.predict(input_data)[0]
            
            st.divider()
            if prediction == 1 or prediction == "Yes":
                st.error("### ⚠️ HASIL: Pelanggan Berpotensi CHURN")
                st.write("Saran: Segera berikan penawaran khusus agar pelanggan tidak berhenti.")
            else:
                st.success("### ✅ HASIL: Pelanggan Diprediksi TETAP Berlangganan")
                st.write("Saran: Pertahankan layanan prima untuk pelanggan ini.")
        except Exception as e:
            st.warning(f"Terjadi kesalahan saat pemrosesan data: {e}")
            st.info("Pastikan model yang diupload mencakup semua langkah preprocessing (Scaling & Encoding).")
    else:
        st.error("Model tidak tersedia di server. Cek file model_churn.pkl di GitHub.")
