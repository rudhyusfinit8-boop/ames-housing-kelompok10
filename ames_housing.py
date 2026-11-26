import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CUSTOM CSS (MODIFIKASI TAMPILAN)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ames Housing Analytics | Kelompok 10",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FUNGSI LOAD DATA (DENGAN FALLBACK UPLOAD)
# ---------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("train.csv")
    except FileNotFoundError:
        return None

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGASI & IDENTITAS KELOMPOK
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/id/6/63/Logo_Universitas_Pamulang.png", width=80)
    st.title("Navigasi Aplikasi")
    menu = st.radio(
        "Pilih Halaman:",
        [
            "🏠 Ringkasan & Dataset",
            "📊 Eksplorasi & Visualisasi",
            "🗺️ Peta & Lokasi Ames",
            "🧮 Simulasi Harga Rumah",
            "🛠️ Upload & Edit Data"
        ]
    )
    
    st.divider()
    st.markdown("### 👨‍🎓 Anggota Kelompok 10")
    st.caption("""
    **Universitas Pamulang**  
    * Sigit Budi Prasetyo  
    * Rudianto Usfinit  
    * Mayolus Paru Liu  
    """)
    st.info("Studi Kasus: Ames Housing Dataset (Kaggle)")

df = load_data()

if df is None:
    st.sidebar.warning("⚠️ File `train.csv` belum ditemukan di folder.")
    file_sidebar = st.sidebar.file_uploader("Unggah train.csv di sini", type=["csv"])
    df = load_data(file_sidebar)

# ---------------------------------------------------------
# 4. KONTEN HALAMAN BERDASARKAN MENU
# ---------------------------------------------------------
if df is None:
    st.error("Silakan letakkan file `train.csv` di folder yang sama dengan `app.py` atau unggah melalui menu di sebelah kiri.")
    st.stop()

st.markdown('<div class="main-header">🏡 Analisis Karakteristik & Harga Properti Ames Housing</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Implementasi Framework Streamlit — Praktikum Machine Learning (Bab 1 & Bab 2)</div>', unsafe_allow_html=True)

# =========================================================
# MENU 1: RINGKASAN & DATASET
# =========================================================
if menu == "🏠 Ringkasan & Dataset":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Properti", f"{len(df):,} Baris", f"{len(df.columns)} Fitur")
    col2.metric("Rata-rata Harga Jual", f"${df['SalePrice'].mean():,.2f}")
    col3.metric("Median Harga Jual", f"${df['SalePrice'].median():,.0f}")
    col4.metric("Harga Tertinggi", f"${df['SalePrice'].max():,.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Pratinjau Data Utama", "📈 Statistik Deskriptif", "ℹ️ Metadata JSON"])
    
    kolom_pilihan = ["Id", "SalePrice", "OverallQual", "GrLivArea", "Neighborhood", "YearBuilt", "TotalBsmtSF", "GarageCars", "KitchenQual"]
    kolom_tersedia = [k for k in kolom_pilihan if k in df.columns]

    with tab1:
        st.subheader("Sampel Fitur Utama Dataset")
        jumlah_tampil = st.slider("Jumlah baris yang ditampilkan:", 5, 50, 10)
        st.dataframe(df[kolom_tersedia].head(jumlah_tampil), use_container_width=True)

    with tab2:
        st.subheader("Ringkasan Statistik Fitur Numerik")
        st.dataframe(df[kolom_tersedia].describe().T, use_container_width=True)

    with tab3:
        st.subheader("Informasi Struktur Dataset (JSON)")
        st.json({
            "dataset": "Ames Housing Dataset",
            "sumber": "Kaggle - House Prices: Advanced Regression Techniques",
            "penyusun_asli": "Dean De Cock (2011)",
            "jumlah_baris": int(len(df)),
            "jumlah_kolom": int(len(df.columns)),
            "variabel_target": "SalePrice (USD)"
        })

# =========================================================
# MENU 2: EKSPLORASI & VISUALISASI
# =========================================================
elif menu == "📊 Eksplorasi & Visualisasi":
    st.subheader("Visualisasi Interaktif Karakteristik Rumah")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_price, max_price = int(df["SalePrice"].min()), int(df["SalePrice"].max())
        rentang_harga = st.slider(
            "Filter Rentang Harga Rumah (USD):",
            min_price, max_price, (100000, 350000), step=5000
        )
    with col_f2:
        pilihan_kualitas = st.multiselect(
            "Filter Kualitas Material (OverallQual 1-10):",
            options=sorted(df["OverallQual"].unique()),
            default=sorted(df["OverallQual"].unique())
        )

    df_filtered = df[
        (df["SalePrice"].between(rentang_harga[0], rentang_harga[1])) &
        (df["OverallQual"].isin(pilihan_kualitas))
    ]
    st.caption(f"Menampilkan **{len(df_filtered)}** dari **{len(df)}** data rumah sesuai filter.")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df_filtered, x="SalePrice", nbins=30,
            title="Distribusi Harga Jual Rumah (SalePrice)",
            color_discrete_sequence=["#2563EB"]
        )
        fig_hist.update_layout(xaxis_title="Harga Jual (USD)", yaxis_title="Jumlah Rumah")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("**Hubungan Luas Bangunan (GrLivArea) vs Harga Jual**")
        chart_altair = alt.Chart(df_filtered).mark_circle(size=55, opacity=0.75).encode(
            x=alt.X("GrLivArea", title="Luas Area Tinggal (sq ft)"),
            y=alt.Y("SalePrice", title="Harga Jual (USD)"),
            color=alt.Color("OverallQual:N", title="Kualitas (1-10)"),
            tooltip=["Id", "Neighborhood", "GrLivArea", "OverallQual", "SalePrice"]
        ).interactive()
        st.altair_chart(chart_altair, use_container_width=True)

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Rata-rata Harga Berdasarkan Kualitas Rumah (OverallQual)**")
        rata_harga = df_filtered.groupby("OverallQual")["SalePrice"].mean()
        st.line_chart(rata_harga, color="#059669")

    with c4:
        st.markdown("**Top 10 Lingkungan (Neighborhood) dengan Properti Terbanyak**")
        top_nb = df_filtered["Neighborhood"].value_counts().head(10)
        st.bar_chart(top_nb, color="#D97706")

# =========================================================
# MENU 3: PETA & LOKASI AMES
# =========================================================
elif menu == "🗺️ Peta & Lokasi Ames":
    st.subheader("Sebaran Koordinat Wilayah Ames, Iowa")
    st.write("Visualisasi titik koordinat sampel area perumahan di sekitar kota Ames, Iowa (42.0347° N, 93.6199° W).")
    
    jumlah_titik = st.slider("Jumlah titik sampel lokasi:", 10, 200, 60)
    np.random.seed(42)
    df_map = pd.DataFrame(
        np.random.randn(jumlah_titik, 2) / [80, 80] + [42.0347, -93.6199],
        columns=["lat", "lon"]
    )
    st.map(df_map)

# =========================================================
# MENU 4: SIMULASI ESTIMASI HARGA
# =========================================================
elif menu == "🧮 Simulasi Harga Rumah":
    st.subheader("Simulasi Estimasi Harga Properti (Regresi Linear Sederhana)")
    st.write("Masukkan spesifikasi rumah di bawah ini untuk menghitung estimasi harga berdasarkan pola data historis Ames Housing.")

    with st.form("form_simulasi"):
        col_a, col_b = st.columns(2)
        with col_a:
            luas = st.number_input("Luas Area Tinggal - GrLivArea (sq ft)", min_value=300, max_value=6000, value=1500, step=50)
            kualitas = st.slider("Kualitas Material & Finishing - OverallQual (1-10)", 1, 10, 7)
            tahun = st.number_input("Tahun Dibangun - YearBuilt", min_value=1870, max_value=2026, value=2005)
        with col_b:
            basement = st.number_input("Luas Basement - TotalBsmtSF (sq ft)", min_value=0, max_value=4000, value=900, step=50)
            garasi = st.selectbox("Kapasitas Garasi - GarageCars", [0, 1, 2, 3, 4], index=2)
            dapur = st.selectbox("Kualitas Dapur - KitchenQual", ["Ex (Excellent)", "Gd (Good)", "TA (Typical)", "Fa (Fair)"])

        submit_btn = st.form_submit_button("🔍 Hitung Estimasi Harga", use_container_width=True)

    if submit_btn:
        fitur_model = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageCars"]
        df_clean = df[fitur_model + ["SalePrice"]].dropna()
        
        model = LinearRegression()
        model.fit(df_clean[fitur_model], df_clean["SalePrice"])
        
        input_data = np.array([[kualitas, luas, tahun, basement, garasi]])
        prediksi = model.predict(input_data)[0]
        
        bonus_dapur = {"Ex (Excellent)": 15000, "Gd (Good)": 5000, "TA (Typical)": 0, "Fa (Fair)": -8000}
        hasil_akhir = max(35000, prediksi + bonus_dapur[dapur])

        st.success(f"### 💰 Estimasi Harga Jual: **${hasil_akhir:,.2f}**")
        st.caption("Catatan: Estimasi dihitung secara real-time menggunakan model Multiple Linear Regression dari data `train.csv`.")

# =========================================================
# MENU 5: UPLOAD & EDIT DATA
# =========================================================
elif menu == "🛠️ Upload & Edit Data":
    st.subheader("Manajemen & Editor Dataset CSV")
    uploaded = st.file_uploader("Unggah file CSV baru untuk diuji (opsional):", type="csv")
    
    data_aktif = pd.read_csv(uploaded) if uploaded is not None else df
    if uploaded is not None:
        st.success(f"Berhasil memuat file baru: {data_aktif.shape[0]} baris, {data_aktif.shape[1]} kolom.")

    st.write("Edit nilai langsung pada tabel di bawah ini, lalu unduh hasilnya:")
    edited_df = st.data_editor(data_aktif.head(15), use_container_width=True)
    
    st.download_button(
        label="⬇️ Unduh Data Hasil Edit (.csv)",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name="ames_housing_modified.csv",
        mime="text/csv"
    )
