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
