import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="Ames Housing Analytics | Kelompok 10",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Analisis Karakteristik & Harga Properti Ames Housing")
st.caption("Implementasi Streamlit — Praktikum Machine Learning (Bab 1 & Bab 2)")

REQUIRED_COLUMNS = ["SalePrice", "OverallQual", "GrLivArea", "Neighborhood"]

# =========================================================
# 2. LOAD DATA
# =========================================================
@st.cache_data
from pathlib import Path
import pandas as pd
import streamlit as st

# Lokasi file train.csv berdasarkan lokasi kode Python
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "train.csv"

# Baca dataset
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
else:
    st.error(
        f"File train.csv tidak ditemukan di: {CSV_PATH}"
    )
    st.stop()
    # Bersihkan nama kolom
    data.columns = (
        data.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return data

def prepare_data(data):
    data = data.copy()
    # Pastikan kolom target dan fitur numerik terbaca sebagai angka
    for col in ["SalePrice", "OverallQual", "GrLivArea", "YearBuilt",
                "TotalBsmtSF", "GarageCars", "Id"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

uploaded_csv = st.sidebar.file_uploader(
    "Unggah file CSV (opsional)", type=["csv"]
)

try:
    if uploaded_csv is not None:
        df = load_csv(uploaded_csv)
        sumber_data = uploaded_csv.name
    else:
        df = load_csv("train.csv")
        sumber_data = "train.csv"
    df = prepare_data(df)
except FileNotFoundError:
    st.error("File train.csv tidak ditemukan. Letakkan train.csv satu folder dengan app.py atau unggah melalui sidebar.")
    st.stop()
except Exception as error:
    st.error(f"File CSV gagal dibaca: {error}")
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    st.error("CSV yang dipilih tidak sesuai dengan dataset Ames Housing.")
    st.write("Kolom yang tidak ditemukan:", missing_columns)
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

st.sidebar.success(f"Data aktif: {sumber_data}")
st.sidebar.caption(f"{len(df):,} baris • {len(df.columns)} kolom")
st.sidebar.divider()
st.sidebar.markdown("### 👨‍🎓 Kelompok 10")
st.sidebar.caption(
    "Universitas Pamulang\n\n"
    "• Sigit Budi Prasetyo\n"
    "• Rudianto Usfinit\n"
    "• Mayolus Paru Liu"
)
menu = st.sidebar.radio(
    "Pilih halaman:",
    [
        "🏠 Ringkasan & Dataset",
        "📊 Eksplorasi & Visualisasi",
        "🗺️ Peta & Lokasi Ames",
        "🧮 Simulasi Harga Rumah",
        "🛠️ Upload & Edit Data"
    ]
)

# =========================================================
# 3. MENU RINGKASAN
# =========================================================
if menu == "🏠 Ringkasan & Dataset":
    st.subheader("Ringkasan Dataset")

    harga = df["SalePrice"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Properti", f"{len(df):,} baris")
    c2.metric("Rata-rata Harga Jual", f"${harga.mean():,.0f}" if not harga.empty else "N/A")
    c3.metric("Median Harga Jual", f"${harga.median():,.0f}" if not harga.empty else "N/A")
    c4.metric("Harga Tertinggi", f"${harga.max():,.0f}" if not harga.empty else "N/A")

    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["📋 Pratinjau Data", "📈 Statistik Deskriptif", "ℹ️ Informasi Dataset"]
    )

    with tab1:
        st.subheader("Sampel Data")
        jumlah = st.slider("Jumlah baris yang ditampilkan", 5, 50, 10)
        st.dataframe(df.head(jumlah), use_container_width=True)

    with tab2:
        st.subheader("Statistik Fitur Numerik")
        st.dataframe(df.describe().T, use_container_width=True)

    with tab3:
        st.subheader("Informasi Dataset")
        st.write({
            "Nama dataset": "Ames Housing",
            "Sumber file": sumber_data,
            "Jumlah baris": int(df.shape[0]),
            "Jumlah kolom": int(df.shape[1]),
            "Target prediksi": "SalePrice",
            "Jumlah missing value": int(df.isna().sum().sum())
        })
        st.caption("SalePrice adalah nilai numerik kontinu, sehingga tugas prediksi harga ini termasuk regresi.")

# =========================================================
# 4. MENU EKSPLORASI
# =========================================================
elif menu == "📊 Eksplorasi & Visualisasi":
    st.subheader("Eksplorasi dan Visualisasi Data")

    min_price = int(df["SalePrice"].min())
    max_price = int(df["SalePrice"].max())
    if min_price == max_price:
        st.info("Rentang harga tidak bervariasi pada data yang dipilih.")
        filtered = df.copy()
    else:
        rentang = st.slider(
            "Filter rentang harga (USD)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=1000
        )
        filtered = df[df["SalePrice"].between(rentang[0], rentang[1])]

    if "OverallQual" in filtered.columns:
        qualities = sorted(filtered["OverallQual"].dropna().unique().tolist())
        selected_quality = st.multiselect(
            "Filter kualitas rumah (OverallQual)",
            options=qualities,
            default=qualities
        )
        filtered = filtered[filtered["OverallQual"].isin(selected_quality)]

    st.caption(f"Menampilkan {len(filtered):,} dari {len(df):,} data rumah.")
    left, right = st.columns(2)

    with left:
        fig_hist = px.histogram(
            filtered, x="SalePrice", nbins=30,
            title="Distribusi Harga Jual Rumah",
            labels={"SalePrice": "Harga Jual (USD)", "count": "Jumlah Rumah"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        fig_scatter = px.scatter(
            filtered, x="GrLivArea", y="SalePrice",
            color="OverallQual" if "OverallQual" in filtered.columns else None,
            hover_data=["Neighborhood"] if "Neighborhood" in filtered.columns else None,
            title="Luas Area Tinggal vs Harga Jual",
            labels={
                "GrLivArea": "Luas Area Tinggal (sq ft)",
                "SalePrice": "Harga Jual (USD)",
                "OverallQual": "Kualitas"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        if "OverallQual" in filtered.columns:
            average_by_quality = filtered.groupby("OverallQual")["SalePrice"].mean()
            st.subheader("Rata-rata Harga berdasarkan Kualitas")
            st.line_chart(average_by_quality)

    with right2:
        if "Neighborhood" in filtered.columns:
            top_neighborhood = filtered["Neighborhood"].value_counts().head(10)
            st.subheader("10 Lingkungan dengan Data Terbanyak")
            st.bar_chart(top_neighborhood)

# =========================================================
# 5. MENU PETA
# =========================================================
elif menu == "🗺️ Peta & Lokasi Ames":
    st.subheader("Peta Ilustratif Area Ames, Iowa")
    st.write(
        "Dataset train.csv tidak menyediakan koordinat latitude dan longitude. "
        "Peta berikut hanya menampilkan titik ilustratif di sekitar Ames, Iowa, "
        "bukan lokasi asli setiap rumah."
    )
    jumlah_titik = st.slider("Jumlah titik ilustratif", 10, 200, 60)
    rng = np.random.default_rng(42)
    map_data = pd.DataFrame({
        "lat": rng.normal(42.0347, 0.015, jumlah_titik),
        "lon": rng.normal(-93.6199, 0.02, jumlah_titik)
    })
    st.map(map_data)

# =========================================================
# 6. MENU SIMULASI HARGA
# =========================================================
elif menu == "🧮 Simulasi Harga Rumah":
    st.subheader("Simulasi Estimasi Harga Rumah")
    st.write("Model regresi linear berganda dilatih dari fitur numerik yang tersedia di train.csv.")

    features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageCars"]
    absent = [col for col in features + ["SalePrice"] if col not in df.columns]

    if absent:
        st.error(f"Kolom untuk model tidak tersedia: {absent}")
        st.stop()

    model_data = df[features + ["SalePrice"]].dropna()
    if len(model_data) < 2:
        st.error("Data valid tidak cukup untuk melatih model.")
        st.stop()

    model = LinearRegression()
    model.fit(model_data[features], model_data["SalePrice"])

    with st.form("form_simulasi"):
        a, b = st.columns(2)
        with a:
            quality = st.slider("Kualitas rumah (OverallQual)", 1, 10, 7)
            area = st.number_input("Luas area tinggal (GrLivArea, sq ft)", 300, 6000, 1500, 50)
            year = st.number_input("Tahun dibangun (YearBuilt)", 1870, 2026, 2005)
        with b:
            basement = st.number_input("Luas basement (TotalBsmtSF, sq ft)", 0, 4000, 900, 50)
            garage = st.selectbox("Kapasitas garasi (GarageCars)", [0, 1, 2, 3, 4], index=2)
        submitted = st.form_submit_button("🔍 Hitung Estimasi Harga", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "OverallQual": quality,
            "GrLivArea": area,
            "YearBuilt": year,
            "TotalBsmtSF": basement,
            "GarageCars": garage
        }])
        estimate = float(model.predict(input_data)[0])
        st.success(f"### Estimasi harga jual: **${max(0, estimate):,.0f}**")
        st.caption(
            f"Model dilatih menggunakan {len(model_data):,} baris valid. "
            "Hasil merupakan estimasi model, bukan penilaian harga resmi."
        )

# =========================================================
# 7. MENU UPLOAD & EDIT DATA
# =========================================================
elif menu == "🛠️ Upload & Edit Data":
    st.subheader("Manajemen dan Editor Dataset")
    uploaded_edit = st.file_uploader(
        "Unggah CSV lain untuk diedit (opsional)",
        type=["csv"],
        key="editor_upload"
    )

    try:
        active_data = load_csv(uploaded_edit) if uploaded_edit is not None else df
    except Exception as error:
        st.error(f"CSV tidak dapat dibaca: {error}")
        st.stop()

    st.write(f"Data aktif: {active_data.shape[0]:,} baris dan {active_data.shape[1]} kolom.")
    edited = st.data_editor(
        active_data.head(100),
        use_container_width=True,
        num_rows="dynamic"
    )
    st.caption("Editor menampilkan maksimal 100 baris pertama.")
    st.download_button(
        "⬇️ Unduh hasil edit (.csv)",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name="ames_housing_modified.csv",
        mime="text/csv"
)
