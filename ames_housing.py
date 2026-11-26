import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="Ames Housing - Kelompok 10",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "train.csv"

# ==================================================
# 2. FUNGSI MEMBACA DATASET
# ==================================================

def load_csv(file):
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")

        # Membersihkan nama kolom
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        return df

    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()


def load_default_dataset():
    if CSV_PATH.exists():
        return load_csv(CSV_PATH)

    return None


# ==================================================
# 3. SIDEBAR DAN UPLOAD DATASET
# ==================================================

st.sidebar.title("🏠 Kelompok 10")
st.sidebar.write("Universitas Pamulang")
st.sidebar.write(
    "Sigit Budi Prasetyo • Rudianto Usfinit • Mayolus Paru Liu"
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Unggah file CSV (opsional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.sidebar.success(
        f"Data aktif: {uploaded_file.name}"
    )
else:
    df = load_default_dataset()

    if df is not None:
        st.sidebar.success("Data aktif: train.csv")
    else:
        st.sidebar.error("File train.csv tidak ditemukan.")

st.sidebar.divider()

menu = st.sidebar.radio(
    "Pilih halaman:",
    [
        "🏠 Ringkasan & Dataset",
        "📊 Eksplorasi & Visualisasi",
        "🗺️ Peta & Lokasi Ames",
        "🏡 Simulasi Harga Rumah",
        "🛠️ Upload & Edit Data"
    ]
)

# ==================================================
# 4. JUDUL APLIKASI
# ==================================================

st.title(
    "🏡 Analisis Karakteristik & Harga Properti Ames Housing"
)

st.write(
    "Implementasi Streamlit — Praktikum Machine Learning (Bab 1 & Bab 2)"
)

# ==================================================
# 5. PEMERIKSAAN DATASET
# ==================================================

if df is None:
    st.error(
        "File train.csv tidak ditemukan. "
        "Letakkan train.csv satu folder dengan ames_housing.py "
        "atau unggah melalui sidebar."
    )
    st.stop()

if df.empty:
    st.error("Dataset kosong.")
    st.stop()

st.sidebar.caption(
    f"{len(df):,} baris • {len(df.columns)} kolom"
)

# Kolom yang digunakan pada analisis
required_columns = [
    "SalePrice",
    "OverallQual",
    "GrLivArea",
    "Neighborhood"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

# ==================================================
# 6. HALAMAN RINGKASAN
# ==================================================

if menu == "🏠 Ringkasan & Dataset":

    st.header("Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Data Properti",
        f"{len(df):,} Baris"
    )

    if "SalePrice" in df.columns:
        prices = pd.to_numeric(
            df["SalePrice"], errors="coerce"
        ).dropna()

        if len(prices) > 0:
            col2.metric(
                "Rata-rata Harga Jual",
                f"${prices.mean():,.0f}"
            )

            col3.metric(
                "Median Harga Jual",
                f"${prices.median():,.0f}"
            )

            col4.metric(
                "Harga Tertinggi",
                f"${prices.max():,.0f}"
            )

    st.divider()

    st.subheader("Informasi Dataset")

    st.write(
        "Dataset Ames Housing berisi informasi mengenai "
        "karakteristik rumah dan harga jualnya."
    )

    st.write(
        f"Jumlah baris: {len(df):,}"
    )

    st.write(
        f"Jumlah kolom: {len(df.columns)}"
    )

    st.write(
        f"Jumlah missing value: {df.isna().sum().sum():,}"
    )

    st.subheader("Preview Dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Informasi Tipe Data")

    info_df = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Missing Value": df.isna().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# ==================================================
# 7. HALAMAN EKSPLORASI DAN VISUALISASI
# ==================================================

elif menu == "📊 Eksplorasi & Visualisasi":

    st.header("Eksplorasi dan Visualisasi Data")

    if "SalePrice" not in df.columns:
        st.warning(
            "Kolom SalePrice tidak ditemukan dalam dataset."
        )
        st.stop()

    prices = pd.to_numeric(
        df["SalePrice"], errors="coerce"
    ).dropna()

    st.subheader("Distribusi Harga Rumah")

    st.bar_chart(
        prices.value_counts().sort_index(),
        use_container_width=True
    )

    if "OverallQual" in df.columns:
        st.subheader(
            "Hubungan Kualitas Rumah dengan Harga Jual"
        )

        quality_df = df[
            ["OverallQual", "SalePrice"]
        ].copy()

        quality_df["OverallQual"] = pd.to_numeric(
            quality_df["OverallQual"], errors="coerce"
        )

        quality_df["SalePrice"] = pd.to_numeric(
            quality_df["SalePrice"], errors="coerce"
        )

        quality_df = quality_df.dropna()

        quality_mean = (
            quality_df.groupby("OverallQual")["SalePrice"]
            .mean()
            .sort_index()
        )

        st.line_chart(
            quality_mean,
            use_container_width=True
        )

    if "GrLivArea" in df.columns:
        st.subheader(
            "Hubungan Luas Rumah dengan Harga Jual"
        )

        area_df = df[
            ["GrLivArea", "SalePrice"]
        ].copy()

        area_df["GrLivArea"] = pd.to_numeric(
            area_df["GrLivArea"], errors="coerce"
        )

        area_df["SalePrice"] = pd.to_numeric(
            area_df["SalePrice"], errors="coerce"
        )

        area_df = area_df.dropna()

        st.scatter_chart(
            area_df,
            x="GrLivArea",
            y="SalePrice",
            use_container_width=True
        )

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

# ==================================================
# 8. HALAMAN PETA DAN LOKASI
# ==================================================

elif menu == "🗺️ Peta & Lokasi Ames":

    st.header("Peta dan Lokasi Ames Housing")

    st.write(
        "Dataset Ames Housing berisi data properti "
        "di Ames, Iowa, Amerika Serikat."
    )

    st.info(
        "Dataset ini tidak menyediakan koordinat GPS "
        "untuk setiap rumah. Oleh karena itu, aplikasi "
        "tidak menampilkan lokasi rumah secara individual."
    )

    if "Neighborhood" in df.columns:

        st.subheader("Jumlah Properti Berdasarkan Wilayah")

        neighborhood_count = (
            df["Neighborhood"]
            .fillna("Tidak diketahui")
            .value_counts()
        )

        st.bar_chart(
            neighborhood_count,
            use_container_width=True
        )

        st.subheader("Harga Rata-rata Berdasarkan Wilayah")

        if "SalePrice" in df.columns:

            neighborhood_df = df[
                ["Neighborhood", "SalePrice"]
            ].copy()

            neighborhood_df["SalePrice"] = pd.to_numeric(
                neighborhood_df["SalePrice"],
                errors="coerce"
            )

            neighborhood_mean = (
                neighborhood_df
                .groupby("Neighborhood")["SalePrice"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(
                neighborhood_mean,
                use_container_width=True
            )

# ==================================================
# 9. HALAMAN SIMULASI HARGA RUMAH
# ==================================================

elif menu == "🏡 Simulasi Harga Rumah":

    st.header("Simulasi Harga Rumah")

    st.write(
        "Simulasi ini menggunakan harga rata-rata "
        "berdasarkan kualitas rumah pada dataset."
    )

    st.info(
        "Hasil simulasi merupakan perkiraan sederhana "
        "dan bukan hasil prediksi model Machine Learning."
    )

    if (
        "OverallQual" not in df.columns
        or "SalePrice" not in df.columns
    ):
        st.warning(
            "Kolom OverallQual atau SalePrice tidak ditemukan."
        )
        st.stop()

    quality_data = df[
        ["OverallQual", "SalePrice"]
    ].copy()

    quality_data["OverallQual"] = pd.to_numeric(
        quality_data["OverallQual"], errors="coerce"
    )

    quality_data["SalePrice"] = pd.to_numeric(
        quality_data["SalePrice"], errors="coerce"
    )

    quality_data = quality_data.dropna()

    if quality_data.empty:
        st.warning("Tidak ada data yang dapat digunakan.")
        st.stop()

    min_quality = int(quality_data["OverallQual"].min())
    max_quality = int(quality_data["OverallQual"].max())

    selected_quality = st.slider(
        "Pilih kualitas rumah",
        min_value=min_quality,
        max_value=max_quality,
        value=min_quality
    )

    selected_data = quality_data[
        quality_data["OverallQual"] == selected_quality
    ]

    if not selected_data.empty:

        estimated_price = selected_data["SalePrice"].mean()

        st.metric(
            "Perkiraan Harga Rumah",
            f"${estimated_price:,.0f}"
        )

        st.write(
            f"Rata-rata harga rumah dengan kualitas "
            f"{selected_quality} adalah "
            f"${estimated_price:,.0f}."
        )

# ==================================================
# 10. HALAMAN UPLOAD DAN EDIT DATA
# ==================================================

elif menu == "🛠️ Upload & Edit Data":

    st.header("Upload dan Edit Data")

    st.write(
        "Gunakan menu upload di sidebar untuk mengganti "
        "dataset yang sedang digunakan."
    )

    st.subheader("Data Aktif")

    st.write(
        f"Jumlah baris: {len(df):,}"
    )

    st.write(
        f"Jumlah kolom: {len(df.columns)}"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Download Dataset")

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="ames_housing_data.csv",
        mime="text/csv"
    )

# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "Kelompok 10 | Machine Learning | Universitas Pamulang"
)
