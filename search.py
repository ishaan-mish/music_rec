import streamlit as st
import pandas as pd
import csv
import ast

# === Safe CSV loader ===
def load_csv_to_df(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = []
        for row in reader:
            cleaned_row = []
            for x in row:
                try:
                    if x.startswith("[") or x.startswith("{"):
                        cleaned_row.append(ast.literal_eval(x))
                    else:
                        cleaned_row.append(x.strip('"'))
                except:
                    cleaned_row.append(x.strip('"'))
            data.append(cleaned_row)
        df = pd.DataFrame(data, columns=headers)
        df.columns = df.columns.str.strip().str.lower()
        return df

# === Load datasets ===
@st.cache_data
def load_data():
    df_artist = load_csv_to_df("data_by_artist.csv")
    df_genres = load_csv_to_df("data_by_genres.csv")
    df_year = load_csv_to_df("data_by_year.csv")
    df_w_genres = load_csv_to_df("data_w_genres.csv")
    return df_artist, df_genres, df_year, df_w_genres

df_artist, df_genres, df_year, df_w_genres = load_data()

# === Streamlit App ===
st.title("🎵 Music Explorer")
st.subheader("Search for music by Singer, Genre, or Year")

mode = st.selectbox("Select Search Mode", ["Singer", "Genre", "Year"])
query = st.text_input("Enter your search term").strip().lower()

if st.button("Search"):
    results = []

    if mode.lower() == 'singer':
        for df in [df_artist, df_genres, df_w_genres]:
            if 'artists' in df.columns:
                matches = df[df['artists'].str.lower().str.contains(query, na=False)]
                results.extend(matches['artists'].tolist())

    elif mode.lower() == 'genre':
        for df in [df_artist, df_genres, df_w_genres]:
            if 'genres' in df.columns and 'artists' in df.columns:
                df['genres_str'] = df['genres'].apply(lambda g: ', '.join(g) if isinstance(g, list) else str(g))
                matches = df[df['genres_str'].str.lower().str.contains(query, na=False)]
                results.extend(matches['artists'].tolist())

    elif mode.lower() == 'year':
        if 'year' in df_year.columns:
            matches = df_year[df_year['year'].astype(str).str.contains(query, na=False)]
            if not matches.empty:
                results.extend([f"Songs from year {query} (details unknown)"] * len(matches))

    if results:
        st.success(f"Found {len(results)} matches!")
        st.write("### 🎶 Matching Entries")
        for r in results:
            st.write(f"- {r}")
    else:
        st.warning("No matches found.")
