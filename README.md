# 🎵 Song Recommender (Content‑Based)

A lightweight, **content‑based music recommender** built from Spotify metadata. It merges tracks, artists, and album data, engineers a text “tag” representation (artist, genres, playlist, lyrics), vectorizes it with `CountVectorizer`, computes **cosine similarity**, and exports ready‑to‑use Parquet artifacts for fast recommendations.

> This repository contains a single Jupyter notebook: `song_recommender.ipynb`. The notebook reads three CSVs, builds a similarity matrix, and saves artifacts you can load in any Python script or app.

---

## ⚙️ What it does (pipeline)
1. **Load data** from:
   - `input/spotify_tracks.csv`
   - `input/spotify_artists.csv`
   - `input/spotify_albums.csv`
2. **Select & clean columns** and **merge** on `track_id`.
3. **Filter** by `popularity >= 60` (keeps widely‑known tracks).
4. **Feature engineering**: build a `tags` text field from columns like `artist_name`, `genres`, `playlist`, `lyrics` (lists are parsed via `ast.literal_eval`, then normalized with a compact string like `rockindie`).
5. **Vectorization**: `sklearn.feature_extraction.text.CountVectorizer` → bag‑of‑words.
6. **Similarity**: `sklearn.metrics.pairwise.cosine_similarity` → `songsimilarity` matrix.
7. **Recommend**: a helper `recommendsong(song_name)` finds nearest neighbors.
8. **Export artifacts** (Parquet) for use outside the notebook:
   - `songparquet/song_similarity.parquet` — similarity matrix
   - `songparquet/songinfo.parquet` — song metadata (names, artist, genres, image URL, etc.)
   - `songparquet/song_list.parquet` — index → song lookup

> The notebook also extracts the **first album image URL** from the `images` JSON‑like column.

---

## 📁 Repository structure

```
.
├─ song_recommender.ipynb
├─ input/
│  ├─ spotify_tracks.csv
│  ├─ spotify_artists.csv
│  └─ spotify_albums.csv
└─ songparquet/                # created by the notebook
   ├─ song_similarity.parquet
   ├─ songinfo.parquet
   └─ song_list.parquet
```

> Make sure the three CSVs exist under `input/` before running the notebook.

---

## 🧰 Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `pyarrow`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart

1. **Clone** this repo and create a virtual environment (optional).
2. Put the three CSV files under `input/` (see “Data” below).
3. **Run the notebook** `song_recommender.ipynb` end‑to‑end.
4. The **Parquet artifacts** will be written to `songparquet/`.

### Run the notebook

```bash
jupyter lab  # or: jupyter notebook
# open song_recommender.ipynb and run all cells
```

---

## 📦 Using the recommender in your own script/app

Once you’ve generated the Parquet files, you can load them and query similar songs:

```python
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# Load artifacts
songinfo = pd.read_parquet("songparquet/songinfo.parquet")
song_list = pd.read_parquet("songparquet/song_list.parquet")
table = pq.read_table("songparquet/song_similarity.parquet")
song_similarity = np.column_stack([col.to_numpy() for col in table.itercolumns()])

# Build a name → index mapping
name_to_idx = {row['song_name_artist_name']: i for i, row in songinfo.reset_index(drop=True).iterrows()}

def recommend(song_name, top_k=10):
    idx = name_to_idx.get(song_name)
    if idx is None:
        raise ValueError(f"Song not found: {song_name}")
    scores = list(enumerate(song_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i for i, _ in scores[1:top_k+1]]  # skip self
    return songinfo.iloc[rec_indices][['song_name_artist_name','genres','playlist','images','popularity']]

# Example
print(recommend("Bowl For Two By The Expendables"))
```

---

## 🗃️ Data

This notebook expects three CSVs in `./input/` with at least these columns (based on the code):

- **tracks** (`spotify_tracks.csv`): `id` (renamed to `track_id`), `name` (→ `song_name`), `popularity`, `playlist`, `lyrics`, plus optional audio features (e.g. `danceability`, `tempo`, `valence`, etc.).
- **artists** (`spotify_artists.csv`): `track_id`, `name` (→ `artist_name`), `genres`.
- **albums** (`spotify_albums.csv`): `track_id`, `images` (list/dict with album image URLs).

> If you got the data from Kaggle or an API, please **add the source link** here and confirm column names match what the notebook uses.

---

## 🧪 Repro tips

- If `ast.literal_eval` throws on any cell (e.g., malformed JSON strings), clean those rows or wrap parsing in `try/except`.
- Ensure `popularity` is numeric; the notebook filters at `>= 60`.
- The recommender is **content‑based**; it won’t use play counts or user history.

---

## 🛣️ Next steps (ideas)

- Add a **Streamlit** UI to search and preview recommendations with album art.
- Switch to **TF‑IDF** or **BM25** to reduce popularity bias.
- Mix **audio features** (danceability, valence, tempo) with text tags for hybrid similarity.
- Deduplicate near‑identical tracks (live vs studio, remaster, etc.).
- Persist a **faiss** index for faster retrieval at scale.

---

## 📜 License

MIT — feel free to use and modify. Replace with your preferred license if needed.

---

## 🙌 Acknowledgements

- Spotify metadata (tracks, artists, albums). If you used a public dataset, **add its citation/link here**.
