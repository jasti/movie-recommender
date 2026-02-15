"""
precompute.py — Build the movie embedding index.

Parses the TMDB 5000 dataset, constructs a rich text representation
for each movie (plot + genres + cast + director + keywords), generates
embeddings with sentence-transformers, and saves a FAISS index to disk.

Usage:
    python precompute.py

Expects:
    data/tmdb_5000_movies.csv
    data/tmdb_5000_credits.csv

Outputs:
    data/faiss_index.bin      — FAISS cosine-similarity index
    data/movies_meta.pkl      — Movie metadata (title, year, genres, overview, etc.)
"""

import ast
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
BATCH_SIZE = 128


# ── Data Parsing Helpers ─────────────────────────────────────────────

def safe_json_parse(val):
    """Parse a JSON-like string column from the TMDB CSV."""
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []


def extract_names(json_list, key="name", top_n=None):
    """Pull 'name' values from a list of dicts."""
    names = [item[key] for item in json_list if key in item]
    return names[:top_n] if top_n else names


def get_director(crew_list):
    """Find the director from a crew list."""
    for person in crew_list:
        if person.get("job") == "Director":
            return person.get("name", "")
    return ""


# ── Main Pipeline ────────────────────────────────────────────────────

def load_and_merge():
    """Load the two TMDB CSVs and merge on movie_id."""
    movies_path = DATA_DIR / "tmdb_5000_movies.csv"
    credits_path = DATA_DIR / "tmdb_5000_credits.csv"

    if not movies_path.exists() or not credits_path.exists():
        print("❌ Dataset not found. Please download the TMDB 5000 dataset:")
        print("   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print(f"   Place both CSVs in: {DATA_DIR.resolve()}")
        raise FileNotFoundError("Missing dataset files")

    movies = pd.read_csv(movies_path, engine="python", on_bad_lines="skip")
    credits = pd.read_csv(credits_path, engine="python", on_bad_lines="skip")

    # The credits CSV has 'movie_id' while movies has 'id'
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})

    # Ensure matching types for merge key
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    credits = credits.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)
    credits["id"] = credits["id"].astype(int)

    df = movies.merge(credits, on="id", suffixes=("", "_credits"))
    print(f"✅ Loaded {len(df)} movies")
    return df


def build_rich_text(row):
    """
    Construct a single text string that captures the movie's identity:
    genres, director, cast, plot overview, and thematic keywords.

    Example output:
    "Drama, Thriller. Directed by Christopher Nolan. Starring Leonardo DiCaprio,
    Joseph Gordon-Levitt, Elliot Page. A thief who steals corporate secrets
    through dream-sharing technology... Keywords: dream, subconscious, heist."
    """
    # Genres
    genres = extract_names(safe_json_parse(row.get("genres", "[]")))
    genre_str = ", ".join(genres) if genres else "Unknown genre"

    # Director
    crew = safe_json_parse(row.get("crew", "[]"))
    director = get_director(crew)
    director_str = f"Directed by {director}" if director else ""

    # Top 3 cast
    cast = extract_names(safe_json_parse(row.get("cast", "[]")), top_n=3)
    cast_str = f"Starring {', '.join(cast)}" if cast else ""

    # Overview (plot)
    overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""

    # Keywords
    keywords = extract_names(safe_json_parse(row.get("keywords", "[]")))
    kw_str = f"Keywords: {', '.join(keywords)}" if keywords else ""

    # Combine all parts
    parts = [p for p in [genre_str, director_str, cast_str, overview, kw_str] if p]
    return ". ".join(parts)


def precompute():
    """Run the full pipeline: load → text → embed → index → save."""
    # 1. Load data
    df = load_and_merge()

    # 2. Build rich text for each movie
    print("📝 Building rich text representations...")
    df["rich_text"] = df.apply(build_rich_text, axis=1)

    # Drop movies with empty text
    df = df[df["rich_text"].str.len() > 20].reset_index(drop=True)
    print(f"   {len(df)} movies with valid text")

    # 3. Generate embeddings
    print(f"🧠 Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🔢 Encoding {len(df)} movies (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    embeddings = model.encode(
        df["rich_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity via inner product
    )
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s — shape: {embeddings.shape}")

    # 4. Build FAISS index (Inner Product = cosine similarity on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"🗂️  FAISS index built: {index.ntotal} vectors, {dim} dimensions")

    # 5. Save metadata (include poster_path for UI)
    keep_cols = ["id", "title", "release_date", "genres", "overview", "vote_average"]
    if "poster_path" in df.columns:
        keep_cols.append("poster_path")
    meta = df[keep_cols].copy()
    meta["genres_list"] = meta["genres"].apply(
        lambda x: extract_names(safe_json_parse(x))
    )
    # Extract year from release_date
    meta["year"] = pd.to_datetime(meta["release_date"], errors="coerce").dt.year
    # Build poster URL
    if "poster_path" in meta.columns:
        meta["poster_url"] = meta["poster_path"].apply(
            lambda p: f"https://image.tmdb.org/t/p/w300{p}" if pd.notna(p) and p else None
        )

    # Save everything
    faiss.write_index(index, str(DATA_DIR / "faiss_index.bin"))
    meta.to_pickle(DATA_DIR / "movies_meta.pkl")

    print(f"\n💾 Saved to {DATA_DIR.resolve()}:")
    print(f"   faiss_index.bin ({(DATA_DIR / 'faiss_index.bin').stat().st_size / 1e6:.1f} MB)")
    print(f"   movies_meta.pkl")
    print("✅ Precompute complete!")

    # Quick sanity check
    print("\n🔍 Sanity check — movies similar to the first entry:")
    query = embeddings[0:1]
    distances, indices = index.search(query, 4)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        title = meta.iloc[idx]["title"]
        year = meta.iloc[idx]["year"]
        print(f"   {rank}. {title} ({year}) — similarity: {dist:.3f}")


if __name__ == "__main__":
    precompute()
