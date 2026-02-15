"""
app.py — Streamlit UI for the Movie Semantic Search Recommender.

Given a movie, finds the 3 most similar movies using pre-computed
sentence-transformer embeddings + FAISS cosine similarity.

Usage:
    streamlit run app.py
"""

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TOP_K = 3
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w300"
MIN_RATING = 7.0  # Minimum vote_average to recommend


# ── Load Resources (cached across reruns) ─────────────────────────────

@st.cache_resource
def load_index():
    index_path = DATA_DIR / "faiss_index.bin"
    if not index_path.exists():
        st.error("FAISS index not found. Run `python precompute.py` first.")
        st.stop()
    return faiss.read_index(str(index_path))


@st.cache_resource
def load_metadata():
    meta_path = DATA_DIR / "movies_meta.pkl"
    if not meta_path.exists():
        st.error("Metadata not found. Run `python precompute.py` first.")
        st.stop()
    return pd.read_pickle(meta_path)


# ── Search Logic ──────────────────────────────────────────────────────

def find_similar(movie_idx: int, top_k: int = TOP_K):
    index = load_index()
    meta = load_metadata()
    query_vec = index.reconstruct(int(movie_idx)).reshape(1, -1)

    # Fetch more candidates to ensure we have enough after filtering
    search_k = min(50, len(meta))
    distances, indices = index.search(query_vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == movie_idx:
            continue

        # Filter by minimum rating
        movie = meta.iloc[idx]
        rating = movie.get("vote_average", 0)
        if pd.notna(rating) and rating >= MIN_RATING:
            results.append((int(idx), float(dist)))

        # Stop once we have enough highly rated recommendations
        if len(results) >= top_k:
            break

    return results[:top_k]


# ── Movie Card (native Streamlit) ────────────────────────────────────

def get_poster_url(movie):
    """Get poster URL from movie data."""
    poster_url = movie.get("poster_url", None)
    if poster_url and not (isinstance(poster_url, float) and pd.isna(poster_url)):
        return poster_url
    pp = movie.get("poster_path", "")
    if pp and pd.notna(pp):
        return f"{TMDB_IMG_BASE}{pp}"
    return None


def render_movie_card(movie, similarity=None, is_source=False):
    """Render a movie card using native Streamlit components."""
    poster_url = get_poster_url(movie)
    title = movie["title"]
    year = int(movie["year"]) if pd.notna(movie.get("year")) else ""
    genres = movie.get("genres_list", [])
    if isinstance(genres, str):
        genre_str = genres
    elif isinstance(genres, list):
        genre_str = ", ".join(genres)
    else:
        genre_str = ""
    overview = str(movie.get("overview", ""))
    if len(overview) > 300:
        overview = overview[:300] + "..."
    rating = movie.get("vote_average", 0)

    # Use columns for poster + info layout
    if poster_url:
        col_img, col_info = st.columns([1, 4])
        with col_img:
            st.image(poster_url, width=120)
        with col_info:
            _render_movie_info(title, year, genre_str, overview, rating, similarity, is_source)
    else:
        _render_movie_info(title, year, genre_str, overview, rating, similarity, is_source)


def _render_movie_info(title, year, genre_str, overview, rating, similarity, is_source):
    """Render the text portion of a movie card."""
    if is_source:
        st.markdown(f"**{title}** ({year})")
    else:
        st.markdown(f"**{title}** ({year})")

    if similarity is not None:
        st.markdown(f":green[**{similarity:.0%} match**]")

    if genre_str:
        st.caption(genre_str)

    if overview:
        st.markdown(f"<small style='color: #aaa;'>{overview}</small>", unsafe_allow_html=True)

    if pd.notna(rating) and rating > 0:
        st.markdown(f"⭐ {rating:.1f}/10")


# ── Main App ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Semantic Movie Recommender",
        page_icon="🎬",
        layout="centered",
    )

    # Header
    st.title("🎬 Semantic Movie Recommender")
    st.markdown("*Find movies that feel like the one you love — powered by AI embeddings, not genre tags.*")

    # Load data
    meta = load_metadata()

    # Build display labels
    meta["label"] = meta.apply(
        lambda r: f"{r['title']} ({int(r['year'])})" if pd.notna(r.get("year")) else r["title"],
        axis=1,
    )

    # Movie selector
    sorted_labels = meta["label"].sort_values().tolist()
    selected_label = st.selectbox(
        "Choose a movie",
        options=sorted_labels,
        index=None,
        placeholder="Start typing a movie name...",
        label_visibility="collapsed",
    )

    if selected_label is None:
        st.markdown("")
        st.markdown(
            "<p style='text-align: center; color: #888;'>Pick a movie above to get 3 semantically similar recommendations.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; color: #666; font-size: 0.9rem;'>Try: Inception, The Matrix, Toy Story, Pulp Fiction, The Godfather</p>",
            unsafe_allow_html=True,
        )
    else:
        # Find selected movie
        selected_idx = meta[meta["label"] == selected_label].index[0]
        selected_movie = meta.iloc[selected_idx]

        # Show source movie
        st.markdown("---")
        st.subheader("Because you picked")
        render_movie_card(selected_movie, is_source=True)

        # Find and show similar movies
        st.markdown("---")
        st.subheader("You might also love")

        with st.spinner("Finding similar movies..."):
            results = find_similar(selected_idx)

        for i, (idx, score) in enumerate(results):
            movie = meta.iloc[idx]
            render_movie_card(movie, similarity=score)
            if i < len(results) - 1:
                st.markdown("---")

    # Footer
    st.markdown("---")
    st.caption(
        "Built with sentence-transformers + FAISS. ~4,800 movies. Zero inference cost. "
        "Embeddings combine plot, genre, cast, director & keywords."
    )


if __name__ == "__main__":
    main()
