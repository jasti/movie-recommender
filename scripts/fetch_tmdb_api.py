"""
fetch_tmdb_api.py — Alternative data source using the free TMDB API.

Use this if you don't have a Kaggle account. Get a free API key at:
https://www.themoviedb.org/settings/api

Usage:
    python scripts/fetch_tmdb_api.py --api-key YOUR_API_KEY

Outputs:
    data/tmdb_5000_movies.csv   — Compatible with precompute.py
    data/tmdb_5000_credits.csv  — Compatible with precompute.py
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
BASE_URL = "https://api.themoviedb.org/3"


def fetch_popular_movies(api_key: str, num_pages: int = 250) -> List[Dict]:
    """Fetch popular movies from TMDB API (20 per page)."""
    movies = []
    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/movie/popular"
        params = {"api_key": api_key, "page": page, "language": "en-US"}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        for m in data.get("results", []):
            movies.append(m)

        if page % 25 == 0:
            print(f"  Fetched {page}/{num_pages} pages ({len(movies)} movies)")
        time.sleep(0.26)  # TMDB rate limit: ~40 requests/10s

    print(f"  Total: {len(movies)} movies from popular endpoint")
    return movies


def fetch_movie_details(api_key: str, movie_id: int) -> Optional[Dict]:
    """Fetch full details + credits for a single movie."""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": api_key,
        "append_to_response": "credits,keywords",
        "language": "en-US",
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def build_csvs(api_key: str, num_pages: int = 250):
    """
    Fetch movies and convert to the same CSV format that precompute.py expects.
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: Get movie IDs from the popular endpoint
    print("📡 Fetching popular movie list...")
    popular = fetch_popular_movies(api_key, num_pages)

    # Deduplicate
    seen = set()
    unique_ids = []
    for m in popular:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique_ids.append(m["id"])
    print(f"  {len(unique_ids)} unique movie IDs")

    # Step 2: Fetch details for each movie
    print("📡 Fetching movie details (this takes ~20 min for 5000 movies)...")
    movies_rows = []
    credits_rows = []

    for i, mid in enumerate(unique_ids):
        details = fetch_movie_details(api_key, mid)
        if not details or not details.get("overview"):
            continue

        # Build movies row (matching TMDB 5000 CSV format)
        genres = json.dumps([{"id": g["id"], "name": g["name"]} for g in details.get("genres", [])])
        keywords_data = details.get("keywords", {}).get("keywords", [])
        keywords = json.dumps([{"id": k["id"], "name": k["name"]} for k in keywords_data])

        movies_rows.append({
            "id": details["id"],
            "title": details.get("title", ""),
            "overview": details.get("overview", ""),
            "genres": genres,
            "keywords": keywords,
            "release_date": details.get("release_date", ""),
            "vote_average": details.get("vote_average", 0),
            "vote_count": details.get("vote_count", 0),
            "popularity": details.get("popularity", 0),
            "poster_path": details.get("poster_path", ""),
            "original_language": details.get("original_language", ""),
        })

        # Build credits row
        credits_data = details.get("credits", {})
        cast = json.dumps([
            {"name": c["name"], "character": c.get("character", ""), "order": c.get("order", 0)}
            for c in credits_data.get("cast", [])[:10]
        ])
        crew = json.dumps([
            {"name": c["name"], "job": c.get("job", ""), "department": c.get("department", "")}
            for c in credits_data.get("crew", [])
            if c.get("job") in ("Director", "Producer", "Screenplay", "Writer")
        ])

        credits_rows.append({
            "movie_id": details["id"],
            "title": details.get("title", ""),
            "cast": cast,
            "crew": crew,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(unique_ids)} movies fetched")
        time.sleep(0.26)

    # Step 3: Save as CSVs
    movies_df = pd.DataFrame(movies_rows)
    credits_df = pd.DataFrame(credits_rows)

    movies_path = DATA_DIR / "tmdb_5000_movies.csv"
    credits_path = DATA_DIR / "tmdb_5000_credits.csv"

    movies_df.to_csv(movies_path, index=False)
    credits_df.to_csv(credits_path, index=False)

    print(f"\n💾 Saved {len(movies_df)} movies:")
    print(f"   {movies_path}")
    print(f"   {credits_path}")
    print("✅ Done! Now run: python precompute.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch movie data from TMDB API")
    parser.add_argument("--api-key", required=True, help="Your TMDB API key (v3)")
    parser.add_argument("--pages", type=int, default=250, help="Number of pages to fetch (20 movies/page)")
    args = parser.parse_args()

    build_csvs(args.api_key, args.pages)
