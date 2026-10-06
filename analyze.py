"""
Analysis layer: turn text into vectors (embeddings), then group similar
videos into clusters. Each cluster is a candidate "pattern" in the data.

Embeddings are pluggable:
  - "sentence-transformers": semantic quality, recommended for real use.
  - "tfidf": offline sklearn fallback, no downloads, good for quick tests.

Clustering uses KMeans (simple, fast, deterministic with a fixed seed).
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.cluster import KMeans

import config
from ingest import Video


# --- Embeddings ---------------------------------------------------------------

def embed_texts(texts: list[str], backend: str) -> np.ndarray:
    """Return an (n_texts, dim) matrix of L2-normalized embeddings."""
    if backend == "sentence-transformers":
        vectors = _embed_sentence_transformers(texts)
    elif backend == "tfidf":
        vectors = _embed_tfidf(texts)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")

    # Normalize so cosine similarity == dot product; helps KMeans behave.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.ST_MODEL_NAME)
    return np.asarray(model.encode(texts, show_progress_bar=False))


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    return vec.fit_transform(texts).toarray()


# --- Clustering ---------------------------------------------------------------

def cluster_videos(
    videos: list[Video],
    n_clusters: int,
    backend: str,
) -> tuple[list[int], np.ndarray]:
    """Assign each video to a cluster.

    Returns (labels, embeddings) where labels[i] is the cluster id of videos[i].
    """
    texts = [v.analysis_text() for v in videos]
    embeddings = embed_texts(texts, backend)

    # Never ask for more clusters than we have samples.
    k = min(n_clusters, len(videos))
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embeddings).tolist()
    return labels, embeddings


def group_by_cluster(
    videos: list[Video], labels: list[int]
) -> dict[int, list[Video]]:
    """Group videos into {cluster_id: [videos...]}, largest cluster first."""
    clusters: dict[int, list[Video]] = {}
    for video, label in zip(videos, labels):
        clusters.setdefault(label, []).append(video)
    # Sort clusters by size (descending) for nicer reporting.
    return dict(sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True))


def cluster_keywords(videos: list[Video], top_n: int = 6) -> list[str]:
    """Cheap, dependency-light keyword summary of a cluster.

    Counts hashtags across the cluster's videos and returns the most common.
    Used both as a fallback label and as a hint for the LLM explainer.
    """
    counts: dict[str, int] = {}
    for v in videos:
        for h in v.hashtags:
            counts[h] = counts.get(h, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [h for h, _ in ranked[:top_n]]


def distinctive_cluster_keywords(
    clusters: dict[int, list[Video]], top_n: int = 6
) -> dict[int, list[str]]:
    """Per-cluster hashtags, ranked by TF-ICF (term frequency x inverse
    cluster frequency) instead of raw count.

    A hashtag-search scrape means the search seed tags (e.g. #jobsearch,
    #resumetips) show up in nearly every cluster, so plain cluster_keywords()
    picks the same 2-3 tags for every cluster and every theme label ends up
    looking alike. Down-weighting tags that recur across many clusters (like
    idf does for documents) surfaces what actually makes each cluster
    different.
    """
    per_cluster_counts: dict[int, dict[str, int]] = {}
    doc_freq: dict[str, int] = {}
    for cid, videos in clusters.items():
        counts: dict[str, int] = {}
        for v in videos:
            for h in v.hashtags:
                counts[h] = counts.get(h, 0) + 1
        per_cluster_counts[cid] = counts
        for h in counts:
            doc_freq[h] = doc_freq.get(h, 0) + 1

    n_clusters = max(len(clusters), 1)
    result: dict[int, list[str]] = {}
    for cid, counts in per_cluster_counts.items():
        scored = [
            (h, count * math.log((1 + n_clusters) / (1 + doc_freq[h])))
            for h, count in counts.items()
        ]
        # Fall back to raw frequency for the tie-break so a cluster with no
        # distinctive tags at all still gets its most common ones back.
        scored.sort(key=lambda hs: (hs[1], counts[hs[0]]), reverse=True)
        result[cid] = [h for h, _ in scored[:top_n]]
    return result
