"""
Multimodal Vector Index service.

Builds and persists two ChromaDB collections:
  - restaurant_articles : text embeddings (SentenceTransformer, 384-d)
  - food_images         : image embeddings (CLIP, 512-d)

Run as a script to (re)build the index:
    python -m backend.services.vector_index
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from config import get_settings

logger = logging.getLogger(__name__)

# ─── Lazy singletons ─────────────────────────────────────────────────────────
_text_model: Optional[SentenceTransformer] = None
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_article_db: Optional[Chroma] = None
_image_db: Optional[Chroma] = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        cfg = get_settings()
        logger.info("Loading text embedding model: %s", cfg.text_embed_model)
        _text_model = SentenceTransformer(cfg.text_embed_model)
    return _text_model


def _get_clip() -> tuple[CLIPModel, CLIPProcessor]:
    global _clip_model, _clip_processor
    if _clip_model is None:
        cfg = get_settings()
        logger.info("Loading CLIP model: %s", cfg.clip_model_name)
        _clip_model = CLIPModel.from_pretrained(cfg.clip_model_name).to("cpu")
        _clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name, use_fast=True)
        _clip_model.eval()
    return _clip_model, _clip_processor


def get_dbs() -> tuple[Chroma, Chroma]:
    """Return (article_db, image_db), opening connections lazily."""
    global _article_db, _image_db
    if _article_db is None or _image_db is None:
        cfg = get_settings()
        db_dir = str(cfg.chroma_persist_dir)
        _article_db = Chroma(
            collection_name="restaurant_articles",
            persist_directory=db_dir,
        )
        _image_db = Chroma(
            collection_name="food_images",
            persist_directory=db_dir,
        )
    return _article_db, _image_db


# ─── Embedding helpers ────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = _get_text_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


@torch.no_grad()
def embed_images(paths: list[str], batch_size: int = 16) -> np.ndarray:
    clip_model, processor = _get_clip()
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i: i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to("cpu")
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


@torch.no_grad()
def embed_query_clip_text(query: str) -> np.ndarray:
    clip_model, processor = _get_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True).to("cpu")
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)


# ─── Index builder ────────────────────────────────────────────────────────────

def build_index(reset: bool = False) -> None:
    """
    Build (or rebuild) the multimodal vector index from the data files.

    Args:
        reset: If True, wipe the existing index before rebuilding.
    """
    cfg = get_settings()
    db_dir = str(cfg.chroma_persist_dir)

    if reset and os.path.isdir(db_dir):
        shutil.rmtree(db_dir)
        logger.info("Removed existing index at %s", db_dir)

    # ── Load data ──
    restaurant_file = cfg.data_dir / cfg.restaurant_data_file
    recipe_file = cfg.data_dir / cfg.recipe_data_file

    with open(restaurant_file, "r") as f:
        restaurants = json.load(f)
    with open(recipe_file, "r") as f:
        recipes = json.load(f)

    img_dir = cfg.data_dir / cfg.recipe_images_dir
    image_paths = sorted(glob.glob(str(img_dir / "**" / "*.png"), recursive=True))

    logger.info("Restaurants: %d | Recipes: %d | Images: %d",
                len(restaurants), len(recipes), len(image_paths))

    # ── Build article documents ──
    article_docs: list[Document] = []
    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}\n"
            f"Vibe: {r.get('vibe', '')}\n"
            f"Environment: {r.get('environment', '')}"
        )
        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": f"rest_{i}",
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "source": "restaurant",
                },
            )
        )

    # ── Build image documents ──
    image_docs: list[Document] = []
    paired = list(zip(image_paths, recipes))
    for i, (p, rec) in enumerate(paired):
        image_docs.append(
            Document(
                page_content=rec.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": f"img_{i}",
                    "image_path": p,
                    "source": "recipe_image",
                    "recipe_id": rec.get("id"),
                    "cuisine": rec.get("cuisine"),
                },
            )
        )

    # ── Embed and upsert articles ──
    a_db, i_db = get_dbs()

    if article_docs:
        A = embed_texts([d.page_content for d in article_docs])
        a_db._collection.upsert(
            ids=[d.metadata["doc_id"] for d in article_docs],
            embeddings=A.tolist(),
            documents=[d.page_content for d in article_docs],
            metadatas=[d.metadata for d in article_docs],
        )
        logger.info("Article DB: upserted %d records", len(article_docs))

    # ── Embed and upsert images ──
    if image_docs:
        V = embed_images([d.metadata["image_path"] for d in image_docs])
        i_db._collection.upsert(
            ids=[d.metadata["doc_id"] for d in image_docs],
            embeddings=V.tolist(),
            documents=[d.page_content for d in image_docs],
            metadatas=[d.metadata for d in image_docs],
        )
        logger.info("Image DB: upserted %d records", len(image_docs))

    logger.info("Multimodal index build complete.")


def index_ready() -> bool:
    """Return True if the index exists and is non-empty."""
    try:
        a_db, i_db = get_dbs()
        return a_db._collection.count() > 0 and i_db._collection.count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    build_index(reset=True)
    print("Index build complete.")
