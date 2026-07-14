"""
Restaurant data management service.

Handles CRUD operations on the local JSON restaurant database,
with LLM-powered paragraph → structured JSON extraction and
self-healing JSON validation.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from config import get_settings
from models import Restaurant

logger = logging.getLogger(__name__)


def _data_path() -> Path:
    cfg = get_settings()
    return cfg.data_dir / cfg.restaurant_data_file


def _backup_path() -> Path:
    return _data_path().with_suffix(".json.bak")


# ─── Low-level helpers ────────────────────────────────────────────────────────

def load_restaurants() -> list[dict]:
    path = _data_path()
    if not path.exists():
        logger.warning("Restaurant data file not found at %s — returning empty list.", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_restaurants(data: list[dict]) -> None:
    path = _data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via temp file
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)
    logger.info("Saved %d restaurant records to %s", len(data), path)


def backup_restaurants() -> None:
    src = _data_path()
    if src.exists():
        shutil.copy2(src, _backup_path())
        logger.info("Backup written to %s", _backup_path())


# ─── LLM helpers ─────────────────────────────────────────────────────────────

_EXAMPLE_INPUT = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. Rating: **4.2/5**. "
    "Price range: $$"
)
_EXAMPLE_OUTPUT = """{
  "name": "Mar de Cortez",
  "location": "Santa Monica",
  "type": "casual taqueria",
  "food_style": "Baja-style seafood",
  "rating": 4.2,
  "price_range": 2,
  "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
  "vibe": "salt-air energy",
  "environment": "sun-drenched spot for open-air dining near the pier",
  "shortcomings": []
}"""

_SYSTEM_EXTRACT = """You are an expert information-extraction assistant.
Extract structured restaurant information from free text and return ONLY valid JSON.

Rules:
1. Return ONLY a JSON object — no markdown fences, no explanations.
2. Missing fields → "" (strings), [] (arrays), null (numbers).
3. Price range: "$"→1, "$$"→2, "$$$"→3, "$$$$"→4.
4. Fields: name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings.
5. Never hallucinate; extract only what is present.
"""

_SYSTEM_REPAIR = """You are a JSON repair assistant.
Return ONLY valid, corrected JSON. No markdown, no explanations.
Preserve all original data; fix only syntax errors."""


def _llm_call(system_msg: str, user_msg: str) -> str:
    """Call the configured LLM (OpenAI or WatsonX fallback)."""
    cfg = get_settings()

    if cfg.openai_api_key:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

def parse_restaurant_paragraph(paragraph: str) -> dict:
    """
    Use the LLM to convert a free-text restaurant description into a
    validated Restaurant dict, with up to 3 self-healing retries.
    """
    user_prompt = (
        f"Extract restaurant data from this description:\n\n{paragraph}\n\n"
        f"Example input:\n{_EXAMPLE_INPUT}\n\nExample output:\n{_EXAMPLE_OUTPUT}"
    )
    response = _llm_call(_SYSTEM_EXTRACT, user_prompt)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Strip accidental markdown fences
            cleaned = response.strip().strip("```json").strip("```").strip()
            data = json.loads(cleaned)
            Restaurant.model_validate(data)   # schema check
            return data
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Validation attempt %d failed: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                repair_prompt = (
                    f"JSON:\n{response}\n\nError:\n{exc}\n\n"
                    "Repair and return ONLY valid JSON."
                )
                response = _llm_call(_SYSTEM_REPAIR, repair_prompt)

    logger.error("Failed to produce valid restaurant JSON after %d attempts.", max_retries)
    raise ValueError("Could not parse restaurant paragraph into valid JSON after retries.")


# ─── Service layer (CRUD) ─────────────────────────────────────────────────────

def get_all_restaurants() -> list[dict]:
    return load_restaurants()


def get_restaurant_by_id(item_id: int) -> Optional[dict]:
    for r in load_restaurants():
        if r.get("itemId") == item_id:
            return r
    return None


def add_restaurant(paragraph: str) -> dict:
    data = load_restaurants()
    new_id = 1_000_000 + len(data) + 1
    restaurant = parse_restaurant_paragraph(paragraph)
    restaurant["itemId"] = new_id
    backup_restaurants()
    data.append(restaurant)
    save_restaurants(data)
    return restaurant


def update_restaurant(item_id: int, paragraph: str) -> Optional[dict]:
    data = load_restaurants()
    for idx, r in enumerate(data):
        if r.get("itemId") == item_id:
            updated = parse_restaurant_paragraph(paragraph)
            updated["itemId"] = item_id
            backup_restaurants()
            data[idx] = updated
            save_restaurants(data)
            return updated
    return None


def delete_restaurant(item_id: int) -> bool:
    data = load_restaurants()
    new_data = [r for r in data if r.get("itemId") != item_id]
    if len(new_data) == len(data):
        return False
    backup_restaurants()
    save_restaurants(new_data)
    return True
