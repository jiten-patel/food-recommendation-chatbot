"""
Production-ready MCP Server for the Food Recommendation System.

Tools
─────
  get_restaurant_info   – structured name-based lookup
  recommend_by_vibe     – vibe/atmosphere semantic search
  get_review            – retrieve a user review

Resource
────────
  culinary-map://california – raw California restaurant text

Run:
    python -m backend.mcp_server
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from config import get_settings

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="FoodRecommendation-MCP",
    instructions=(
        "You are a California culinary expert. Use the available tools to look up "
        "restaurants, find places by vibe, and retrieve user reviews."
    ),
)

# ─── Data helpers ─────────────────────────────────────────────────────────────

def _restaurant_data() -> list[dict]:
    cfg = get_settings()
    path = cfg.data_dir / cfg.restaurant_data_file
    if not path.exists():
        logger.warning("Restaurant data file not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _review_data() -> list[dict]:
    cfg = get_settings()
    path = cfg.data_dir / cfg.user_review_file
    if not path.exists():
        logger.warning("Review data file not found: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Resource ─────────────────────────────────────────────────────────────────

@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """
    The full raw California Culinary Map text.
    Contains detailed descriptions of 100+ restaurants including vibes,
    cuisines, ratings, and price ranges.
    """
    cfg = get_settings()
    map_file = cfg.data_dir / "California-Culinary-Map.txt"
    if not map_file.exists():
        return "California Culinary Map not available."
    return map_file.read_text(encoding="utf-8")


# ─── Tool 1 – Structured search by name ──────────────────────────────────────

@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """
    Search for a restaurant by name and return its structured details
    (cuisine, rating, price range, signature dishes, environment).

    Args:
        restaurant_name: Full or partial restaurant name to search for.
    """
    restaurants = _restaurant_data()
    query = restaurant_name.lower().strip()

    matches = [
        r for r in restaurants
        if query in r.get("name", "").lower() or r.get("name", "").lower() in query
    ]

    if not matches:
        return json.dumps({
            "status": "not_found",
            "message": f"No restaurant found matching '{restaurant_name}'.",
            "suggestion": "Try a partial name, e.g. 'Iron' or 'Sakura'.",
        }, indent=2)

    return json.dumps({
        "status": "found",
        "count": len(matches),
        "results": matches,
    }, indent=2)


# ─── Tool 2 – Vibe / atmosphere search ───────────────────────────────────────

@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """
    Find restaurants that match a given vibe or atmosphere keyword.
    Searches both structured vibe fields and free-text descriptions.

    Args:
        vibe: Atmosphere keyword, e.g. 'moody', 'sun-drenched', 'romantic'.
    """
    restaurants = _restaurant_data()
    vibe_lower = vibe.lower().strip()

    structured_matches: list[dict] = []
    for r in restaurants:
        r_vibe = str(r.get("vibe", "")).lower()
        r_env  = str(r.get("environment", "")).lower()
        if vibe_lower in r_vibe or vibe_lower in r_env:
            structured_matches.append({
                "name":        r.get("name"),
                "location":    r.get("location"),
                "cuisine":     r.get("food_style"),
                "rating":      r.get("rating"),
                "vibe":        r.get("vibe"),
                "price_range": r.get("price_range"),
            })

    return json.dumps({
        "vibe_searched":     vibe,
        "structured_matches": structured_matches,
        "match_count":       len(structured_matches),
    }, indent=2)


# ─── Tool 3 – Get user review ─────────────────────────────────────────────────

@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """
    Retrieve the full user review for a restaurant, including rating,
    review text, visit date, and image description (if available).

    Args:
        restaurant_name: Full or partial restaurant name.
    """
    reviews = _review_data()
    query = restaurant_name.lower().strip()

    matching = next(
        (r for r in reviews if query in r.get("restaurant_name", "").lower()),
        None,
    )

    if matching is None:
        return json.dumps({
            "status": "not_found",
            "message": f"No review found for '{restaurant_name}'.",
        }, indent=2)

    return json.dumps({
        "status":            "found",
        "restaurant":        matching.get("restaurant_name"),
        "reviewer":          matching.get("reviewer"),
        "rating":            matching.get("rating"),
        "review_text":       matching.get("review_text") or matching.get("text"),
        "image_description": matching.get("image_description", "N/A"),
        "visit_date":        matching.get("visit_date", "N/A"),
    }, indent=2)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    mcp.run()
