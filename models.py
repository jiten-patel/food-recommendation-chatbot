"""
Shared Pydantic schemas for request / response validation across the API.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Restaurant
# ─────────────────────────────────────────────

class Restaurant(BaseModel):
    itemId: Optional[int] = None
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = Field(None, ge=1, le=4)
    signatures: list[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str = ""
    shortcomings: list[str] = Field(default_factory=list)


class RestaurantCreate(BaseModel):
    paragraph: str = Field(..., min_length=10, description="Free-text restaurant description")


class RestaurantUpdate(BaseModel):
    paragraph: str = Field(..., min_length=10, description="Updated free-text restaurant description")


# ─────────────────────────────────────────────
#  Recipe
# ─────────────────────────────────────────────

class Recipe(BaseModel):
    id: Optional[int] = None
    name: str
    cuisine: str
    servings: Optional[int] = None
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    total_time: Optional[str] = None
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    image_description: Optional[str] = None


# ─────────────────────────────────────────────
#  Chat / Recommendation
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    user_input: str = Field(..., min_length=3, description="Free-text user preference description")
    recommendation_type: str = Field("both", pattern="^(restaurant|recipe|both)$")


class RecommendationItem(BaseModel):
    name: str
    reasoning: str
    cuisine: Optional[str] = None
    price: Optional[str] = None
    difficulty: Optional[str] = None


class RecommendationResponse(BaseModel):
    restaurants: list[RecommendationItem] = Field(default_factory=list)
    recipes: list[RecommendationItem] = Field(default_factory=list)
    user_profile: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
#  Retrieval / Search
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    k: int = Field(5, ge=1, le=20)
    w_text: float = Field(0.6, ge=0.0, le=1.0)
    w_img: float = Field(0.4, ge=0.0, le=1.0)
    location_filter: Optional[str] = None


class SearchHit(BaseModel):
    modality: str
    id: str
    cuisine: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    fused_score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# ─────────────────────────────────────────────
#  Generic API response wrapper
# ─────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None
