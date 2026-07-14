"""
FastAPI application – production-ready REST API for the Food Recommendation system.

Endpoints
─────────
GET    /api/health                          Health check
GET    /api/restaurants                     List all restaurants
GET    /api/restaurants/{item_id}           Get single restaurant
POST   /api/restaurants                     Add restaurant (paragraph)
PUT    /api/restaurants/{item_id}           Update restaurant (paragraph)
DELETE /api/restaurants/{item_id}           Delete restaurant

GET    /api/recipes                         List all recipes

POST   /api/search                          Multimodal semantic search
POST   /api/recommend                       Run full multi-agent recommendation
POST   /api/chat                            Conversational chat interface

POST   /api/index/build                     (Admin) Rebuild vector index
GET    /api/index/status                    Check index readiness
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from config import get_settings
from models import (
    APIResponse,
    ChatRequest,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    RestaurantCreate,
    RestaurantUpdate,
    SearchRequest,
    SearchResponse,
    SearchHit,
)
from services import restaurant_service, retrieval_service
from services.agents import (
    classify_intent,
    extract_preferences,
    run_recommendation_workflow,
)
from services.vector_index import build_index, index_ready

# ─── App setup ────────────────────────────────────────────────────────────────

cfg = get_settings()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Multimodal Restaurant Recommendation System",
    description=(
        "A production-ready API combining multi-agent AI orchestration, "
        "multimodal vector search (text + images), and an LLM-powered chat interface."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": time.time()}


# ═══════════════════════════════════════════════════════════════════════════════
#  RESTAURANTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/restaurants", response_model=APIResponse, tags=["Restaurants"])
def list_restaurants():
    data = restaurant_service.get_all_restaurants()
    return APIResponse(data=data, message=f"{len(data)} restaurants found")


@app.get("/api/restaurants/{item_id}", response_model=APIResponse, tags=["Restaurants"])
def get_restaurant(item_id: int):
    restaurant = restaurant_service.get_restaurant_by_id(item_id)
    if restaurant is None:
        raise HTTPException(status_code=404, detail=f"Restaurant {item_id} not found")
    return APIResponse(data=restaurant)


@app.post("/api/restaurants", response_model=APIResponse, status_code=status.HTTP_201_CREATED, tags=["Restaurants"])
def create_restaurant(payload: RestaurantCreate):
    try:
        restaurant = restaurant_service.add_restaurant(payload.paragraph)
        return APIResponse(data=restaurant, message=f"Restaurant '{restaurant.get('name')}' added successfully")
    except Exception as exc:
        logger.exception("Failed to add restaurant")
        raise HTTPException(status_code=422, detail=str(exc))


@app.put("/api/restaurants/{item_id}", response_model=APIResponse, tags=["Restaurants"])
def update_restaurant(item_id: int, payload: RestaurantUpdate):
    try:
        updated = restaurant_service.update_restaurant(item_id, payload.paragraph)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Restaurant {item_id} not found")
        return APIResponse(data=updated, message="Restaurant updated successfully")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to update restaurant")
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/api/restaurants/{item_id}", response_model=APIResponse, tags=["Restaurants"])
def delete_restaurant(item_id: int):
    deleted = restaurant_service.delete_restaurant(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Restaurant {item_id} not found")
    return APIResponse(message=f"Restaurant {item_id} deleted")


# ═══════════════════════════════════════════════════════════════════════════════
#  RECIPES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/recipes", response_model=APIResponse, tags=["Recipes"])
def list_recipes():
    recipe_file = cfg.data_dir / cfg.recipe_data_file
    if not recipe_file.exists():
        return APIResponse(data=[], message="Recipe file not found")
    with open(recipe_file, "r") as f:
        data = json.load(f)
    return APIResponse(data=data, message=f"{len(data)} recipes found")


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTIMODAL SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
def semantic_search(payload: SearchRequest):
    if not index_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector index is not ready. Call POST /api/index/build first.",
        )
    where_text = {"location": payload.location_filter} if payload.location_filter else None
    rows = retrieval_service.fuse_rank(
        query=payload.query,
        k_text=payload.k,
        k_img=payload.k,
        w_text=payload.w_text,
        w_img=payload.w_img,
        where_text=where_text,
        top_n=payload.k,
    )
    hits = [
        SearchHit(
            modality=r["modality"],
            id=r["id"],
            cuisine=r.get("cuisine"),
            location=r.get("location"),
            source=r.get("source"),
            fused_score=r["fused_score"],
            snippet=r["snippet"],
        )
        for r in rows
    ]
    return SearchResponse(query=payload.query, hits=hits)


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-AGENT RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
def recommend(payload: RecommendationRequest):
    try:
        result = run_recommendation_workflow(payload.user_input)
    except Exception as exc:
        logger.exception("Recommendation workflow failed")
        raise HTTPException(status_code=500, detail=str(exc))

    recs = result.get("final_recommendations", {})

    def _parse_items(raw: list, kind: str) -> list[RecommendationItem]:
        items = []
        for r in raw:
            if isinstance(r, dict):
                items.append(RecommendationItem(
                    name=r.get("name", ""),
                    reasoning=r.get("reasoning", ""),
                    cuisine=r.get("cuisine"),
                    price=r.get("price"),
                    difficulty=r.get("difficulty"),
                ))
        return items

    return RecommendationResponse(
        restaurants=_parse_items(recs.get("restaurants", []), "restaurant"),
        recipes=_parse_items(recs.get("recipes", []), "recipe"),
        user_profile=result.get("user_profile", {}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=APIResponse, tags=["Chat"])
def chat(payload: ChatRequest):
    try:
        intent = classify_intent(payload.message)

        if intent == "clarification":
            return APIResponse(data={
                "intent": intent,
                "reply": (
                    "I'm your food recommendation assistant! I can help you with:\n\n"
                    "🍽️ **Restaurant recommendations** – describe your cuisine preferences, "
                    "dietary restrictions, and occasion.\n"
                    "👨‍🍳 **Recipe recommendations** – tell me what you'd like to cook.\n"
                    "🔍 **Semantic search** – find places by vibe, ingredient, or mood.\n\n"
                    "Just describe what you're looking for!"
                ),
            })

        if intent == "database":
            return APIResponse(data={
                "intent": intent,
                "reply": (
                    "To manage the database, use the **Restaurants** tab in the UI "
                    "or call the REST API endpoints directly."
                ),
            })

        if intent in ("restaurant", "recipe", "both"):
            prefs = extract_preferences(payload.message)
            result = run_recommendation_workflow(payload.message)
            recs = result.get("final_recommendations", {})
            return APIResponse(data={
                "intent": intent,
                "preferences": prefs,
                "recommendations": recs,
            })

        return APIResponse(data={
            "intent": intent,
            "reply": "I'm not sure how to help with that. Can you rephrase?",
        })

    except Exception as exc:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  INDEX MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/index/build", response_model=APIResponse, tags=["Admin"])
def trigger_index_build(reset: bool = True):
    try:
        build_index(reset=reset)
        return APIResponse(message="Vector index built successfully")
    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/index/status", response_model=APIResponse, tags=["Admin"])
def index_status():
    ready = index_ready()
    return APIResponse(
        data={"ready": ready},
        message="Index is ready" if ready else "Index is not ready — call POST /api/index/build",
    )


