"""
Multi-agent recommendation workflow.

Six specialized agents run in a structured pipeline:

  Phase 1 (sequential)  → UserProfileAgent
  Phase 2 (sequential)  → RAGRetrieverAgent
  Phase 3 (parallel)    → FoodTrendAgent | FoodStyleAgent | NutritionAgent
  Phase 4 (sequential)  → RecommendationAgent

All agents share a single state dict threaded through each phase.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)

# ─── Agent configs ────────────────────────────────────────────────────────────

AGENT_CONFIGS: dict[str, dict] = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": (
            "You are an expert user-behavior analyst with 10 years of experience in the food industry. "
            "You identify patterns in dining behavior and build rich user profiles."
        ),
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": "You are a data-retrieval specialist with expertise in vector databases and semantic search.",
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": "You are a culinary journalist who has spent 15 years covering food culture across global markets.",
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": "You are a trained chef and culinary anthropologist with expertise in global cuisines.",
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": "You are a registered dietitian with 8 years of clinical experience.",
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final, personalized recommendations.",
        "backstory": (
            "You are a recommendation-systems architect with experience building personalization engines "
            "for major food-delivery platforms and recipe apps."
        ),
    },
}


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _call_agent(agent_key: str, user_message: str) -> str:
    cfg = get_settings()
    config = AGENT_CONFIGS[agent_key]
    system_prompt = (
        f"You are a {config['role']}.\n\n"
        f"Your goal: {config['goal']}\n\n"
        f"Your background: {config['backstory']}\n\n"
        "Respond with structured, actionable output in valid JSON only. "
        "Do NOT wrap in markdown code fences."
    )

    if cfg.openai_api_key:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=cfg.openai_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

def _safe_json(raw: str, fallback: Any = None) -> Any:
    """Parse JSON, stripping accidental markdown fences."""
    cleaned = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for response: %s…", raw[:120])
        return fallback if fallback is not None else {}


# ─── Workflow nodes ───────────────────────────────────────────────────────────

def node_generate_profile(state: dict) -> dict:
    logger.info("[Phase 1] Generating user profile…")
    msg = (
        f"Analyze this user data and create a comprehensive profile:\n\n{state['user_input']}\n\n"
        "Return JSON with keys: favorite_cuisines (list), dietary_restrictions (list), "
        "dining_occasions (list), price_range (string), adventurousness_score (1-10), "
        "flavor_preferences (list), summary (string)."
    )
    raw = _call_agent("user_profile_generator", msg)
    state["user_profile"] = _safe_json(raw, {})
    state["workflow_step"] = "profile_generated"
    logger.info("[Phase 1] Done – profile summary: %s", state["user_profile"].get("summary", "N/A"))
    return state


def node_retrieve_candidates(state: dict) -> dict:
    logger.info("[Phase 2] Retrieving candidates…")
    msg = (
        f"Based on this user profile:\n{json.dumps(state['user_profile'], indent=2)}\n\n"
        "Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.\n\n"
        'Return JSON: {"restaurants": [{"name": str, "cuisine": str, "price": str, '
        '"rating": float, "description": str}], '
        '"recipes": [{"name": str, "cuisine": str, "difficulty": str, '
        '"prep_time": str, "description": str}]}. '
        "Make results realistic and diverse."
    )
    raw = _call_agent("rag_retriever", msg)
    data = _safe_json(raw, {"restaurants": [], "recipes": []})
    state["retrieved_restaurants"] = data.get("restaurants", [])
    state["retrieved_recipes"] = data.get("recipes", [])
    logger.info("[Phase 2] Retrieved %d restaurants, %d recipes",
                len(state["retrieved_restaurants"]), len(state["retrieved_recipes"]))
    state["workflow_step"] = "candidates_retrieved"
    return state


def node_analyze_trends(state: dict) -> dict:
    logger.info("[Phase 3a] Analyzing trends…")
    msg = (
        f"Analyze current food trends in these options:\n\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        'Identify 3-5 relevant trends. Return JSON: {"trends": [{"name": str, "description": str, "relevance": str}]}'
    )
    raw = _call_agent("food_trend_analyst", msg)
    state["trend_analysis"] = _safe_json(raw, {"trends": []})
    return state


def node_analyze_styles(state: dict) -> dict:
    logger.info("[Phase 3b] Analyzing food styles…")
    msg = (
        f"Analyze cuisine types and flavor profiles:\n\n"
        f"User Profile: {json.dumps(state['user_profile'], indent=2)}\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        "Identify cuisine types, dominant flavor profiles, key ingredients/techniques, "
        "and rank items by fit to user preferences.\n"
        'Return JSON: {"cuisine_types": [], "flavor_profiles": [], "ranked_matches": []}'
    )
    raw = _call_agent("food_style_expert", msg)
    state["style_analysis"] = _safe_json(raw, {})
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    logger.info("[Phase 3c] Evaluating nutrition…")
    msg = (
        f"Evaluate nutritional fit:\n\n"
        f"User Profile: {json.dumps(state['user_profile'], indent=2)}\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}\n\n"
        "Check dietary restrictions, allergens, and nutritional balance.\n"
        'Return JSON: {"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}'
    )
    raw = _call_agent("nutrition_expert", msg)
    state["nutrition_analysis"] = _safe_json(raw, {})
    return state


def node_generate_recommendations(state: dict) -> dict:
    logger.info("[Phase 4] Generating final recommendations…")
    msg = (
        f"Synthesize these insights into top-5 restaurant and top-5 recipe recommendations:\n\n"
        f"User Profile: {json.dumps(state['user_profile'], indent=2)}\n"
        f"Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}\n"
        f"Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}\n"
        f"Trends: {json.dumps(state['trend_analysis'], indent=2)}\n"
        f"Styles: {json.dumps(state['style_analysis'], indent=2)}\n"
        f"Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}\n\n"
        'Return JSON: {'
        '"restaurants": [{"name": str, "cuisine": str, "price": str, "reasoning": str}], '
        '"recipes": [{"name": str, "cuisine": str, "difficulty": str, "reasoning": str}]'
        '}. Each reasoning should be 2-3 engaging sentences.'
    )
    raw = _call_agent("recommendation_expert", msg)
    state["final_recommendations"] = _safe_json(raw, {"restaurants": [], "recipes": []})
    state["workflow_step"] = "complete"
    return state


# ─── Public entry point ───────────────────────────────────────────────────────

def run_recommendation_workflow(user_input: str) -> dict:
    """
    Execute the full 4-phase multi-agent workflow and return the final state.

    Returns a dict with:
        user_profile, retrieved_restaurants, retrieved_recipes,
        trend_analysis, style_analysis, nutrition_analysis,
        final_recommendations, workflow_step
    """
    state: dict = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }

    # Phase 1 & 2 – sequential
    state = node_generate_profile(state)
    state = node_retrieve_candidates(state)

    # Phase 3 – parallel
    logger.info("[Phase 3] Running analysis agents in parallel…")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends    = executor.submit(node_analyze_trends,    dict(state))
        future_styles    = executor.submit(node_analyze_styles,    dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        state["trend_analysis"]    = future_trends.result()["trend_analysis"]
        state["style_analysis"]    = future_styles.result()["style_analysis"]
        state["nutrition_analysis"] = future_nutrition.result()["nutrition_analysis"]

    # Phase 4 – sequential
    state = node_generate_recommendations(state)
    return state


# ─── Lightweight intent/preference helpers (used by chat endpoint) ────────────

def classify_intent(message: str) -> str:
    """Classify user message intent using the LLM."""
    cfg = get_settings()
    system = (
        "You are an intent classifier for a food recommendation system.\n"
        "Classify the message as ONE of: restaurant, recipe, both, clarification, database.\n"
        "Respond with ONLY the label."
    )
    if cfg.openai_api_key:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
        )
        intent = resp.choices[0].message.content.strip().lower()

    valid = {"restaurant", "recipe", "both", "clarification", "database"}
    return intent if intent in valid else "clarification"


def extract_preferences(message: str) -> dict:
    """Extract structured user preferences from a natural-language message."""
    cfg = get_settings()
    system = (
        "You are a preference extractor for a food recommendation system.\n"
        "Extract user preferences and return JSON with keys: "
        "favorite_cuisines, dietary_restrictions, dining_occasion, "
        "price_range, flavor_preferences, other_preferences.\n"
        "Respond with ONLY valid JSON."
    )
    if cfg.openai_api_key:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        resp = client.chat.completions.create(
            model=cfg.openai_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
        )
        raw = resp.choices[0].message.content
        
    return _safe_json(raw, {
        "favorite_cuisines": [],
        "dietary_restrictions": [],
        "dining_occasion": "not specified",
        "price_range": "not specified",
        "flavor_preferences": [],
        "other_preferences": "",
    })
