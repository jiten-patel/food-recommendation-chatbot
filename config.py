"""
Central configuration for the Food Recommendation backend.
All settings are read from environment variables with safe defaults.
"""
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent  # project root


class Settings(BaseSettings):
    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-nano"
    openai_temperature: float = 0.7

    # ── IBM WatsonX (optional fallback) ────────────────────────────────────
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = ""
    watsonx_model: str = ""

    # ── Anthropic (MCP sampling) ────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ── Data paths ──────────────────────────────────────────────────────────
    data_dir: Path = BASE_DIR / "data"
    restaurant_data_file: str = "structured_restaurant_data.json"
    recipe_data_file: str = "augmented_food_recipe.json"
    user_review_file: str = "augmented_user_review.json"
    recipe_images_dir: str = "recipe_images"

    # ── Vector DB ───────────────────────────────────────────────────────────
    chroma_persist_dir: Path = BASE_DIR / ".chroma_db"
    text_embed_model: str = "all-MiniLM-L6-v2"
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # ── API server ──────────────────────────────────────────────────────────
    api_host: str = "localhost"
    api_port: int = 3000
    api_reload: bool = True
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "*"]

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
