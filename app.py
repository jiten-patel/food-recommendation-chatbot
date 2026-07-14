"""
Application entry point.

Starts the FastAPI + Gradio server with Uvicorn.

The Gradio UI is mounted at  /ui
The FastAPI REST API stays at /api/*
The FastAPI docs stay at     /api/docs
"""
import uvicorn
import gradio as gr
from fastapi import FastAPI

from config import get_settings


def create_app() -> FastAPI:
    """Build and return the combined FastAPI + Gradio ASGI app."""
    # Import the FastAPI app (all /api/* routes already registered)
    from api import app as fastapi_app

    # Import and build the Gradio Blocks app
    from gradio_ui import build_gradio_app
    gradio_app = build_gradio_app()

    # Mount Gradio at / — FastAPI handles everything else
    fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/")

    return fastapi_app


def app():
    cfg = get_settings()
    uvicorn.run(
        "app:create_app",
        factory=True,
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.api_reload,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    app()
