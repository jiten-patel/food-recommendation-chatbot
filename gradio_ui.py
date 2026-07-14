"""
Gradio UI for the AI-Powered Food Recommendation System.

Tabs
────
1. 💬 Chat          – conversational food recommendations
2. 🍽️ Recommend     – structured multi-agent recommendation workflow
3. 🔍 Search        – multimodal semantic search
4. 🏪 Restaurants   – full CRUD management
5. ⚙️  Admin         – vector index management
"""
from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr

from services import restaurant_service, retrieval_service
from services.agents import (
    classify_intent,
    extract_preferences,
    run_recommendation_workflow,
)
from services.vector_index import build_index, index_ready
from config import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()
# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_json(obj: Any) -> str:
    """Pretty-print any object as indented JSON string."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _restaurant_table(data: list[dict]) -> list[list]:
    """Convert restaurant dicts to rows for a Gradio Dataframe."""
    rows = []
    for r in data:
        rows.append([
            r.get("itemId", ""),
            r.get("name", ""),
            r.get("location", ""),
            r.get("food_style", ""),
            r.get("type", ""),
            r.get("rating", ""),
            r.get("price_range", ""),
            r.get("vibe", ""),
        ])
    return rows


_RESTAURANT_HEADERS = ["ID", "Name", "Location", "Cuisine", "Type", "Rating", "Price", "Vibe"]


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def chat_respond(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Gradio chatbot handler — returns (empty_input, updated_history)."""
    if not message.strip():
        return "", history
    
    if not cfg.openai_api_key:
        history = history + [{"role": "assistant", "content": "Requires valid OpenAI API key"}]
        return "", history
    try:
        intent = classify_intent(message)

        if intent == "clarification":
            reply = (
                "I'm your food recommendation assistant! I can help you with:\n\n"
                "🍽️ **Restaurant recommendations** — describe your preferences, "
                "dietary needs, and occasion.\n"
                "👨‍🍳 **Recipe recommendations** — tell me what you'd like to cook.\n"
                "🔍 **Semantic search** — find places by vibe, ingredient, or mood.\n\n"
                "Just describe what you're looking for!"
            )

        elif intent == "database":
            reply = (
                "To manage the restaurant database, switch to the **🏪 Restaurants** tab "
                "where you can browse, add, edit, or delete entries."
            )

        elif intent in ("restaurant", "recipe", "both"):
            prefs = extract_preferences(message)
            result = run_recommendation_workflow(message)
            recs = result.get("final_recommendations", {})

            lines = [f"**Intent detected:** `{intent}`\n"]

            restaurants = recs.get("restaurants", [])
            if restaurants:
                lines.append("### 🍽️ Restaurant Recommendations")
                for i, r in enumerate(restaurants, 1):
                    lines.append(
                        f"**{i}. {r.get('name', 'Unknown')}** ({r.get('cuisine', '')})\n"
                        f"*Price:* {r.get('price', 'N/A')} · "
                        f"{r.get('reasoning', '')}\n"
                    )

            recipes = recs.get("recipes", [])
            if recipes:
                lines.append("### 👨‍🍳 Recipe Recommendations")
                for i, r in enumerate(recipes, 1):
                    lines.append(
                        f"**{i}. {r.get('name', 'Unknown')}** ({r.get('cuisine', '')})\n"
                        f"*Difficulty:* {r.get('difficulty', 'N/A')} · "
                        f"{r.get('reasoning', '')}\n"
                    )

            reply = "\n".join(lines) if len(lines) > 1 else "No recommendations generated."

        else:
            reply = "I'm not sure how to help with that. Could you rephrase?"

    except Exception as exc:
        logger.exception("Chat error")
        reply = f"⚠️ Error: {exc}"

    history = history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", history


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – RECOMMEND
# ═══════════════════════════════════════════════════════════════════════════════

def run_recommend(user_input: str, rec_type: str) -> tuple[str, str, str]:
    """
    Returns (restaurants_md, recipes_md, profile_json).
    """
    if not user_input.strip():
        return "Please enter a preference description.", "", ""
    if not cfg.openai_api_key:
        return("Please Enter OpenAI API key")
    try:
        result = run_recommendation_workflow(user_input)
    except Exception as exc:
        logger.exception("Recommendation failed")
        return f"⚠️ Error: {exc}", "", ""

    recs = result.get("final_recommendations", {})
    profile = result.get("user_profile", {})

    # Build restaurant markdown
    rest_lines = []
    for i, r in enumerate(recs.get("restaurants", []), 1):
        rest_lines.append(
            f"### {i}. {r.get('name', 'Unknown')}\n"
            f"**Cuisine:** {r.get('cuisine', 'N/A')}  \n"
            f"**Price:** {r.get('price', 'N/A')}  \n"
            f"**Why:** {r.get('reasoning', '')}\n"
        )

    # Build recipe markdown
    recipe_lines = []
    for i, r in enumerate(recs.get("recipes", []), 1):
        recipe_lines.append(
            f"### {i}. {r.get('name', 'Unknown')}\n"
            f"**Cuisine:** {r.get('cuisine', 'N/A')}  \n"
            f"**Difficulty:** {r.get('difficulty', 'N/A')}  \n"
            f"**Why:** {r.get('reasoning', '')}\n"
        )

    restaurants_md = "\n".join(rest_lines) or "No restaurant recommendations."
    recipes_md = "\n".join(recipe_lines) or "No recipe recommendations."
    profile_json = _fmt_json(profile)

    return restaurants_md, recipes_md, profile_json


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def run_search(query: str, k: int, w_text: float, w_img: float, location: str) -> str:
    if not query.strip():
        return "Please enter a search query."

    if not index_ready():
        return (
            "⚠️ Vector index is not ready.\n\n"
            "Go to the **⚙️ Admin** tab and click **Build Index** first."
        )
    
    if not cfg.openai_api_key:
        return("Please Enter OpenAI API key")

    try:
        where_text = {"location": location} if location.strip() else None
        rows = retrieval_service.fuse_rank(
            query=query,
            k_text=k,
            k_img=k,
            w_text=w_text,
            w_img=w_img,
            where_text=where_text,
            top_n=k,
        )
    except Exception as exc:
        logger.exception("Search failed")
        return f"⚠️ Error: {exc}"

    if not rows:
        return "No results found."

    lines = [f"**{len(rows)} result(s) for:** `{query}`\n"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"**{i}. [{r['modality'].upper()}]** Score: `{r['fused_score']:.4f}`  \n"
            f"{r.get('snippet', '')}  \n"
            f"*Cuisine:* {r.get('cuisine', 'N/A')} · "
            f"*Location:* {r.get('location', 'N/A')}\n"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – RESTAURANTS (CRUD)
# ═══════════════════════════════════════════════════════════════════════════════

def load_restaurant_table() -> list[list]:
    return _restaurant_table(restaurant_service.get_all_restaurants())


def refresh_table() -> list[list]:
    return load_restaurant_table()


def add_restaurant(paragraph: str) -> tuple[str, list[list]]:
    if not paragraph.strip():
        return "⚠️ Please enter a restaurant description.", load_restaurant_table()
    if not cfg.openai_api_key:
        return("Please Enter OpenAI API key")
    try:
        r = restaurant_service.add_restaurant(paragraph)
        msg = f"✅ Added **{r.get('name')}** (ID: {r.get('itemId')})"
    except Exception as exc:
        logger.exception("Add restaurant failed")
        msg = f"⚠️ Error: {exc}"
    return msg, load_restaurant_table()


def update_restaurant(item_id_str: str, paragraph: str) -> tuple[str, list[list]]:
    if not item_id_str.strip() or not paragraph.strip():
        return "⚠️ Please provide both an ID and a description.", load_restaurant_table()
    if not cfg.openai_api_key:
        return(" Please Enter OpenAI API key")
    try:
        item_id = int(item_id_str)
        r = restaurant_service.update_restaurant(item_id, paragraph)
        if r is None:
            msg = f"⚠️ Restaurant with ID {item_id} not found."
        else:
            msg = f"✅ Updated **{r.get('name')}** (ID: {item_id})"
    except ValueError:
        msg = "⚠️ ID must be a number."
    except Exception as exc:
        logger.exception("Update restaurant failed")
        msg = f"⚠️ Error: {exc}"
    return msg, load_restaurant_table()


def delete_restaurant(item_id_str: str) -> tuple[str, list[list]]:
    if not item_id_str.strip():
        return "⚠️ Please provide a restaurant ID.", load_restaurant_table()
    try:
        item_id = int(item_id_str)
        deleted = restaurant_service.delete_restaurant(item_id)
        msg = f"✅ Deleted restaurant ID {item_id}" if deleted else f"⚠️ ID {item_id} not found."
    except ValueError:
        msg = "⚠️ ID must be a number."
    except Exception as exc:
        logger.exception("Delete restaurant failed")
        msg = f"⚠️ Error: {exc}"
    return msg, load_restaurant_table()


def get_restaurant_detail(item_id_str: str) -> str:
    if not item_id_str.strip():
        return "⚠️ Please enter a restaurant ID."
    try:
        item_id = int(item_id_str)
        r = restaurant_service.get_restaurant_by_id(item_id)
        return _fmt_json(r) if r else f"Restaurant ID {item_id} not found."
    except ValueError:
        return "⚠️ ID must be a number."


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

def check_index_status() -> str:
    ready = index_ready()
    if ready:
        return "✅ **Index is ready.** Vector search is available."
    return "⚠️ **Index is NOT ready.** Click **Build Index** to create it."


def trigger_build_index(reset: bool) -> str:
    try:
        build_index(reset=reset)
        return "✅ Vector index built successfully."
    except Exception as exc:
        logger.exception("Index build failed")
        return f"⚠️ Build failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title="🍴 AI Food Recommendation System",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
        css="""
        .tab-header { font-size: 1.1rem; font-weight: 600; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🍴 AI-Powered Food Recommendation System
            Multi-agent LLM pipeline · Multimodal vector search · Full restaurant CRUD
            """
        )

        # ── Tab 1: Chat ──────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Conversational Food Assistant")
            chatbot = gr.Chatbot(
                label="Food Assistant",
                height=480
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="e.g. I want spicy Asian food near downtown for a date night…",
                    label="Your message",
                    scale=5,
                )
                chat_send = gr.Button("Send ➤", variant="primary", scale=1)
            chat_clear = gr.Button("🗑️ Clear conversation", size="m")

            chat_send.click(
                fn=chat_respond,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot],
            )
            chat_input.submit(
                fn=chat_respond,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot],
            )
            chat_clear.click(lambda: ([], ""), outputs=[chatbot, chat_input])

        # ── Tab 2: Recommend ─────────────────────────────────────────────────
        with gr.Tab("🍽️ Recommend"):
            gr.Markdown("### Multi-Agent Recommendation Workflow")
            with gr.Row():
                rec_input = gr.Textbox(
                    label="Describe your preferences",
                    placeholder=(
                        "e.g. I love bold flavours, vegetarian-friendly, "
                        "budget-conscious, looking for a cozy dinner experience…"
                    ),
                    lines=3,
                    scale=4,
                )
                rec_type = gr.Radio(
                    choices=["both", "restaurant", "recipe"],
                    value="both",
                    label="Recommendation type",
                    scale=1,
                )
            rec_btn = gr.Button("🚀 Get Recommendations", variant="primary")

            with gr.Row():
                rec_restaurants = gr.Markdown(label="🏪 Restaurants")
                rec_recipes = gr.Markdown(label="👨‍🍳 Recipes")
            rec_profile = gr.Code(
                label="📋 User profile (JSON)", language="json", lines=12
            )

            rec_btn.click(
                fn=run_recommend,
                inputs=[rec_input, rec_type],
                outputs=[rec_restaurants, rec_recipes, rec_profile],
            )

        # ── Tab 3: Search ─────────────────────────────────────────────────────
        with gr.Tab("🔍 Semantic Search"):
            gr.Markdown("### Multimodal Vector Search (text + image embeddings)")
            search_query = gr.Textbox(
                label="Search query",
                placeholder="e.g. cozy ramen place with rich broth…",
            )
            with gr.Row():
                search_k = gr.Slider(1, 20, value=5, step=1, label="Top-K results")
                search_w_text = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Text weight")
                search_w_img = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Image weight")
                search_location = gr.Textbox(label="Location filter (optional)", placeholder="e.g. New York")
            search_btn = gr.Button("🔍 Search", variant="primary")
            search_results = gr.Markdown(label="Results")

            search_btn.click(
                fn=run_search,
                inputs=[search_query, search_k, search_w_text, search_w_img, search_location],
                outputs=search_results,
            )

        # ── Tab 4: Restaurants ────────────────────────────────────────────────
        with gr.Tab("🏪 Restaurants"):
            gr.Markdown("### Restaurant Database Management")

            with gr.Row():
                rest_refresh_btn = gr.Button("🔄 Refresh table", size="sm")

            rest_table = gr.Dataframe(
                headers=_RESTAURANT_HEADERS,
                value=load_restaurant_table,
                interactive=False,
                label="All Restaurants",
                wrap=True,
            )

            with gr.Accordion("➕ Add Restaurant", open=False):
                add_para = gr.Textbox(
                    label="Free-text description",
                    placeholder=(
                        "e.g. Miso Kitchen in Brooklyn is a cozy Japanese izakaya "
                        "known for their house ramen. Rating 4.5/5, price $$."
                    ),
                    lines=4,
                )
                add_btn = gr.Button("Add", variant="primary")
                add_status = gr.Markdown()

            with gr.Accordion("✏️ Update Restaurant", open=False):
                upd_id = gr.Textbox(label="Restaurant ID", placeholder="e.g. 1000001")
                upd_para = gr.Textbox(
                    label="Updated description", lines=4,
                    placeholder="Updated restaurant details…"
                )
                upd_btn = gr.Button("Update", variant="primary")
                upd_status = gr.Markdown()

            with gr.Accordion("🗑️ Delete Restaurant", open=False):
                del_id = gr.Textbox(label="Restaurant ID", placeholder="e.g. 1000001")
                del_btn = gr.Button("Delete", variant="stop")
                del_status = gr.Markdown()

            with gr.Accordion("🔎 View Details", open=False):
                detail_id = gr.Textbox(label="Restaurant ID", placeholder="e.g. 1000001")
                detail_btn = gr.Button("Get Details")
                detail_out = gr.Code(label="Restaurant JSON", language="json", lines=15)

            # Wire up
            rest_refresh_btn.click(fn=refresh_table, outputs=rest_table)

            add_btn.click(
                fn=add_restaurant,
                inputs=add_para,
                outputs=[add_status, rest_table],
            )
            upd_btn.click(
                fn=update_restaurant,
                inputs=[upd_id, upd_para],
                outputs=[upd_status, rest_table],
            )
            del_btn.click(
                fn=delete_restaurant,
                inputs=del_id,
                outputs=[del_status, rest_table],
            )
            detail_btn.click(
                fn=get_restaurant_detail,
                inputs=detail_id,
                outputs=detail_out,
            )

        # ── Tab 5: Admin ──────────────────────────────────────────────────────
        with gr.Tab("⚙️ Admin"):
            gr.Markdown("### Vector Index Management")

            with gr.Row():
                idx_status_btn = gr.Button("🔍 Check Index Status", variant="secondary")
                idx_reset_chk = gr.Checkbox(label="Reset (wipe existing index)", value=True)
                idx_build_btn = gr.Button("🔨 Build Index", variant="primary")

            idx_status_out = gr.Markdown(value=check_index_status)
            idx_build_out = gr.Markdown()

            idx_status_btn.click(fn=check_index_status, outputs=idx_status_out)
            idx_build_btn.click(
                fn=trigger_build_index,
                inputs=idx_reset_chk,
                outputs=idx_build_out,
            )

    return demo
