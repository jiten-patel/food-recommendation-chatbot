# AI-Powered Multimodal Food Recommendation System

A production-ready application that combines a **6-agent AI orchestration pipeline**, **multimodal vector search** (text + images via CLIP), a **Gradio browser UI**, and a **FastAPI REST API** for restaurant and recipe recommendations.

---

## Architecture

```
foodrecomendation/
│   ├── api.py                    ← FastAPI REST endpoints (/api/*)
│   ├── config.py                 ← Pydantic-settings config (reads .env)
│   ├── models.py                 ← Pydantic request/response schemas
│   ├── app.py                   ← Uvicorn entry point; mounts Gradio at /
│   ├── gradio_ui.py              ← 5-tab Gradio UI (Chat, Recommend, Search, Restaurants, Admin)
│   ├── mcp_server.py             ← FastMCP server (3 tools + 1 resource)
│   └── services/
│       ├── agents.py             ← 6-agent workflow (4 phases)
│       ├── restaurant_service.py ← CRUD + LLM paragraph extraction with self-healing
│       ├── retrieval_service.py  ← Multimodal fusion ranking (text + image)
│       └── vector_index.py       ← ChromaDB index builder (SentenceTransformer + CLIP)
│
├── data/                         ← JSON datasets
│   ├── structured_restaurant_data.json
│   ├── augmented_food_recipe.json
│   ├── augmented_user_review.json
│   ├── Synthetic-User-Reviews.json
│   └── recipe_images/            ← PNG recipe images for CLIP index
│
├── .env.example                  ← Environment variable template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Clone and set up the environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — minimum required: OPENAI_API_KEY
```

### 3. Start the server

```bash
python -m backend.main
```

| Endpoint | URL |
|----------|-----|
| Gradio UI | `http://localhost:3000/` |
| REST API docs | `http://localhost:3000/api/docs` |
| Health check | `http://localhost:3000/api/health` |

> **Default port is 3000.** Override with `API_PORT=8000` in `.env`.

### 4. Build the vector index (enables semantic search)

Via the **⚙️ Admin** tab in the UI, or via the API:

```bash
curl -X POST http://localhost:3000/api/index/build
```

---

## Gradio UI — 5 Tabs

The Gradio interface is mounted at `/` and served alongside the REST API.

| Tab | Description |
|-----|-------------|
| 💬 **Chat** | Conversational food assistant — intent-aware, runs the full agent pipeline for recommendation requests |
| 🍽️ **Recommend** | Structured multi-agent recommendation form — returns top-5 restaurants + top-5 recipes with reasoning |
| 🔍 **Semantic Search** | Multimodal vector search with adjustable Top-K, text/image weight sliders, and optional location filter |
| 🏪 **Restaurants** | Full CRUD management — browse, add (free-text paragraph), update, delete, and inspect individual records |
| ⚙️ **Admin** | Vector index management — check status, build, or reset the ChromaDB index |

---

## Multi-Agent Recommendation Workflow

The core recommendation engine uses 6 specialized LLM agents across 4 phases:

```
User Input
   │
   ▼ Phase 1 — sequential
UserProfileAgent ─────────→ Structured user profile
   │                        (cuisines, dietary restrictions, price range,
   │                         adventurousness score, flavor preferences)
   ▼ Phase 2 — sequential
RAGRetrieverAgent ────────→ Top-20 restaurants + top-20 recipes
   │
   ├─ Phase 3 — parallel (ThreadPoolExecutor, 3 workers) ──────────────────┐
   │  FoodTrendAgent          FoodStyleAgent          NutritionAgent       │
   │  (emerging trends)       (cuisine + flavor fit)  (dietary compliance) │
   └────────────────────────────────────────────────────────────────────────┘
   │
   ▼ Phase 4 — sequential
RecommendationAgent ──────→ Top-5 restaurants + top-5 recipes (with reasoning)
```

### Agent Roles

| Agent | Role |
|-------|------|
| `UserProfileAgent` | Builds a structured profile from natural-language user input |
| `RAGRetrieverAgent` | Simulates vector-DB retrieval based on the user profile |
| `FoodTrendAgent` | Identifies 3–5 current food trends relevant to the candidates |
| `FoodStyleAgent` | Matches cuisine types and flavor profiles to user preferences |
| `NutritionAgent` | Flags allergens, dietary violations, and nutritional highlights |
| `RecommendationAgent` | Synthesizes all analysis into final personalized recommendations |

All agents call OpenAI (`gpt-4.1-nano` by default) and fall back to IBM WatsonX Granite if no OpenAI key is set. Responses are parsed as structured JSON with automatic self-healing retries.

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/restaurants` | List all restaurants |
| `GET` | `/api/restaurants/{id}` | Get single restaurant by ID |
| `POST` | `/api/restaurants` | Add restaurant from free-text paragraph |
| `PUT` | `/api/restaurants/{id}` | Update restaurant from free-text paragraph |
| `DELETE` | `/api/restaurants/{id}` | Delete restaurant |
| `GET` | `/api/recipes` | List all recipes |
| `POST` | `/api/search` | Multimodal semantic search (text + image fusion) |
| `POST` | `/api/recommend` | Run full 6-agent recommendation workflow |
| `POST` | `/api/chat` | Conversational chat with intent classification |
| `POST` | `/api/index/build` | (Admin) Rebuild vector index |
| `GET` | `/api/index/status` | Check index readiness |

Interactive docs: `http://localhost:3000/api/docs`

### Example: recommend

```bash
curl -X POST http://localhost:3000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_input": "I love bold spicy food, vegetarian-friendly, budget-conscious"}'
```

### Example: semantic search

```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cozy ramen with rich broth", "k": 5, "w_text": 0.6, "w_img": 0.4}'
```

---

## Multimodal Vector Search

The search engine fuses two ChromaDB collections:

| Collection | Embedding model | Dimension | Source data |
|------------|-----------------|-----------|-------------|
| `restaurant_articles` | `all-MiniLM-L6-v2` (SentenceTransformers) | 384-d | `structured_restaurant_data.json` |
| `food_images` | `openai/clip-vit-base-patch32` (CLIP) | 512-d | `recipe_images/*.png` paired with recipes |

**Fusion formula:** `fused_score = w_text × norm(text_score) + w_img × norm(image_score)`

Default weights: text `0.6`, image `0.4` (adjustable per request).

---

## LLM-Powered Restaurant CRUD

Adding or updating a restaurant accepts a **free-text paragraph** — no JSON required.  
The LLM extracts the structured fields automatically, with up to **3 self-healing retries** on parse failure.

**Extracted fields:** `name`, `location`, `type`, `food_style`, `rating`, `price_range` (1–4), `signatures`, `vibe`, `environment`, `shortcomings`

```bash
curl -X POST http://localhost:3000/api/restaurants \
  -H "Content-Type: application/json" \
  -d '{"paragraph": "Miso Kitchen in Brooklyn is a cozy Japanese izakaya known for house ramen. Rating 4.5/5, price $$."}'
```

---

## MCP Server

A standalone FastMCP server for tool-calling agents:

```bash
python -m backend.mcp_server
```

### Tools

| Tool | Description |
|------|-------------|
| `get_restaurant_info` | Search for a restaurant by name — returns structured details (cuisine, rating, price, signature dishes) |
| `recommend_by_vibe` | Find restaurants by atmosphere/vibe keyword (e.g. `"romantic"`, `"moody"`, `"sun-drenched"`) |
| `get_review` | Retrieve a user review including rating, text, visit date, and image description |

### Resource

| Resource URI | Description |
|-------------|-------------|
| `culinary-map://california` | Full California Culinary Map text — 100+ restaurant descriptions |

---

## Chat Intent Classification

The `/api/chat` endpoint and the Chat UI tab classify each message into one of five intents before routing:

| Intent | Behaviour |
|--------|-----------|
| `restaurant` | Runs the full agent pipeline, returns restaurant recommendations |
| `recipe` | Runs the full agent pipeline, returns recipe recommendations |
| `both` | Runs the full agent pipeline, returns both |
| `clarification` | Returns a help message describing available capabilities |
| `database` | Redirects the user to the Restaurants management tab/endpoints |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-nano` | OpenAI model to use |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature |
| `ANTHROPIC_API_KEY` | `""` | Anthropic key (MCP sampling) |
| `API_HOST` | `localhost` | Server bind host |
| `API_PORT` | `3000` | Server bind port |
| `API_RELOAD` | `False` | Enable Uvicorn hot-reload |
| `CHROMA_PERSIST_DIR` | `.chroma_db` | ChromaDB persistence directory |
| `DATA_DIR` | `data/` | Directory for JSON data files |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Docker

```bash
# Build and run (exposes port 8000)
docker compose up --build

# View logs
docker compose logs -f
```

The compose file mounts `./data` and `./.chroma_db` as volumes so data and the vector index persist between container restarts.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI + Uvicorn |
| Browser UI | Gradio 4 (5-tab Blocks app, mounted at `/`) |
| LLM — primary | OpenAI `gpt-4.1-nano` |
| LLM — fallback | IBM WatsonX `ibm/granite-4-h-small` |
| Agent orchestration | Custom 4-phase pipeline with `ThreadPoolExecutor` |
| Vector DB | ChromaDB (persisted) via LangChain Chroma |
| Text embeddings | `all-MiniLM-L6-v2` (SentenceTransformers, 384-d) |
| Image embeddings | CLIP `openai/clip-vit-base-patch32` (512-d) |
| MCP | FastMCP |
| Config | pydantic-settings |
| Containerisation | Docker + docker-compose |
