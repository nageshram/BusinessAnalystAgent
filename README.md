# Business Analyst Agent

> AI-powered business analyst that ingests CSV/XLSX business data, profiles datasets, automates EDA, generates KPI insights, and answers business questions via tool-calling agents.

**Python · FastAPI · LangGraph · RAG · PostgreSQL · Vector Search · BM25**

---

## Features

| Feature | Description |
|:--------|:------------|
| 📊 **Dataset Profiling** | Auto-profile uploaded datasets — types, nulls, distributions, semantic classification |
| 📈 **KPI Computation** | Sum, avg, growth rates, grouped metrics (e.g., revenue by region) |
| 📉 **Trend Detection** | Linear regression, period-over-period comparison, moving averages |
| 🔍 **Anomaly Detection** | IQR and Z-score methods for outlier identification |
| 🔗 **Correlation Analysis** | Pearson/Spearman correlations with strength classification |
| 📊 **Chart Suggestions** | Smart chart type recommendations with axis mappings |
| 📄 **RAG Knowledge Base** | Upload business docs, hybrid search (vector + BM25 + RRF re-ranking) |
| 🤖 **Conversational Q&A** | Natural language questions answered via LangGraph tool-calling agent |

---

## Architecture

```
Query → LangGraph Agent
           ├→ profile_dataset()      → Profiler Engine
           ├→ query_data()           → Pandas Operations
           ├→ compute_metrics()      → Metrics Engine
           ├→ detect_trends()        → Trend Engine (scipy)
           ├→ detect_anomalies()     → Anomaly Engine (IQR/Z-score)
           ├→ compute_correlations() → Correlation Engine
           ├→ suggest_charts()       → Chart Engine
           └→ search_knowledge_base()→ Hybrid RAG (Vector + BM25 + RRF)
```

## Production Features

- **LangGraph StateGraph** with message trimming, step guards, recursion limits
- **Circuit Breaker Pattern** — prevents cascading failures on LLM/DB outages
- **PostgreSQL Checkpointing** — durable agent state across restarts
- **BM25 + Vector Hybrid Retrieval** with Reciprocal Rank Fusion re-ranking
- **Rate Limiting** — per-IP rate limiting middleware
- **Health Checks** — simple and detailed endpoints for monitoring

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- PostgreSQL 16+
- [OpenRouter API Key](https://openrouter.ai/keys)

### 2. Setup

```bash
# Clone and enter the project
cd BusinessAnalystAgent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Database Setup

```bash
# Start PostgreSQL (via Docker)
docker compose up -d postgres

# Or use existing PostgreSQL and update DATABASE_URL in .env
```

### 4. Run

```bash
# Development
uvicorn app.main:app --reload --port 8200

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8200 --workers 4
```

### 5. Docker (Full Stack)

```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f app
```

---

## API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/api/chat` | Chat (sync response) |
| `POST` | `/api/chat/stream` | Chat (SSE streaming) |
| `POST` | `/api/upload/dataset` | Upload CSV/XLSX dataset |
| `POST` | `/api/upload/document` | Upload document for RAG knowledge base |
| `GET` | `/api/datasets` | List uploaded datasets |
| `GET` | `/api/datasets/{id}` | Dataset details + profile |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{thread_id}/messages` | Message history |
| `GET` | `/health` | Health check |
| `GET` | `/health/detailed` | Detailed health check |

Interactive API docs available at: `http://localhost:8200/docs`

---

## Example Usage

### Upload & Analyze

```bash
# Upload a dataset
curl -X POST "http://localhost:8200/api/upload/dataset?thread_id=demo-1" \
  -F "file=@sales_data.csv"

# Ask a question
curl -X POST "http://localhost:8200/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Profile this dataset", "thread_id": "demo-1", "dataset_id": "abc12345"}'

# Follow up
curl -X POST "http://localhost:8200/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top KPIs? Show me revenue trends", "thread_id": "demo-1", "dataset_id": "abc12345"}'
```

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **API** | FastAPI + Uvicorn |
| **Agent** | LangGraph StateGraph |
| **LLM** | GPT-4o-mini via OpenRouter |
| **Database** | PostgreSQL + SQLAlchemy |
| **Vector Store** | ChromaDB |
| **Keyword Search** | BM25 (rank-bm25) |
| **Re-ranking** | Reciprocal Rank Fusion (RRF) |
| **Data Analysis** | Pandas + NumPy + SciPy |
| **Deployment** | Docker + Docker Compose |

---

## Project Structure

```
BusinessAnalystAgent/
├── app/
│   ├── main.py              # FastAPI entry point + lifecycle
│   ├── config.py            # Pydantic settings
│   ├── circuit_breaker.py   # Thread-safe circuit breaker
│   ├── api/                 # Routes + middleware
│   ├── agent/               # LangGraph agent + tools + prompts
│   ├── db/                  # PostgreSQL models + CRUD
│   ├── analysis/            # 6 analysis engines
│   ├── data/                # Dataset management
│   └── rag/                 # Hybrid retrieval (Vector + BM25)
├── tests/                   # Pytest test suite
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # PostgreSQL + App
└── requirements.txt         # Python dependencies
```

---

## License

MIT
