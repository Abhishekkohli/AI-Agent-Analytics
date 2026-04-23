# AI-Developer-Workflow-Copilot

An LLM-powered analytics agent that converts natural language business queries into SQL, executes them against a structured dataset, and returns formatted results. Uses embedding-based retrieval (FAISS + sentence-transformers) to supply relevant schema context and few-shot query history to the LLM, improving SQL generation accuracy.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│   Context Manager               │
│  ┌────────────┐ ┌─────────────┐ │
│  │ Schema     │ │ Query       │ │
│  │ Retrieval  │ │ History     │ │
│  │ (FAISS)    │ │ Retrieval   │ │
│  └─────┬──────┘ └──────┬──────┘ │
│        └───────┬───────┘        │
└────────────────┼────────────────┘
                 ▼
         ┌──────────────┐
         │   LLM (GPT)  │
         │  SQL Gen      │
         └──────┬───────┘
                ▼
         ┌──────────────┐
         │   SQLite DB   │
         │  Execute SQL  │
         └──────┬───────┘
                ▼
         Formatted Results
         (+ feedback loop → vector store)
```

## Project Structure

```
ai_agent_analytics/
├── app.py                # Interactive CLI entry point
├── sql_agent.py          # Core NL→SQL agent (LLM + execution)
├── context_manager.py    # Retrieval-augmented prompt builder
├── vector_store.py       # FAISS vector store + embedding logic
├── setup_database.py     # SQLite database with sample e-commerce data
├── evaluate.py           # Evaluation harness (200+ test queries)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# 4. Initialise database + vector store
python app.py --setup
```

## Usage

### Interactive Mode

```bash
python app.py
```

Ask questions in plain English:

```
📊 Ask > What are the top 5 products by revenue?
📊 Ask > Show me monthly order trends for 2025
📊 Ask > Which customers have spent more than $500?
📊 Ask > What is the cancellation rate by state?
```

### Evaluation

```bash
# Run on all 200+ test queries
python evaluate.py

# Run on first 20 queries (quick test)
python app.py --eval 20
```

### Options

```bash
python app.py --model gpt-4o       # Use a different model
python app.py --verbose             # Show context retrieval debug info
```

## Key Components

### Vector Store (`vector_store.py`)
- Uses `sentence-transformers/all-MiniLM-L6-v2` for local embeddings (no API cost)
- FAISS `IndexFlatIP` for fast cosine-similarity search
- Indexes two document types: **schema descriptions** (auto-introspected from SQLite) and **query history** (question→SQL pairs)
- Persists to disk for fast subsequent loads

### Context Manager (`context_manager.py`)
- Retrieves top-k most relevant schemas and few-shot examples for each query
- Assembles a structured system prompt with schema context + similar examples
- Runtime feedback loop: successful queries get added back to the store

### SQL Agent (`sql_agent.py`)
- Calls OpenAI API with retrieved context
- Cleans LLM output (strips markdown fences, etc.)
- Executes SQL against SQLite and returns pandas DataFrames
- Auto-feeds successful pairs back into vector store

### Evaluation (`evaluate.py`)
- 200+ test queries covering aggregation, joins, filtering, date queries, subqueries
- Measures execution accuracy (SQL runs and returns results) and fragment accuracy
- Supports comparison between "with retrieval" vs "without retrieval" modes

## Sample Database

The SQLite database (`business.db`) contains:
- **6 categories** across 4 departments
- **31 products** with prices and stock levels
- **60 customers** across 12 US cities
- **400 orders** with multiple statuses
- **~1000 order items** with line totals
- **300 product reviews** with 1-5 star ratings
