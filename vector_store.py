"""
FAISS-backed vector store for embedding-based retrieval.
Indexes two types of documents:
  1. Schema descriptions  – table/column metadata for the database
  2. Query history        – past (question, SQL) pairs for few-shot context

Uses sentence-transformers for local embedding (no API key needed).
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

STORE_DIR = os.path.join(os.path.dirname(__file__), ".vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatIP | None = None  # inner-product (cosine after norm)
        self.documents: list[dict] = []

    # ── Build ──────────────────────────────────────────────────────

    def add_documents(self, docs: list[dict]) -> None:
        """
        Each doc dict should have:
          - 'text': string to embed
          - 'type': 'schema' | 'history'
          - 'metadata': arbitrary dict (table name, SQL, etc.)
        """
        self.documents.extend(docs)

    def build_index(self) -> None:
        """Encode all documents and build the FAISS index."""
        texts = [d["text"] for d in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    # ── Retrieve ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
    ) -> list[dict]:
        """
        Return the top-k most relevant documents for `query`.
        Optionally filter by doc_type ('schema' or 'history').
        """
        if self.index is None:
            raise RuntimeError("Index not built — call build_index() first")

        q_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")

        # Retrieve more than top_k so filtering by type still yields enough results
        fetch_k = min(top_k * 3, self.index.ntotal)
        scores, ids = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if doc_type and doc["type"] != doc_type:
                continue
            results.append({**doc, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    # ── Persist / Load ─────────────────────────────────────────────

    def save(self, directory: str = STORE_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump(self.documents, f, indent=2)

    def load(self, directory: str = STORE_DIR) -> None:
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "documents.json")) as f:
            self.documents = json.load(f)


# ── Helper: build schema docs from a live SQLite connection ────────

def schema_docs_from_db(db_path: str) -> list[dict]:
    """
    Introspect the SQLite database and produce one document per table
    describing its columns, types, and foreign keys.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    docs = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = cur.fetchall()
        cur.execute(f"PRAGMA foreign_key_list({table})")
        fks = cur.fetchall()

        col_lines = []
        for _, name, ctype, notnull, default, pk in cols:
            parts = [f"{name} {ctype}"]
            if pk:
                parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")
            col_lines.append(", ".join(parts))

        fk_lines = []
        for fk in fks:
            fk_lines.append(f"  FK: {fk[3]} -> {fk[2]}({fk[4]})")

        description = f"Table: {table}\nColumns:\n  " + "\n  ".join(col_lines)
        if fk_lines:
            description += "\nForeign keys:\n" + "\n".join(fk_lines)

        # Also add sample rows summary for richer retrieval
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]
        description += f"\nRow count: {row_count}"

        docs.append({
            "text": description,
            "type": "schema",
            "metadata": {"table": table},
        })

    conn.close()
    return docs


# ── Helper: build query-history docs ──────────────────────────────

# Seed history: pairs of (natural language question, SQL) that act as
# few-shot examples and get indexed for retrieval.
SEED_QUERY_HISTORY = [
    {
        "question": "What are the top 5 best-selling products by revenue?",
        "sql": (
            "SELECT p.product_name, SUM(oi.line_total) AS revenue "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 5;"
        ),
    },
    {
        "question": "How many orders were placed each month in 2025?",
        "sql": (
            "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS order_count "
            "FROM orders WHERE order_date >= '2025-01-01' "
            "GROUP BY month ORDER BY month;"
        ),
    },
    {
        "question": "Which customers have spent more than $500 in total?",
        "sql": (
            "SELECT c.first_name || ' ' || c.last_name AS customer, "
            "SUM(o.total_amount) AS total_spent "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY c.customer_id HAVING total_spent > 500 "
            "ORDER BY total_spent DESC;"
        ),
    },
    {
        "question": "What is the average product rating by category?",
        "sql": (
            "SELECT cat.category_name, ROUND(AVG(r.rating), 2) AS avg_rating "
            "FROM reviews r "
            "JOIN products p ON r.product_id = p.product_id "
            "JOIN categories cat ON p.category_id = cat.category_id "
            "GROUP BY cat.category_name ORDER BY avg_rating DESC;"
        ),
    },
    {
        "question": "Show the cancellation rate per month.",
        "sql": (
            "SELECT strftime('%Y-%m', order_date) AS month, "
            "ROUND(100.0 * SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) "
            "/ COUNT(*), 1) AS cancel_pct "
            "FROM orders GROUP BY month ORDER BY month;"
        ),
    },
    {
        "question": "Which state has the most customers?",
        "sql": (
            "SELECT state, COUNT(*) AS customer_count "
            "FROM customers GROUP BY state ORDER BY customer_count DESC LIMIT 1;"
        ),
    },
    {
        "question": "List products that have never been ordered.",
        "sql": (
            "SELECT p.product_name FROM products p "
            "LEFT JOIN order_items oi ON p.product_id = oi.product_id "
            "WHERE oi.item_id IS NULL;"
        ),
    },
    {
        "question": "What is the total revenue by department?",
        "sql": (
            "SELECT cat.department, SUM(oi.line_total) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN categories cat ON p.category_id = cat.category_id "
            "GROUP BY cat.department ORDER BY revenue DESC;"
        ),
    },
    {
        "question": "Who are the top 3 customers by number of orders?",
        "sql": (
            "SELECT c.first_name || ' ' || c.last_name AS customer, "
            "COUNT(o.order_id) AS num_orders "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_id ORDER BY num_orders DESC LIMIT 3;"
        ),
    },
    {
        "question": "What is the average order value for delivered orders?",
        "sql": (
            "SELECT ROUND(AVG(total_amount), 2) AS avg_order_value "
            "FROM orders WHERE status = 'delivered';"
        ),
    },
]


def history_docs() -> list[dict]:
    """Convert seed query history into embeddable documents."""
    docs = []
    for pair in SEED_QUERY_HISTORY:
        text = f"Question: {pair['question']}\nSQL: {pair['sql']}"
        docs.append({
            "text": text,
            "type": "history",
            "metadata": {"question": pair["question"], "sql": pair["sql"]},
        })
    return docs


def build_vector_store(db_path: str) -> VectorStore:
    """One-call helper: build & persist a full vector store."""
    store = VectorStore()
    store.add_documents(schema_docs_from_db(db_path))
    store.add_documents(history_docs())
    store.build_index()
    store.save()
    print(f"Vector store built — {len(store.documents)} documents indexed")
    return store
