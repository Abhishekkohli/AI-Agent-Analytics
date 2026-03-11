"""
LLM-powered SQL generation agent.

Pipeline:
  user question ─► context_manager (vector retrieval) ─► LLM ─► SQL
  SQL ─► SQLite execution ─► formatted results
  successful pair ─► fed back into vector store for future retrieval
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from context_manager import ContextManager
from vector_store import VectorStore, build_vector_store, schema_docs_from_db
from setup_database import DB_PATH

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


class SQLAgent:
    """End-to-end NL → SQL → results agent."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
    ):
        self.db_path = db_path
        self.model = model
        self.verbose = verbose
        self.client = OpenAI()  # reads OPENAI_API_KEY from env

        # Initialise or load vector store
        store_dir = os.path.join(os.path.dirname(__file__), ".vector_store")
        self.store = VectorStore()
        if os.path.exists(os.path.join(store_dir, "index.faiss")):
            self.store.load(store_dir)
        else:
            self.store = build_vector_store(db_path)

        self.ctx = ContextManager(self.store)

    # ── Public API ─────────────────────────────────────────────────

    def ask(self, question: str) -> dict[str, Any]:
        """
        Full pipeline: question → SQL → execute → result dict.
        Returns:
          {
            "question": str,
            "sql": str,
            "columns": list[str],
            "rows": list[tuple],
            "dataframe": pd.DataFrame,
            "error": str | None,
          }
        """
        messages = self.ctx.build_prompt(question)

        if self.verbose:
            print(f"\n[Context] System prompt length: {len(messages[0]['content'])} chars")

        sql = self._call_llm(messages)
        sql = self._clean_sql(sql)

        if self.verbose:
            print(f"[SQL] {sql}")

        result = self._execute(sql)
        result["question"] = question
        result["sql"] = sql

        # Feed successful queries back into the store for future retrieval
        if result["error"] is None:
            self.ctx.add_to_history(question, sql)

        return result

    def ask_text(self, question: str) -> str:
        """Convenience wrapper that returns a formatted text answer."""
        r = self.ask(question)
        if r["error"]:
            return f"SQL:\n  {r['sql']}\n\nError: {r['error']}"
        df = r["dataframe"]
        table_str = df.to_string(index=False) if len(df) <= 50 else df.head(30).to_string(index=False) + "\n... (truncated)"
        return f"SQL:\n  {r['sql']}\n\nResults ({len(df)} rows):\n{table_str}"

    # ── Internals ──────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Strip markdown fences and trailing semicolons."""
        cleaned = re.sub(r"```(?:sql)?\s*", "", raw)
        cleaned = cleaned.strip().rstrip(";")
        return cleaned

    def _execute(self, sql: str) -> dict[str, Any]:
        """Run SQL against SQLite and return columns + rows."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return {
                "columns": list(df.columns),
                "rows": df.values.tolist(),
                "dataframe": df,
                "error": None,
            }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "dataframe": pd.DataFrame(),
                "error": str(e),
            }
