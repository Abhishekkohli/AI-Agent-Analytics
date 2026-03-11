"""
Context management pipeline for the SQL generation agent.

Given a user's natural-language question, this module:
  1. Retrieves the most relevant table schemas via vector similarity
  2. Retrieves similar past (question → SQL) pairs as few-shot examples
  3. Assembles a structured prompt that maximises LLM accuracy
"""

from __future__ import annotations
from vector_store import VectorStore


SYSTEM_PROMPT = """\
You are an expert SQL analyst. You have access to a SQLite database with the
schema described below. Your job is to convert the user's natural-language
business question into a single, correct SQL query.

Rules:
- Output ONLY the SQL query, no explanations.
- Use SQLite-compatible syntax (e.g. strftime for dates).
- Do NOT use LIMIT unless the user explicitly asks for a specific count.
- Always qualify ambiguous column names with table aliases.
- Use JOINs, not sub-selects, when possible.
- Return readable column aliases (e.g. "total_revenue", not "SUM(oi.line_total)").
"""


class ContextManager:
    """Builds the LLM prompt by injecting retrieved schema + few-shot context."""

    def __init__(self, store: VectorStore, schema_top_k: int = 4, history_top_k: int = 3):
        self.store = store
        self.schema_top_k = schema_top_k
        self.history_top_k = history_top_k

    def build_prompt(self, user_question: str) -> list[dict]:
        """
        Returns an OpenAI-style messages list:
          [system, user]
        with retrieved context injected into the system message.
        """
        schema_ctx = self._retrieve_schema(user_question)
        history_ctx = self._retrieve_history(user_question)
        system_msg = self._assemble_system(schema_ctx, history_ctx)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_question},
        ]

    # ── Private helpers ────────────────────────────────────────────

    def _retrieve_schema(self, question: str) -> str:
        results = self.store.search(question, top_k=self.schema_top_k, doc_type="schema")
        if not results:
            return "No schema information retrieved."
        blocks = []
        for r in results:
            blocks.append(r["text"])
        return "\n\n".join(blocks)

    def _retrieve_history(self, question: str) -> str:
        results = self.store.search(question, top_k=self.history_top_k, doc_type="history")
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            lines.append(f"Example {i}:")
            lines.append(f"  Q: {meta['question']}")
            lines.append(f"  SQL: {meta['sql']}")
        return "\n".join(lines)

    def _assemble_system(self, schema_ctx: str, history_ctx: str) -> str:
        parts = [SYSTEM_PROMPT, "=== DATABASE SCHEMA ===", schema_ctx]
        if history_ctx:
            parts.append("=== SIMILAR QUERY EXAMPLES ===")
            parts.append(history_ctx)
        return "\n\n".join(parts)

    # ── Runtime history injection ──────────────────────────────────

    def add_to_history(self, question: str, sql: str) -> None:
        """
        Add a successful (question, SQL) pair to the vector store at
        runtime so the agent improves as the session progresses.
        """
        doc = {
            "text": f"Question: {question}\nSQL: {sql}",
            "type": "history",
            "metadata": {"question": question, "sql": sql},
        }
        self.store.add_documents([doc])
        # Rebuild index to include the new document (cheap for small stores)
        self.store.build_index()
