#!/usr/bin/env python3
"""
AI Agent Analytics Assistant — Interactive CLI

Usage:
  python app.py             # interactive REPL
  python app.py --setup     # initialise DB + vector store only
  python app.py --eval 20   # run evaluation on 20 test queries
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from setup_database import build_database, DB_PATH
from vector_store import build_vector_store
from sql_agent import SQLAgent


BANNER = r"""
 ╔══════════════════════════════════════════════════╗
 ║   AI Agent Analytics Assistant                   ║
 ║   Natural Language → SQL → Results               ║
 ╚══════════════════════════════════════════════════╝
"""


def setup() -> None:
    """Create DB and build vector store from scratch."""
    print("Setting up database...")
    db = build_database()
    print("Building vector store...")
    build_vector_store(db)
    print("Setup complete.\n")


def interactive(agent: SQLAgent) -> None:
    """REPL loop: ask questions, get SQL + results."""
    print(BANNER)
    print("  Type a business question in plain English.")
    print("  Commands:  :sql   — show last SQL only")
    print("             :schema — show retrieved schema")
    print("             :quit  — exit\n")

    last_result = None

    while True:
        try:
            question = input("📊 Ask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in (":quit", ":exit", ":q"):
            print("Goodbye!")
            break
        if question == ":sql" and last_result:
            print(f"\n  {last_result['sql']}\n")
            continue
        if question == ":schema":
            msgs = agent.ctx.build_prompt("show all tables")
            schema_section = msgs[0]["content"].split("=== DATABASE SCHEMA ===")[-1]
            schema_section = schema_section.split("=== SIMILAR")[0]
            print(textwrap.indent(schema_section.strip(), "  "))
            print()
            continue

        print()
        result = agent.ask(question)
        last_result = result

        if result["error"]:
            print(f"  SQL:   {result['sql']}")
            print(f"  Error: {result['error']}\n")
            continue

        # Display SQL
        print(f"  SQL:\n    {result['sql']}\n")

        # Display results as a formatted table
        df = result["dataframe"]
        if df.empty:
            print("  (No results)\n")
        elif len(df) > 30:
            print(df.head(30).to_string(index=False))
            print(f"  ... ({len(df)} total rows, showing first 30)\n")
        else:
            print(df.to_string(index=False))
            print(f"  ({len(df)} rows)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent Analytics Assistant")
    parser.add_argument("--setup", action="store_true", help="Initialise DB + vector store")
    parser.add_argument("--eval", type=int, metavar="N", help="Run evaluation on N queries")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--verbose", action="store_true", help="Show debug info")
    args = parser.parse_args()

    # Always ensure DB exists
    if not os.path.exists(DB_PATH):
        print("Database not found — running setup first.\n")
        setup()
    elif args.setup:
        setup()
        return

    if args.eval:
        from evaluate import evaluate_agent, print_report, save_results
        agent = SQLAgent(model=args.model, verbose=args.verbose)
        metrics = evaluate_agent(agent, max_queries=args.eval)
        print_report(metrics)
        save_results(metrics, os.path.join(os.path.dirname(__file__), "eval_results.json"))
        return

    agent = SQLAgent(model=args.model, verbose=args.verbose)
    interactive(agent)


if __name__ == "__main__":
    main()
