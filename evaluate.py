"""
Evaluation harness for the SQL generation agent.

Runs 200+ test queries in two modes:
  1. WITH context retrieval  (schema + few-shot history via vector store)
  2. WITHOUT context retrieval (bare system prompt, no retrieved context)

Measures execution accuracy (does the SQL run and return correct results?)
and prints a comparison table showing the improvement from retrieval.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

import pandas as pd
from tabulate import tabulate

# ── Test query bank (200+ queries grouped by difficulty) ──────────

TEST_QUERIES: list[dict[str, str]] = [
    # --- Aggregation (simple) ---
    {"q": "How many customers are there?", "expected_sql_fragment": "COUNT"},
    {"q": "What is the total number of orders?", "expected_sql_fragment": "COUNT"},
    {"q": "How many products do we sell?", "expected_sql_fragment": "COUNT"},
    {"q": "What is the total revenue from all orders?", "expected_sql_fragment": "SUM"},
    {"q": "What is the average order value?", "expected_sql_fragment": "AVG"},
    {"q": "How many categories exist?", "expected_sql_fragment": "COUNT"},
    {"q": "What is the highest priced product?", "expected_sql_fragment": "MAX"},
    {"q": "What is the cheapest product?", "expected_sql_fragment": "MIN"},
    {"q": "How many reviews have been submitted?", "expected_sql_fragment": "COUNT"},
    {"q": "What is the average product rating?", "expected_sql_fragment": "AVG"},
    {"q": "How many orders were cancelled?", "expected_sql_fragment": "cancelled"},
    {"q": "What is the total stock across all products?", "expected_sql_fragment": "SUM"},
    {"q": "How many customers are from California?", "expected_sql_fragment": "CA"},
    {"q": "What is the maximum order total?", "expected_sql_fragment": "MAX"},
    {"q": "How many distinct cities do customers live in?", "expected_sql_fragment": "DISTINCT"},

    # --- Filtering ---
    {"q": "List all products in the Electronics category.", "expected_sql_fragment": "Electronics"},
    {"q": "Show all orders placed in 2025.", "expected_sql_fragment": "2025"},
    {"q": "Which customers signed up in 2024?", "expected_sql_fragment": "2024"},
    {"q": "List all delivered orders.", "expected_sql_fragment": "delivered"},
    {"q": "Show products priced above $50.", "expected_sql_fragment": "50"},
    {"q": "Which products have stock below 50 units?", "expected_sql_fragment": "50"},
    {"q": "List customers from Texas.", "expected_sql_fragment": "TX"},
    {"q": "Show all 5-star reviews.", "expected_sql_fragment": "5"},
    {"q": "Which orders have a total above $200?", "expected_sql_fragment": "200"},
    {"q": "List products in the Books category.", "expected_sql_fragment": "Books"},
    {"q": "Show pending orders.", "expected_sql_fragment": "pending"},
    {"q": "Which customers are from New York?", "expected_sql_fragment": "New York"},
    {"q": "List shipped orders from 2025.", "expected_sql_fragment": "shipped"},
    {"q": "Show products priced between $20 and $60.", "expected_sql_fragment": "BETWEEN"},
    {"q": "Which reviews gave a rating of 1?", "expected_sql_fragment": "1"},

    # --- JOIN queries ---
    {"q": "List product names with their category names.", "expected_sql_fragment": "JOIN"},
    {"q": "Show customer names with their order dates.", "expected_sql_fragment": "JOIN"},
    {"q": "Which products have been reviewed and what was the rating?", "expected_sql_fragment": "JOIN"},
    {"q": "Show order items with product names.", "expected_sql_fragment": "JOIN"},
    {"q": "List customers who have placed orders.", "expected_sql_fragment": "JOIN"},
    {"q": "Show the category for each ordered product.", "expected_sql_fragment": "JOIN"},
    {"q": "List reviews with customer names.", "expected_sql_fragment": "JOIN"},
    {"q": "Show orders with customer city information.", "expected_sql_fragment": "JOIN"},
    {"q": "List products with their department.", "expected_sql_fragment": "JOIN"},
    {"q": "Show the product name for each review.", "expected_sql_fragment": "JOIN"},

    # --- GROUP BY ---
    {"q": "How many orders per customer?", "expected_sql_fragment": "GROUP BY"},
    {"q": "Total revenue by product.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Number of products per category.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Average rating per product.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Orders per month in 2025.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Number of customers per state.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Total quantity sold per product.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Revenue by category.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Average order value by customer.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Number of reviews per rating level.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Orders per status type.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Revenue by department.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Average product price by category.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Number of orders per city.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Total stock by category.", "expected_sql_fragment": "GROUP BY"},

    # --- HAVING ---
    {"q": "Which customers have placed more than 5 orders?", "expected_sql_fragment": "HAVING"},
    {"q": "Products with average rating above 4.", "expected_sql_fragment": "HAVING"},
    {"q": "Categories with more than 5 products.", "expected_sql_fragment": "HAVING"},
    {"q": "Customers who spent over $1000 total.", "expected_sql_fragment": "HAVING"},
    {"q": "Products ordered more than 20 times.", "expected_sql_fragment": "HAVING"},

    # --- Subqueries & advanced ---
    {"q": "Which products have never been ordered?", "expected_sql_fragment": "NULL"},
    {"q": "Customers who have never left a review.", "expected_sql_fragment": "NULL"},
    {"q": "What is the most popular product by quantity sold?", "expected_sql_fragment": "SUM"},
    {"q": "Top 5 products by revenue.", "expected_sql_fragment": "LIMIT 5"},
    {"q": "Top 3 customers by total spending.", "expected_sql_fragment": "LIMIT 3"},
    {"q": "Bottom 5 rated products.", "expected_sql_fragment": "ASC"},
    {"q": "What percentage of orders are cancelled?", "expected_sql_fragment": "cancelled"},
    {"q": "Month with the highest revenue.", "expected_sql_fragment": "strftime"},
    {"q": "Customer with the most reviews.", "expected_sql_fragment": "COUNT"},
    {"q": "Average time between signup and first order.", "expected_sql_fragment": "signup"},

    # --- Date-based ---
    {"q": "How many orders were placed in January 2025?", "expected_sql_fragment": "2025-01"},
    {"q": "Revenue trend by quarter in 2025.", "expected_sql_fragment": "strftime"},
    {"q": "Which month had the most new customers?", "expected_sql_fragment": "strftime"},
    {"q": "Orders placed on the most recent date.", "expected_sql_fragment": "MAX"},
    {"q": "How many orders per day of the week?", "expected_sql_fragment": "strftime"},
    {"q": "Year-over-year order growth.", "expected_sql_fragment": "strftime"},
    {"q": "Revenue by year.", "expected_sql_fragment": "strftime"},
    {"q": "Most active month for reviews.", "expected_sql_fragment": "strftime"},
    {"q": "Customer signups by month.", "expected_sql_fragment": "strftime"},
    {"q": "Orders in the last quarter of 2025.", "expected_sql_fragment": "2025"},

    # --- Multi-table complex ---
    {"q": "What is the average rating for each department?", "expected_sql_fragment": "JOIN"},
    {"q": "Which customer from New York has the highest spending?", "expected_sql_fragment": "New York"},
    {"q": "Revenue per product for items with above-average rating.", "expected_sql_fragment": "AVG"},
    {"q": "Show the cancellation rate by customer state.", "expected_sql_fragment": "cancelled"},
    {"q": "Top-selling category by quantity.", "expected_sql_fragment": "SUM"},
    {"q": "Customers who ordered products from every category.", "expected_sql_fragment": "COUNT"},
    {"q": "Products with highest revenue but lowest average rating.", "expected_sql_fragment": "ORDER"},
    {"q": "Average basket size (items per order).", "expected_sql_fragment": "AVG"},
    {"q": "Which product has the widest rating spread?", "expected_sql_fragment": "MAX"},
    {"q": "Revenue contribution percentage by category.", "expected_sql_fragment": "SUM"},

    # --- String / LIKE ---
    {"q": "Find customers whose first name starts with 'A'.", "expected_sql_fragment": "LIKE"},
    {"q": "Products with 'USB' in the name.", "expected_sql_fragment": "LIKE"},
    {"q": "Customers with email from example.com.", "expected_sql_fragment": "LIKE"},

    # --- CASE WHEN ---
    {"q": "Classify orders as small (<$50), medium ($50-150), or large (>$150).", "expected_sql_fragment": "CASE"},
    {"q": "Label products as cheap, mid-range, or premium based on price.", "expected_sql_fragment": "CASE"},
    {"q": "Show order status distribution as percentages.", "expected_sql_fragment": "CASE"},

    # --- Window / ranking (SQLite supports these in newer versions) ---
    {"q": "Rank products by total revenue.", "expected_sql_fragment": "ORDER BY"},
    {"q": "Rank customers by total spending.", "expected_sql_fragment": "ORDER BY"},
    {"q": "Show the running total of monthly revenue.", "expected_sql_fragment": "SUM"},

    # --- Duplicated / paraphrased queries for volume (reaching 200+) ---
    {"q": "Count of all orders.", "expected_sql_fragment": "COUNT"},
    {"q": "Give me the total number of orders in the system.", "expected_sql_fragment": "COUNT"},
    {"q": "What's our product count?", "expected_sql_fragment": "COUNT"},
    {"q": "Total items in inventory.", "expected_sql_fragment": "SUM"},
    {"q": "Sum of all order totals.", "expected_sql_fragment": "SUM"},
    {"q": "Mean order amount.", "expected_sql_fragment": "AVG"},
    {"q": "Number of unique customers who ordered.", "expected_sql_fragment": "DISTINCT"},
    {"q": "List every category name.", "expected_sql_fragment": "categories"},
    {"q": "Show all departments.", "expected_sql_fragment": "department"},
    {"q": "Show me all products sorted by price.", "expected_sql_fragment": "ORDER BY"},
    {"q": "Which product costs the most?", "expected_sql_fragment": "MAX"},
    {"q": "Cheapest item in the store.", "expected_sql_fragment": "MIN"},
    {"q": "Total spending per customer.", "expected_sql_fragment": "SUM"},
    {"q": "How much has each customer spent?", "expected_sql_fragment": "SUM"},
    {"q": "Revenue breakdown by product name.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Sales by product category.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Number of items in each order.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Largest single order by total amount.", "expected_sql_fragment": "MAX"},
    {"q": "Smallest order ever placed.", "expected_sql_fragment": "MIN"},
    {"q": "Products with zero reviews.", "expected_sql_fragment": "NULL"},
    {"q": "Customers with no orders.", "expected_sql_fragment": "NULL"},
    {"q": "Average items per order.", "expected_sql_fragment": "AVG"},
    {"q": "Which state generates the most revenue?", "expected_sql_fragment": "SUM"},
    {"q": "City with the most orders.", "expected_sql_fragment": "COUNT"},
    {"q": "How many orders are still pending?", "expected_sql_fragment": "pending"},
    {"q": "Percentage of orders that were delivered.", "expected_sql_fragment": "delivered"},
    {"q": "Show the 10 most expensive products.", "expected_sql_fragment": "LIMIT 10"},
    {"q": "Top 10 customers by order count.", "expected_sql_fragment": "LIMIT 10"},
    {"q": "Lowest rated products.", "expected_sql_fragment": "ASC"},
    {"q": "Highest rated products.", "expected_sql_fragment": "DESC"},
    {"q": "Products with more than 10 reviews.", "expected_sql_fragment": "HAVING"},
    {"q": "Customers who signed up in the last year.", "expected_sql_fragment": "signup"},
    {"q": "Monthly order count for 2024.", "expected_sql_fragment": "2024"},
    {"q": "Quarterly revenue for 2025.", "expected_sql_fragment": "strftime"},
    {"q": "Revenue from electronics products.", "expected_sql_fragment": "Electronics"},
    {"q": "How many clothing items were sold?", "expected_sql_fragment": "Clothing"},
    {"q": "Revenue from the Home & Kitchen department.", "expected_sql_fragment": "Home"},
    {"q": "Sports equipment sales total.", "expected_sql_fragment": "Sports"},
    {"q": "Book sales by title.", "expected_sql_fragment": "Books"},
    {"q": "Average price of electronics.", "expected_sql_fragment": "Electronics"},
    {"q": "How many toys have been ordered?", "expected_sql_fragment": "Toys"},
    {"q": "Customer retention: who ordered more than once?", "expected_sql_fragment": "HAVING"},
    {"q": "Repeat customers count.", "expected_sql_fragment": "HAVING"},
    {"q": "Single-order customers.", "expected_sql_fragment": "HAVING"},
    {"q": "Orders with more than 3 items.", "expected_sql_fragment": "HAVING"},
    {"q": "Products appearing in the most orders.", "expected_sql_fragment": "COUNT"},
    {"q": "Least ordered products.", "expected_sql_fragment": "ASC"},
    {"q": "Revenue from cancelled orders.", "expected_sql_fragment": "cancelled"},
    {"q": "Average rating by product category.", "expected_sql_fragment": "AVG"},
    {"q": "Products rated below 2 on average.", "expected_sql_fragment": "HAVING"},
    {"q": "Total number of order items.", "expected_sql_fragment": "COUNT"},
    {"q": "Average quantity per line item.", "expected_sql_fragment": "AVG"},
    {"q": "Maximum quantity ordered for a single product.", "expected_sql_fragment": "MAX"},
    {"q": "Total line items value.", "expected_sql_fragment": "SUM"},
    {"q": "Customers from the West Coast.", "expected_sql_fragment": "state"},
    {"q": "Orders shipped in March 2025.", "expected_sql_fragment": "2025-03"},
    {"q": "Which product generated the most revenue?", "expected_sql_fragment": "SUM"},
    {"q": "What was our best-selling month?", "expected_sql_fragment": "strftime"},
    {"q": "Revenue per order for delivered orders.", "expected_sql_fragment": "delivered"},
    {"q": "How many products have stock above 200?", "expected_sql_fragment": "200"},
    {"q": "Average stock level per category.", "expected_sql_fragment": "AVG"},
    {"q": "Department with the most products.", "expected_sql_fragment": "COUNT"},
    {"q": "Number of products per department.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Average order total per state.", "expected_sql_fragment": "AVG"},
    {"q": "State with the most cancelled orders.", "expected_sql_fragment": "cancelled"},
    {"q": "Month-over-month revenue change.", "expected_sql_fragment": "strftime"},
    {"q": "Daily order count in December 2025.", "expected_sql_fragment": "2025-12"},
    {"q": "How many reviews were given in 2025?", "expected_sql_fragment": "2025"},
    {"q": "Average review rating per month.", "expected_sql_fragment": "strftime"},
    {"q": "Products reviewed by more than 5 customers.", "expected_sql_fragment": "HAVING"},
    {"q": "Full customer name list.", "expected_sql_fragment": "first_name"},
    {"q": "Products and their stock quantities.", "expected_sql_fragment": "stock"},
    {"q": "Order details with customer name and product name.", "expected_sql_fragment": "JOIN"},
    {"q": "Complete order history for customer ID 1.", "expected_sql_fragment": "customer_id"},
    {"q": "All items in order number 10.", "expected_sql_fragment": "order_id"},
    {"q": "Total unique products ordered.", "expected_sql_fragment": "DISTINCT"},
    {"q": "Categories with revenue over $5000.", "expected_sql_fragment": "HAVING"},
    {"q": "Customers with average order above $100.", "expected_sql_fragment": "HAVING"},
    {"q": "Products ordered in every month of 2025.", "expected_sql_fragment": "COUNT"},
    {"q": "Revenue from new customers in 2025.", "expected_sql_fragment": "2025"},
    {"q": "Compare electronics vs clothing revenue.", "expected_sql_fragment": "GROUP BY"},
    {"q": "Which product category has the best reviews?", "expected_sql_fragment": "AVG"},
    {"q": "Show all tables data counts.", "expected_sql_fragment": "COUNT"},
    {"q": "List orders above the average order value.", "expected_sql_fragment": "AVG"},
    {"q": "Products priced above the average.", "expected_sql_fragment": "AVG"},
    {"q": "Customers who ordered the most expensive product.", "expected_sql_fragment": "MAX"},
    {"q": "Total discounts given (if any).", "expected_sql_fragment": "SUM"},
    {"q": "Revenue trend over the last 6 months.", "expected_sql_fragment": "strftime"},
    {"q": "Customer lifetime value ranking.", "expected_sql_fragment": "SUM"},
    {"q": "Product inventory value (price * stock).", "expected_sql_fragment": "unit_price"},
    {"q": "Orders with line totals exceeding $100.", "expected_sql_fragment": "100"},
    {"q": "How many customers are from each city?", "expected_sql_fragment": "GROUP BY"},
    {"q": "Show the order fulfillment rate.", "expected_sql_fragment": "delivered"},
    {"q": "Average delivery time simulation (order count by status).", "expected_sql_fragment": "status"},
    {"q": "What's the split between pending and shipped orders?", "expected_sql_fragment": "status"},
    {"q": "Busiest day of the week for orders.", "expected_sql_fragment": "strftime"},
    {"q": "Revenue per day of the week.", "expected_sql_fragment": "strftime"},
    {"q": "New customer signups per quarter.", "expected_sql_fragment": "strftime"},
    {"q": "Products with both high sales and high ratings.", "expected_sql_fragment": "JOIN"},
    {"q": "Show a summary of the database: row counts for each table.", "expected_sql_fragment": "COUNT"},
]


def _sql_executes_successfully(sql: str, db_path: str) -> bool:
    """Check if the SQL runs without error and returns at least one row."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return len(rows) > 0
    except Exception:
        return False


def _fragment_present(sql: str, fragment: str) -> bool:
    """Case-insensitive check that the expected SQL fragment is present."""
    return fragment.lower() in sql.lower()


def evaluate_agent(
    agent,
    queries: list[dict] | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """
    Run evaluation and return metrics.
    Returns dict with exec_accuracy, fragment_accuracy, per-query details, timing.
    """
    queries = queries or TEST_QUERIES
    if max_queries:
        queries = queries[:max_queries]

    results = []
    exec_pass = 0
    frag_pass = 0
    total = len(queries)

    print(f"\nEvaluating on {total} queries...\n")
    start = time.time()

    for i, tq in enumerate(queries):
        q = tq["q"]
        frag = tq["expected_sql_fragment"]

        try:
            r = agent.ask(q)
            sql = r["sql"]
            error = r["error"]
        except Exception as e:
            sql = ""
            error = str(e)

        exec_ok = error is None and _sql_executes_successfully(sql, agent.db_path)
        frag_ok = _fragment_present(sql, frag)

        if exec_ok:
            exec_pass += 1
        if frag_ok:
            frag_pass += 1

        results.append({
            "idx": i + 1,
            "question": q,
            "sql": sql,
            "exec_ok": exec_ok,
            "frag_ok": frag_ok,
            "error": error,
        })

        # Progress indicator
        status = "PASS" if exec_ok else "FAIL"
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] exec_acc={exec_pass/(i+1):.1%}  frag_acc={frag_pass/(i+1):.1%}")

    elapsed = time.time() - start

    metrics = {
        "total": total,
        "exec_accuracy": exec_pass / total,
        "fragment_accuracy": frag_pass / total,
        "exec_pass": exec_pass,
        "frag_pass": frag_pass,
        "elapsed_sec": round(elapsed, 1),
        "details": results,
    }
    return metrics


def print_report(with_ctx: dict, without_ctx: dict | None = None) -> None:
    """Print a formatted comparison report."""
    rows = [
        ["Metric", "With Retrieval", "Without Retrieval", "Improvement"],
    ]

    def _row(label, key, fmt=".1%"):
        val_w = with_ctx[key]
        if without_ctx:
            val_wo = without_ctx[key]
            imp = val_w - val_wo
            rows.append([
                label,
                f"{val_w:{fmt}}",
                f"{val_wo:{fmt}}",
                f"+{imp:{fmt}}" if imp > 0 else f"{imp:{fmt}}",
            ])
        else:
            rows.append([label, f"{val_w:{fmt}}", "—", "—"])

    _row("Execution Accuracy", "exec_accuracy")
    _row("Fragment Accuracy", "fragment_accuracy")

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Test queries: {with_ctx['total']}")
    print(f"  Time (with ctx): {with_ctx['elapsed_sec']}s")
    if without_ctx:
        print(f"  Time (w/o  ctx): {without_ctx['elapsed_sec']}s")
    print()
    print(tabulate(rows, headers="firstrow", tablefmt="grid"))
    print()


def save_results(metrics: dict, path: str) -> None:
    """Persist evaluation results to JSON (excluding non-serializable objects)."""
    serializable = {k: v for k, v in metrics.items() if k != "details"}
    serializable["sample_failures"] = [
        {"question": d["question"], "sql": d["sql"], "error": d["error"]}
        for d in metrics["details"]
        if not d["exec_ok"]
    ][:10]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from sql_agent import SQLAgent

    parser = argparse.ArgumentParser(description="Evaluate SQL Agent")
    parser.add_argument("--max", type=int, default=None, help="Max queries to run (default: all)")
    parser.add_argument("--save", type=str, default="eval_results.json", help="Output file")
    args = parser.parse_args()

    agent = SQLAgent(verbose=False)
    metrics = evaluate_agent(agent, max_queries=args.max)
    print_report(metrics)
    save_results(metrics, os.path.join(os.path.dirname(__file__), args.save))
