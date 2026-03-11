"""
Sets up a sample e-commerce SQLite database with realistic business data.
Tables: customers, products, categories, orders, order_items, reviews.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "business.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL,
    department    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES categories(category_id),
    unit_price   REAL NOT NULL,
    stock_qty    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    city          TEXT NOT NULL,
    state         TEXT NOT NULL,
    signup_date   DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   DATE NOT NULL,
    status       TEXT NOT NULL CHECK(status IN ('pending','shipped','delivered','cancelled')),
    total_amount REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id     INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL NOT NULL,
    line_total  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id   INTEGER PRIMARY KEY,
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_date DATE NOT NULL
);
"""

# ── Seed data ──────────────────────────────────────────────────────

CATEGORIES = [
    ("Electronics", "Tech"), ("Clothing", "Apparel"), ("Books", "Media"),
    ("Home & Kitchen", "Home"), ("Sports", "Outdoors"), ("Toys", "Kids"),
]

PRODUCT_TEMPLATES = {
    "Electronics": [
        ("Wireless Headphones", 79.99), ("USB-C Hub", 34.99),
        ("Bluetooth Speaker", 49.99), ("Webcam HD", 59.99),
        ("Mechanical Keyboard", 109.99), ("Portable Charger", 29.99),
    ],
    "Clothing": [
        ("Cotton T-Shirt", 19.99), ("Denim Jeans", 49.99),
        ("Running Shoes", 89.99), ("Winter Jacket", 129.99),
        ("Baseball Cap", 14.99), ("Wool Socks 3-Pack", 12.99),
    ],
    "Books": [
        ("Python Crash Course", 29.99), ("Data Science Handbook", 39.99),
        ("SQL Cookbook", 34.99), ("Clean Code", 37.99),
        ("Designing Data-Intensive Apps", 44.99),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Pan", 39.99), ("Coffee Maker", 64.99),
        ("Knife Set", 54.99), ("Cutting Board", 19.99),
        ("Blender", 44.99),
    ],
    "Sports": [
        ("Yoga Mat", 24.99), ("Dumbbells 10lb Pair", 34.99),
        ("Resistance Bands", 14.99), ("Jump Rope", 9.99),
        ("Water Bottle", 12.99),
    ],
    "Toys": [
        ("Building Blocks Set", 29.99), ("Board Game Classic", 24.99),
        ("RC Car", 39.99), ("Puzzle 1000pc", 17.99),
    ],
}

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Vince", "Wendy", "Xander",
]
LAST_NAMES = [
    "Smith", "Johnson", "Lee", "Brown", "Davis", "Wilson", "Moore",
    "Taylor", "Anderson", "Thomas", "Martin", "Garcia", "Clark", "Hall",
]
CITIES_STATES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Atlanta", "GA"),
    ("Miami", "FL"), ("Portland", "OR"), ("Austin", "TX"),
]


def _random_date(start: datetime, end: datetime) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def build_database(db_path: str = DB_PATH, seed: int = 42) -> str:
    """Create and populate the database. Returns the path to the .db file."""
    random.seed(seed)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)

    # Categories
    for i, (name, dept) in enumerate(CATEGORIES, start=1):
        cur.execute("INSERT INTO categories VALUES (?,?,?)", (i, name, dept))

    # Products
    pid = 1
    cat_map = {name: i for i, (name, _) in enumerate(CATEGORIES, start=1)}
    for cat_name, products in PRODUCT_TEMPLATES.items():
        for pname, price in products:
            stock = random.randint(20, 500)
            cur.execute(
                "INSERT INTO products VALUES (?,?,?,?,?)",
                (pid, pname, cat_map[cat_name], price, stock),
            )
            pid += 1

    total_products = pid - 1

    # Customers (60 customers)
    emails_seen = set()
    for cid in range(1, 61):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        email = f"{fn.lower()}.{ln.lower()}{cid}@example.com"
        while email in emails_seen:
            email = f"{fn.lower()}{random.randint(1,999)}@example.com"
        emails_seen.add(email)
        city, state = random.choice(CITIES_STATES)
        signup = _random_date(datetime(2023, 1, 1), datetime(2025, 6, 30))
        cur.execute(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)",
            (cid, fn, ln, email, city, state, signup),
        )

    # Orders & order_items (≈400 orders)
    statuses = ["pending", "shipped", "delivered", "delivered", "delivered", "cancelled"]
    oid = 1
    iid = 1
    for _ in range(400):
        cust = random.randint(1, 60)
        odate = _random_date(datetime(2024, 1, 1), datetime(2025, 12, 31))
        status = random.choice(statuses)
        n_items = random.randint(1, 5)
        items = []
        for _ in range(n_items):
            prod = random.randint(1, total_products)
            qty = random.randint(1, 4)
            cur.execute("SELECT unit_price FROM products WHERE product_id=?", (prod,))
            price = cur.fetchone()[0]
            line = round(price * qty, 2)
            items.append((iid, oid, prod, qty, price, line))
            iid += 1
        total = round(sum(it[5] for it in items), 2)
        cur.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?)",
            (oid, cust, odate, status, total),
        )
        cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
        oid += 1

    # Reviews
    rid = 1
    for _ in range(300):
        prod = random.randint(1, total_products)
        cust = random.randint(1, 60)
        # slight positive skew
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
        rdate = _random_date(datetime(2024, 3, 1), datetime(2025, 12, 31))
        cur.execute(
            "INSERT INTO reviews VALUES (?,?,?,?,?)",
            (rid, prod, cust, rating, rdate),
        )
        rid += 1

    conn.commit()
    conn.close()
    print(f"Database created at {db_path}")
    print(f"  {len(CATEGORIES)} categories, {total_products} products, 60 customers")
    print(f"  {oid-1} orders, {iid-1} order items, {rid-1} reviews")
    return db_path


if __name__ == "__main__":
    build_database()
