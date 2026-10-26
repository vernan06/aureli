"""SQLite persistence. Monetary values are stored as integer minor units."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def database_path() -> Path:
    return Path(os.environ.get("AURELI_DB_PATH", "data/aureli.sqlite3"))


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
          password_hash TEXT NOT NULL, display_name TEXT NOT NULL,
          currency TEXT NOT NULL DEFAULT 'INR', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS transactions(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          kind TEXT NOT NULL CHECK(kind IN ('income','expense')), amount_minor INTEGER NOT NULL CHECK(amount_minor>0),
          category TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', occurred_on TEXT NOT NULL,
          recurring INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS transactions_owner_date ON transactions(user_id, occurred_on);
        CREATE TABLE IF NOT EXISTS budgets(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          category TEXT NOT NULL, monthly_limit_minor INTEGER NOT NULL CHECK(monthly_limit_minor>0),
          UNIQUE(user_id, category)
        );
        CREATE TABLE IF NOT EXISTS goals(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name TEXT NOT NULL, target_minor INTEGER NOT NULL CHECK(target_minor>0),
          saved_minor INTEGER NOT NULL DEFAULT 0 CHECK(saved_minor>=0), due_on TEXT
        );
        CREATE TABLE IF NOT EXISTS loans(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name TEXT NOT NULL, principal_minor INTEGER NOT NULL CHECK(principal_minor>0),
          annual_rate REAL NOT NULL CHECK(annual_rate>=0), term_months INTEGER NOT NULL CHECK(term_months>0),
          start_date TEXT NOT NULL, extra_payment_minor INTEGER NOT NULL DEFAULT 0 CHECK(extra_payment_minor>=0)
        );
        CREATE TABLE IF NOT EXISTS loan_payments(
          id INTEGER PRIMARY KEY, loan_id INTEGER NOT NULL REFERENCES loans(id) ON DELETE CASCADE,
          amount_minor INTEGER NOT NULL CHECK(amount_minor>0), paid_on TEXT NOT NULL, note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS profiles(
          user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
          monthly_income_minor INTEGER NOT NULL DEFAULT 0, emergency_fund_minor INTEGER NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
        """)
        existing={row["name"] for row in db.execute("PRAGMA table_info(loan_payments)").fetchall()}
        if "principal_minor" not in existing:
            db.execute("ALTER TABLE loan_payments ADD COLUMN principal_minor INTEGER NOT NULL DEFAULT 0")
        if "interest_minor" not in existing:
            db.execute("ALTER TABLE loan_payments ADD COLUMN interest_minor INTEGER NOT NULL DEFAULT 0")
        if "payment_type" not in existing:
            db.execute("ALTER TABLE loan_payments ADD COLUMN payment_type TEXT NOT NULL DEFAULT 'scheduled'")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS categories(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('income','expense')),
          UNIQUE(user_id,name,kind)
        );
        CREATE TABLE IF NOT EXISTS recurring_obligations(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          name TEXT NOT NULL, category TEXT NOT NULL, amount_minor INTEGER NOT NULL CHECK(amount_minor>0),
          due_day INTEGER NOT NULL CHECK(due_day BETWEEN 1 AND 31), active INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS goal_contributions(
          id INTEGER PRIMARY KEY, goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
          amount_minor INTEGER NOT NULL CHECK(amount_minor>0), contributed_on TEXT NOT NULL, note TEXT NOT NULL DEFAULT ''
        );
        INSERT OR IGNORE INTO categories(user_id,name,kind)
          SELECT DISTINCT user_id,category,kind FROM transactions;
        """)
        if "category_id" not in {row["name"] for row in db.execute("PRAGMA table_info(transactions)").fetchall()}:
            db.execute("ALTER TABLE transactions ADD COLUMN category_id INTEGER REFERENCES categories(id)")
        db.execute("UPDATE transactions SET category_id=(SELECT id FROM categories c WHERE c.user_id=transactions.user_id AND c.name=transactions.category AND c.kind=transactions.kind) WHERE category_id IS NULL")
        db.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)")
