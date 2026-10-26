"""Single local account system using PBKDF2-HMAC-SHA256."""
from __future__ import annotations
import hashlib
import hmac
import secrets
from aureli.persistence.db import connect

ITERATIONS = 310_000

def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"

def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256": return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds)).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False

def register(email: str, password: str, name: str) -> int:
    email = email.strip().lower()
    if "@" not in email or not name.strip(): raise ValueError("Enter a valid email and name.")
    with connect() as db:
        cur = db.execute("INSERT INTO users(email,password_hash,display_name) VALUES(?,?,?)",
                          (email, hash_password(password), name.strip()))
        db.execute("INSERT INTO profiles(user_id) VALUES(?)", (cur.lastrowid,))
        return int(cur.lastrowid)

def authenticate(email: str, password: str):
    with connect() as db:
        row = db.execute("SELECT id,password_hash,display_name FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return {"id": row["id"], "name": row["display_name"]}
    return None

def delete_local_account(user_id: int) -> bool:
    """Delete exactly one owner record; used only to clear a temporary guest session."""
    with connect() as db:
        return db.execute("DELETE FROM users WHERE id=?",(user_id,)).rowcount==1
