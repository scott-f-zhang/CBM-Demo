"""
SQLite database helper for user auth and grade history.
"""

import os
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from passlib.context import CryptContext


# Password hashing context (avoid bcrypt native dep issues)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def _ensure_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)


@contextmanager
def get_conn(db_path: str):
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_db(db_path: str) -> None:
    """
    Initialize database schema if not exists.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        # users table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        # grade_records table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS grade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                model_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                probabilities_json TEXT NOT NULL,
                concept_predictions_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )


def _get_user_id(conn: sqlite3.Connection, username: str) -> Optional[int]:
    cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return int(row["id"]) if row else None


def create_demo_user(db_path: str) -> None:
    """
    Ensure demo user exists with username/password 'demo'.
    """
    create_user(db_path, "demo", "demo", ignore_exists=True)


def create_user(db_path: str, username: str, password: str, ignore_exists: bool = False) -> None:
    """
    Create a new user. Raises sqlite3.IntegrityError if username exists unless ignore_exists=True.
    """
    password_hash = pwd_context.hash(password)
    with get_conn(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
        except sqlite3.IntegrityError:
            if ignore_exists:
                return
            raise


def verify_user(db_path: str, username: str, password: str) -> bool:
    """
    Verify username and password.
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        )
        row = cur.fetchone()
        if not row:
            return False
        return pwd_context.verify(password, row["password_hash"])


def insert_grade_record(
    db_path: str,
    username: str,
    *,
    text: str,
    model_name: str,
    mode: str,
    prediction: int,
    rating: int,
    probabilities: List[float],
    concept_predictions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Insert a grade record for the given user and return new record id.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, username)
        if user_id is None:
            raise ValueError("User does not exist")
        cur = conn.execute(
            """
            INSERT INTO grade_records
            (user_id, text, model_name, mode, prediction, rating, probabilities_json, concept_predictions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                text,
                model_name,
                mode,
                int(prediction),
                int(rating),
                json.dumps(probabilities),
                json.dumps(concept_predictions) if concept_predictions is not None else None,
            ),
        )
        return int(cur.lastrowid)


def list_grade_records(
    db_path: str,
    username: str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List grade record summaries for a user.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, username)
        if user_id is None:
            return []
        cur = conn.execute(
            """
            SELECT id, created_at, model_name, mode, rating, text
            FROM grade_records
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        )
        rows = cur.fetchall()
        items: List[Dict[str, Any]] = []
        for r in rows:
            text_preview = (r["text"][:100] + "…") if len(r["text"]) > 100 else r["text"]
            items.append(
                {
                    "id": int(r["id"]),
                    "created_at": r["created_at"],
                    "model_name": r["model_name"],
                    "mode": r["mode"],
                    "rating": int(r["rating"]),
                    "text_preview": text_preview,
                }
            )
        return items


def get_grade_record(
    db_path: str,
    username: str,
    record_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific grade record detail for a user.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, username)
        if user_id is None:
            return None
        cur = conn.execute(
            """
            SELECT id, created_at, model_name, mode, prediction, rating, text,
                   probabilities_json, concept_predictions_json
            FROM grade_records
            WHERE id = ? AND user_id = ?
            """,
            (record_id, user_id),
        )
        r = cur.fetchone()
        if not r:
            return None
        probabilities = json.loads(r["probabilities_json"]) if r["probabilities_json"] else []
        concept_predictions = (
            json.loads(r["concept_predictions_json"]) if r["concept_predictions_json"] else None
        )
        return {
            "id": int(r["id"]),
            "created_at": r["created_at"],
            "model_name": r["model_name"],
            "mode": r["mode"],
            "prediction": int(r["prediction"]),
            "rating": int(r["rating"]),
            "text": r["text"],
            "probabilities": probabilities,
            "concept_predictions": concept_predictions,
        }


def delete_grade_record(db_path: str, username: str, record_id: int) -> bool:
    """
    Delete a specific grade record for the given user. Returns True if deleted.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, username)
        if user_id is None:
            return False
        cur = conn.execute(
            "DELETE FROM grade_records WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        return cur.rowcount > 0


