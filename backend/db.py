"""
SQLite database helper for user auth and grade history.
"""

import os
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

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
                email TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
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
                edited_concepts_json TEXT,
                original_prediction INTEGER,
                original_rating INTEGER,
                pinned INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        # Migration for existing tables
        try:
            cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
            # Migrate existing username to email if adding column
            cur.execute("UPDATE users SET email = username WHERE email IS NULL")
            # Since username was unique, email should be unique too
            # But we can't easily add UNIQUE constraint via ALTER TABLE in SQLite without recreating
            # For now, we assume app logic handles uniqueness or rebuild table manually if needed
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("ALTER TABLE grade_records ADD COLUMN original_prediction INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("ALTER TABLE grade_records ADD COLUMN original_rating INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("ALTER TABLE grade_records ADD COLUMN pinned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass


def _get_user_id(conn: sqlite3.Connection, email: str) -> Optional[int]:
    cur = conn.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    return int(row["id"]) if row else None


def create_demo_user(db_path: str) -> None:
    """
    Ensure demo user exists with email 'demo@asu.edu', username 'Demo', and password 'demo'.
    """
    create_user(db_path, "demo@asu.edu", "Demo", "demo", ignore_exists=True)


def create_user(db_path: str, email: str, username: str, password: str, ignore_exists: bool = False) -> None:
    """
    Create a new user. Raises sqlite3.IntegrityError if email exists unless ignore_exists=True.
    """
    password_hash = pwd_context.hash(password)
    with get_conn(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
                (email, username, password_hash),
            )
        except sqlite3.IntegrityError:
            if ignore_exists:
                return
            raise


def verify_user(db_path: str, email: str, password: str) -> bool:
    """
    Verify email and password.
    """
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", (email,)
        )
        row = cur.fetchone()
        if not row:
            return False
        return pwd_context.verify(password, row["password_hash"])


def insert_grade_record(
    db_path: str,
    email: str,
    *,
    text: str,
    model_name: str,
    mode: str,
    prediction: int,
    rating: int,
    probabilities: List[float],
    concept_predictions: Optional[List[Dict[str, Any]]] = None,
    edited_concepts: Optional[Dict[str, int]] = None,
    original_prediction: Optional[int] = None,
    original_rating: Optional[int] = None,
    pinned: bool = False,
) -> int:
    """
    Insert a grade record for the given user and return new record id.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, email)
        if user_id is None:
            raise ValueError("User does not exist")
        cur = conn.execute(
            """
            INSERT INTO grade_records
            (user_id, text, model_name, mode, prediction, rating, probabilities_json, concept_predictions_json,
             edited_concepts_json, original_prediction, original_rating, pinned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(edited_concepts) if edited_concepts is not None else None,
                original_prediction,
                original_rating,
                1 if pinned else 0
            ),
        )
        return int(cur.lastrowid)


def list_grade_records(
    db_path: str,
    email: str,
    *,
    limit: int = 20,
    offset: int = 0,
    pinned: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    List grade record summaries for a user.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, email)
        if user_id is None:
            return []
        
        query = """
            SELECT id, created_at, model_name, mode, rating, text, pinned
            FROM grade_records
            WHERE user_id = ?
        """
        params = [user_id]
        
        if pinned is not None:
            query += " AND pinned = ?"
            params.append(1 if pinned else 0)
            
        query += """
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cur = conn.execute(query, tuple(params))
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
                    "pinned": bool(r["pinned"]),
                }
            )
        return items


def get_grade_record(
    db_path: str,
    email: str,
    record_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific grade record detail for a user.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, email)
        if user_id is None:
            return None
        cur = conn.execute(
            """
            SELECT id, created_at, model_name, mode, prediction, rating, text,
                   probabilities_json, concept_predictions_json,
                   edited_concepts_json, original_prediction, original_rating, pinned
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
        edited_concepts = (
            json.loads(r["edited_concepts_json"]) if r["edited_concepts_json"] else None
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
            "edited_concepts": edited_concepts,
            "original_prediction": r["original_prediction"],
            "original_rating": r["original_rating"],
            "pinned": bool(r["pinned"]),
        }


def delete_grade_record(db_path: str, email: str, record_id: int) -> bool:
    """
    Delete a specific grade record for the given user. Returns True if deleted.
    """
    with get_conn(db_path) as conn:
        user_id = _get_user_id(conn, email)
        if user_id is None:
            return False
        cur = conn.execute(
            "DELETE FROM grade_records WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        return cur.rowcount > 0


