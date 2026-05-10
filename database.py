"""
database.py — Supabase (PostgreSQL) persistence for evaluation history.
Replaces the local SQLite file with a persistent cloud database.

Supabase setup (one-time):
  1. Go to supabase.com → New project
  2. Open SQL Editor and run the CREATE TABLE statement below
  3. Copy your Project URL and anon/service_role key into secrets
"""

import json
from datetime import datetime
from supabase import create_client, Client

TABLE = "evaluations"


# ── SQL to run once in Supabase SQL Editor ───────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    student_name      TEXT        DEFAULT 'Anonymous',
    question          TEXT        NOT NULL,
    student_answer    TEXT        NOT NULL,
    marks_awarded     INTEGER     NOT NULL,
    max_marks         INTEGER     NOT NULL,
    percentage        FLOAT       NOT NULL,
    grade             TEXT        NOT NULL,
    concepts_covered  JSONB,
    concepts_missing  JSONB,
    strengths         JSONB,
    weaknesses        JSONB,
    detailed_feedback TEXT,
    improved_answer   TEXT,
    context_used      TEXT
);
"""


def _client(url: str, key: str) -> Client:
    return create_client(url, key)


def save_evaluation(url: str, key: str, result: dict,
                    question: str, student_answer: str,
                    student_name: str = "Anonymous", context: str = "") -> int | None:
    """Insert one evaluation into Supabase. Returns the new row id or None."""
    row = {
        "created_at":       datetime.utcnow().isoformat(),
        "student_name":     student_name,
        "question":         question,
        "student_answer":   student_answer,
        "marks_awarded":    result.get("marks_awarded", 0),
        "max_marks":        result.get("max_marks", 10),
        "percentage":       result.get("percentage", 0.0),
        "grade":            result.get("grade", "N/A"),
        "concepts_covered": result.get("concepts_covered", []),
        "concepts_missing": result.get("concepts_missing", []),
        "strengths":        result.get("strengths", []),
        "weaknesses":       result.get("weaknesses", []),
        "detailed_feedback": result.get("detailed_feedback", ""),
        "improved_answer":  result.get("improved_answer", ""),
        "context_used":     context[:2000],
    }
    try:
        client = _client(url.rstrip("/"), key)
        resp = client.table(TABLE).insert(row).execute()
        if resp.data:
            return resp.data[0]["id"]
        else:
            print(f"[DB] save_evaluation: no data returned. resp={resp}")
            return None
    except Exception as e:
        print(f"[DB] save_evaluation error: {type(e).__name__}: {e}")
        raise  # re-raise so app.py can show it


def get_recent_evaluations(url: str, key: str, limit: int = 50) -> list[dict]:
    """Return the most recent evaluations, newest first."""
    try:
        resp = (
            _client(url, key)
            .table(TABLE)
            .select("*")
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        print(f"[DB] get_recent_evaluations error: {e}")
        return []


def get_stats(url: str, key: str) -> dict:
    """Aggregate stats: total, avg %, top/low score, grade breakdown."""
    try:
        resp = _client(url, key).table(TABLE).select(
            "marks_awarded, percentage, grade"
        ).execute()
        rows = resp.data or []
    except Exception as e:
        print(f"[DB] get_stats error: {e}")
        return {"total": 0, "avg_pct": 0, "top_score": 0, "low_score": 0, "grades": {}}

    if not rows:
        return {"total": 0, "avg_pct": 0, "top_score": 0, "low_score": 0, "grades": {}}

    total    = len(rows)
    avg_pct  = round(sum(r["percentage"] for r in rows) / total, 1)
    top      = max(r["marks_awarded"] for r in rows)
    low      = min(r["marks_awarded"] for r in rows)
    grades: dict[str, int] = {}
    for r in rows:
        g = r["grade"]
        grades[g] = grades.get(g, 0) + 1

    return {"total": total, "avg_pct": avg_pct, "top_score": top, "low_score": low, "grades": grades}


def delete_all(url: str, key: str):
    """Delete every row from the evaluations table."""
    try:
        # Supabase requires a filter; use neq on id (always true)
        _client(url, key).table(TABLE).delete().neq("id", -1).execute()
    except Exception as e:
        print(f"[DB] delete_all error: {e}")
