from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskRecord:
    id: int
    language: str
    category: str
    difficulty: int
    prompt: str
    test_cases: str
    created_at: str


@dataclass(frozen=True)
class SubmissionMetricsRecord:
    submission_id: int
    runtime_ms: int
    memory_usage_kb: int | None
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    created_at: str


@dataclass(frozen=True)
class SubmissionStaticAnalysisRecord:
    submission_id: int
    language: str
    pylint_score: float | None
    complexity_score: float | None
    security_warnings: list[dict[str, object]]
    pylint_warnings: list[dict[str, object]]
    created_at: str


DEFAULT_SKILL_PROFILE: dict[str, object] = {
    "version": 1,
    "language_scores": {},
    "category_scores": {},
    "recent_scores": [],
}
DEFAULT_SKILL_PROFILE_JSON = json.dumps(
    DEFAULT_SKILL_PROFILE,
    sort_keys=True,
    separators=(",", ":"),
)
DEFAULT_SKILL_PROFILE_SQL_JSON = (
    '{"category_scores":{},"language_scores":{},"recent_scores":[],"version":1}'
)

CREATE_SCHEMA_MIGRATIONS_TABLE_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_USERS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id BIGINT NOT NULL UNIQUE,
    preferred_language TEXT,
    skill_profile TEXT NOT NULL DEFAULT '{DEFAULT_SKILL_PROFILE_SQL_JSON}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_USER_SQLITE_SQL = """
INSERT OR IGNORE INTO users (telegram_id)
VALUES (?);
"""

SELECT_PREFERRED_LANGUAGE_SQLITE_SQL = """
SELECT preferred_language
FROM users
WHERE telegram_id = ?;
"""

SELECT_SKILL_PROFILE_SQLITE_SQL = """
SELECT skill_profile
FROM users
WHERE telegram_id = ?;
"""

UPDATE_PREFERRED_LANGUAGE_SQLITE_SQL = """
UPDATE users
SET preferred_language = ?
WHERE telegram_id = ?
  AND COALESCE(preferred_language, '') <> ?;
"""

UPDATE_USER_SKILL_PROFILE_IF_MISSING_SQLITE_SQL = """
UPDATE users
SET skill_profile = ?
WHERE telegram_id = ?
  AND (skill_profile IS NULL OR TRIM(skill_profile) = '');
"""

BOOTSTRAP_USERS_SKILL_PROFILE_SQLITE_SQL = """
UPDATE users
SET skill_profile = ?
WHERE skill_profile IS NULL
   OR TRIM(skill_profile) = '';
"""

CREATE_TASKS_TABLE_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty INTEGER NOT NULL CHECK (difficulty >= 1),
    prompt TEXT NOT NULL,
    test_cases TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_TASK_SQLITE_SQL = """
INSERT INTO tasks (language, category, difficulty, prompt, test_cases)
VALUES (?, ?, ?, ?, ?);
"""

SELECT_TASK_SQLITE_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases, created_at
FROM tasks
WHERE id = ?;
"""

SELECT_TASKS_SQLITE_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases, created_at
FROM tasks
ORDER BY id ASC
LIMIT ?;
"""

SELECT_TASKS_BY_LANGUAGE_SQLITE_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases, created_at
FROM tasks
WHERE language = ?
ORDER BY id ASC
LIMIT ?;
"""

UPDATE_TASK_SQLITE_SQL = """
UPDATE tasks
SET language = ?, category = ?, difficulty = ?, prompt = ?, test_cases = ?
WHERE id = ?;
"""

DELETE_TASK_SQLITE_SQL = """
DELETE FROM tasks
WHERE id = ?;
"""

CREATE_USERS_TABLE_POSTGRES_SQL = f"""
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL UNIQUE,
    preferred_language TEXT,
    skill_profile JSONB NOT NULL DEFAULT '{DEFAULT_SKILL_PROFILE_SQL_JSON}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

ALTER_USERS_TABLE_POSTGRES_SQL = """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS preferred_language TEXT;
"""

ALTER_USERS_TABLE_POSTGRES_ADD_SKILL_PROFILE_SQL = """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS skill_profile JSONB;
"""

BOOTSTRAP_USERS_SKILL_PROFILE_POSTGRES_SQL = f"""
UPDATE users
SET skill_profile = '{DEFAULT_SKILL_PROFILE_SQL_JSON}'::jsonb
WHERE skill_profile IS NULL;
"""

ALTER_USERS_TABLE_POSTGRES_ENFORCE_SKILL_PROFILE_SQL = f"""
ALTER TABLE users
ALTER COLUMN skill_profile SET DEFAULT '{DEFAULT_SKILL_PROFILE_SQL_JSON}'::jsonb;
"""

ALTER_USERS_TABLE_POSTGRES_ENFORCE_SKILL_PROFILE_NOT_NULL_SQL = """
ALTER TABLE users
ALTER COLUMN skill_profile SET NOT NULL;
"""

CREATE_SCHEMA_MIGRATIONS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SELECT_SCHEMA_MIGRATION_POSTGRES_SQL = """
SELECT version
FROM schema_migrations
WHERE version = %s;
"""

INSERT_SCHEMA_MIGRATION_POSTGRES_SQL = """
INSERT INTO schema_migrations (version)
VALUES (%s)
ON CONFLICT (version) DO NOTHING;
"""

SELECT_SCHEMA_MIGRATION_SQLITE_SQL = """
SELECT version
FROM schema_migrations
WHERE version = ?;
"""

INSERT_SCHEMA_MIGRATION_SQLITE_SQL = """
INSERT OR IGNORE INTO schema_migrations (version)
VALUES (?);
"""

ALTER_USERS_TABLE_SQLITE_ADD_SKILL_PROFILE_SQL = f"""
ALTER TABLE users
ADD COLUMN skill_profile TEXT NOT NULL DEFAULT '{DEFAULT_SKILL_PROFILE_SQL_JSON}';
"""

INSERT_USER_POSTGRES_SQL = """
INSERT INTO users (telegram_id)
VALUES (%s)
ON CONFLICT (telegram_id) DO NOTHING;
"""

CREATE_SUBMISSIONS_TABLE_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id BIGINT NOT NULL,
    task_id INTEGER,
    language TEXT NOT NULL,
    code TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (telegram_id) REFERENCES users(telegram_id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE SET NULL
);
"""

INSERT_SUBMISSION_SQLITE_SQL = """
INSERT INTO submissions (telegram_id, task_id, language, code)
VALUES (?, ?, ?, ?);
"""

CREATE_SUBMISSION_METRICS_TABLE_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS submission_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL UNIQUE,
    runtime_ms INTEGER NOT NULL CHECK (runtime_ms >= 0),
    memory_usage_kb INTEGER,
    exit_code INTEGER,
    stdout TEXT NOT NULL,
    stderr TEXT NOT NULL,
    timed_out INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
);
"""

INSERT_SUBMISSION_METRICS_SQLITE_SQL = """
INSERT INTO submission_metrics (
    submission_id,
    runtime_ms,
    memory_usage_kb,
    exit_code,
    stdout,
    stderr,
    timed_out
)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(submission_id) DO UPDATE SET
    runtime_ms = excluded.runtime_ms,
    memory_usage_kb = excluded.memory_usage_kb,
    exit_code = excluded.exit_code,
    stdout = excluded.stdout,
    stderr = excluded.stderr,
    timed_out = excluded.timed_out;
"""

SELECT_SUBMISSION_METRICS_SQLITE_SQL = """
SELECT submission_id, runtime_ms, memory_usage_kb, exit_code, stdout, stderr, timed_out, created_at
FROM submission_metrics
WHERE submission_id = ?;
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_TABLE_SQLITE_SQL = """
CREATE TABLE IF NOT EXISTS submission_static_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL UNIQUE,
    language TEXT NOT NULL,
    pylint_score REAL,
    complexity_score REAL,
    security_warnings TEXT NOT NULL DEFAULT '[]',
    pylint_warnings TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
);
"""

INSERT_SUBMISSION_STATIC_ANALYSIS_SQLITE_SQL = """
INSERT INTO submission_static_analysis (
    submission_id,
    language,
    pylint_score,
    complexity_score,
    security_warnings,
    pylint_warnings
)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(submission_id) DO UPDATE SET
    language = excluded.language,
    pylint_score = excluded.pylint_score,
    complexity_score = excluded.complexity_score,
    security_warnings = excluded.security_warnings,
    pylint_warnings = excluded.pylint_warnings;
"""

SELECT_SUBMISSION_STATIC_ANALYSIS_SQLITE_SQL = """
SELECT
    submission_id,
    language,
    pylint_score,
    complexity_score,
    security_warnings,
    pylint_warnings,
    created_at
FROM submission_static_analysis
WHERE submission_id = ?;
"""

CREATE_SUBMISSIONS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS submissions (
    id BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    task_id BIGINT REFERENCES tasks(id) ON DELETE SET NULL,
    language TEXT NOT NULL,
    code TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_SUBMISSION_POSTGRES_SQL = """
INSERT INTO submissions (telegram_id, task_id, language, code)
VALUES (%s, %s, %s, %s)
RETURNING id;
"""

CREATE_SUBMISSION_METRICS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS submission_metrics (
    id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL UNIQUE REFERENCES submissions(id) ON DELETE CASCADE,
    runtime_ms INTEGER NOT NULL CHECK (runtime_ms >= 0),
    memory_usage_kb INTEGER,
    exit_code INTEGER,
    stdout TEXT NOT NULL,
    stderr TEXT NOT NULL,
    timed_out BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_SUBMISSION_METRICS_POSTGRES_SQL = """
INSERT INTO submission_metrics (
    submission_id,
    runtime_ms,
    memory_usage_kb,
    exit_code,
    stdout,
    stderr,
    timed_out
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (submission_id) DO UPDATE SET
    runtime_ms = EXCLUDED.runtime_ms,
    memory_usage_kb = EXCLUDED.memory_usage_kb,
    exit_code = EXCLUDED.exit_code,
    stdout = EXCLUDED.stdout,
    stderr = EXCLUDED.stderr,
    timed_out = EXCLUDED.timed_out;
"""

SELECT_SUBMISSION_METRICS_POSTGRES_SQL = """
SELECT submission_id, runtime_ms, memory_usage_kb, exit_code, stdout, stderr, timed_out, created_at
FROM submission_metrics
WHERE submission_id = %s;
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS submission_static_analysis (
    id BIGSERIAL PRIMARY KEY,
    submission_id BIGINT NOT NULL UNIQUE REFERENCES submissions(id) ON DELETE CASCADE,
    language TEXT NOT NULL,
    pylint_score DOUBLE PRECISION,
    complexity_score DOUBLE PRECISION,
    security_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    pylint_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_SUBMISSION_STATIC_ANALYSIS_POSTGRES_SQL = """
INSERT INTO submission_static_analysis (
    submission_id,
    language,
    pylint_score,
    complexity_score,
    security_warnings,
    pylint_warnings
)
VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
ON CONFLICT (submission_id) DO UPDATE SET
    language = EXCLUDED.language,
    pylint_score = EXCLUDED.pylint_score,
    complexity_score = EXCLUDED.complexity_score,
    security_warnings = EXCLUDED.security_warnings,
    pylint_warnings = EXCLUDED.pylint_warnings;
"""

SELECT_SUBMISSION_STATIC_ANALYSIS_POSTGRES_SQL = """
SELECT
    submission_id,
    language,
    pylint_score,
    complexity_score,
    security_warnings::text,
    pylint_warnings::text,
    created_at
FROM submission_static_analysis
WHERE submission_id = %s;
"""

ALTER_SUBMISSIONS_TABLE_POSTGRES_ADD_TASK_ID_SQL = """
ALTER TABLE submissions
ADD COLUMN IF NOT EXISTS task_id BIGINT;
"""

ALTER_SUBMISSIONS_TABLE_SQLITE_ADD_TASK_ID_SQL = """
ALTER TABLE submissions
ADD COLUMN task_id INTEGER;
"""

CREATE_USERS_CREATED_AT_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_users_created_at
ON users(created_at);
"""

CREATE_TASKS_LANGUAGE_DIFFICULTY_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_tasks_language_difficulty
ON tasks(language, difficulty);
"""

CREATE_TASKS_CATEGORY_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_tasks_category
ON tasks(category);
"""

CREATE_SUBMISSIONS_TELEGRAM_CREATED_AT_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submissions_telegram_created_at
ON submissions(telegram_id, created_at);
"""

CREATE_SUBMISSIONS_TASK_CREATED_AT_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submissions_task_created_at
ON submissions(task_id, created_at);
"""

CREATE_SUBMISSION_METRICS_SUBMISSION_ID_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_metrics_submission_id
ON submission_metrics(submission_id);
"""

CREATE_SUBMISSION_METRICS_CREATED_AT_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_metrics_created_at
ON submission_metrics(created_at);
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_SUBMISSION_ID_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_static_analysis_submission_id
ON submission_static_analysis(submission_id);
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_CREATED_AT_INDEX_SQLITE_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_static_analysis_created_at
ON submission_static_analysis(created_at);
"""

CREATE_TASKS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id BIGSERIAL PRIMARY KEY,
    language TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty INTEGER NOT NULL CHECK (difficulty >= 1),
    prompt TEXT NOT NULL,
    test_cases JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_TASK_POSTGRES_SQL = """
INSERT INTO tasks (language, category, difficulty, prompt, test_cases)
VALUES (%s, %s, %s, %s, %s::jsonb)
RETURNING id;
"""

SELECT_TASK_POSTGRES_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases::text, created_at
FROM tasks
WHERE id = %s;
"""

SELECT_TASKS_POSTGRES_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases::text, created_at
FROM tasks
ORDER BY id ASC
LIMIT %s;
"""

SELECT_TASKS_BY_LANGUAGE_POSTGRES_SQL = """
SELECT id, language, category, difficulty, prompt, test_cases::text, created_at
FROM tasks
WHERE language = %s
ORDER BY id ASC
LIMIT %s;
"""

UPDATE_TASK_POSTGRES_SQL = """
UPDATE tasks
SET language = %s, category = %s, difficulty = %s, prompt = %s, test_cases = %s::jsonb
WHERE id = %s;
"""

DELETE_TASK_POSTGRES_SQL = """
DELETE FROM tasks
WHERE id = %s;
"""

SELECT_TASK_EXISTS_SQLITE_SQL = """
SELECT id
FROM tasks
WHERE id = ?;
"""

SELECT_TASK_EXISTS_POSTGRES_SQL = """
SELECT id
FROM tasks
WHERE id = %s;
"""

SELECT_SUBMISSION_EXISTS_SQLITE_SQL = """
SELECT id
FROM submissions
WHERE id = ?;
"""

SELECT_SUBMISSION_EXISTS_POSTGRES_SQL = """
SELECT id
FROM submissions
WHERE id = %s;
"""

CREATE_USERS_CREATED_AT_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_users_created_at
ON users(created_at);
"""

CREATE_TASKS_LANGUAGE_DIFFICULTY_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_tasks_language_difficulty
ON tasks(language, difficulty);
"""

CREATE_TASKS_CATEGORY_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_tasks_category
ON tasks(category);
"""

CREATE_SUBMISSIONS_TELEGRAM_CREATED_AT_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submissions_telegram_created_at
ON submissions(telegram_id, created_at);
"""

CREATE_SUBMISSIONS_TASK_CREATED_AT_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submissions_task_created_at
ON submissions(task_id, created_at);
"""

CREATE_SUBMISSION_METRICS_SUBMISSION_ID_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_metrics_submission_id
ON submission_metrics(submission_id);
"""

CREATE_SUBMISSION_METRICS_CREATED_AT_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_metrics_created_at
ON submission_metrics(created_at);
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_SUBMISSION_ID_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_static_analysis_submission_id
ON submission_static_analysis(submission_id);
"""

CREATE_SUBMISSION_STATIC_ANALYSIS_CREATED_AT_INDEX_POSTGRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_submission_static_analysis_created_at
ON submission_static_analysis(created_at);
"""

SELECT_PREFERRED_LANGUAGE_POSTGRES_SQL = """
SELECT preferred_language
FROM users
WHERE telegram_id = %s;
"""

SELECT_SKILL_PROFILE_POSTGRES_SQL = """
SELECT skill_profile::text
FROM users
WHERE telegram_id = %s;
"""

UPDATE_USER_SKILL_PROFILE_IF_MISSING_POSTGRES_SQL = """
UPDATE users
SET skill_profile = %s::jsonb
WHERE telegram_id = %s
  AND skill_profile IS NULL;
"""

UPDATE_PREFERRED_LANGUAGE_POSTGRES_SQL = """
UPDATE users
SET preferred_language = %s
WHERE telegram_id = %s
  AND preferred_language IS DISTINCT FROM %s;
"""


class UserRepository(Protocol):
    async def ensure_schema(self) -> None: ...

    async def register_user(self, telegram_id: int) -> bool: ...

    async def get_preferred_language(self, telegram_id: int) -> str | None: ...

    async def get_skill_profile(self, telegram_id: int) -> dict[str, object] | None: ...

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool: ...

    async def save_submission(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> bool: ...

    async def save_submission_with_id(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> int | None: ...

    async def save_submission_metrics(
        self,
        submission_id: int,
        runtime_ms: int,
        memory_usage_kb: int | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> bool: ...

    async def get_submission_metrics(
        self, submission_id: int
    ) -> SubmissionMetricsRecord | None: ...

    async def save_submission_static_analysis(
        self,
        submission_id: int,
        language: str,
        pylint_score: float | None,
        complexity_score: float | None,
        security_warnings: list[dict[str, object]],
        pylint_warnings: list[dict[str, object]],
    ) -> bool: ...

    async def get_submission_static_analysis(
        self,
        submission_id: int,
    ) -> SubmissionStaticAnalysisRecord | None: ...

    async def create_task(
        self,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> int: ...

    async def get_task(self, task_id: int) -> TaskRecord | None: ...

    async def list_tasks(
        self, language: str | None = None, limit: int = 20
    ) -> list[TaskRecord]: ...

    async def update_task(
        self,
        task_id: int,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> bool: ...

    async def delete_task(self, task_id: int) -> bool: ...


class SQLiteUserRepository:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    async def ensure_schema(self) -> None:
        await asyncio.to_thread(self._ensure_schema_sync)

    async def register_user(self, telegram_id: int) -> bool:
        return await asyncio.to_thread(self._register_user_sync, telegram_id)

    async def get_preferred_language(self, telegram_id: int) -> str | None:
        return await asyncio.to_thread(self._get_preferred_language_sync, telegram_id)

    async def get_skill_profile(self, telegram_id: int) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_skill_profile_sync, telegram_id)

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool:
        return await asyncio.to_thread(
            self._set_preferred_language_sync,
            telegram_id,
            preferred_language,
        )

    async def save_submission(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._save_submission_sync, telegram_id, code, task_id)

    async def save_submission_with_id(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> int | None:
        return await asyncio.to_thread(
            self._save_submission_with_id_sync,
            telegram_id,
            code,
            task_id,
        )

    async def save_submission_metrics(
        self,
        submission_id: int,
        runtime_ms: int,
        memory_usage_kb: int | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> bool:
        return await asyncio.to_thread(
            self._save_submission_metrics_sync,
            submission_id,
            runtime_ms,
            memory_usage_kb,
            exit_code,
            stdout,
            stderr,
            timed_out,
        )

    async def get_submission_metrics(self, submission_id: int) -> SubmissionMetricsRecord | None:
        return await asyncio.to_thread(self._get_submission_metrics_sync, submission_id)

    async def save_submission_static_analysis(
        self,
        submission_id: int,
        language: str,
        pylint_score: float | None,
        complexity_score: float | None,
        security_warnings: list[dict[str, object]],
        pylint_warnings: list[dict[str, object]],
    ) -> bool:
        return await asyncio.to_thread(
            self._save_submission_static_analysis_sync,
            submission_id,
            language,
            pylint_score,
            complexity_score,
            security_warnings,
            pylint_warnings,
        )

    async def get_submission_static_analysis(
        self,
        submission_id: int,
    ) -> SubmissionStaticAnalysisRecord | None:
        return await asyncio.to_thread(self._get_submission_static_analysis_sync, submission_id)

    async def create_task(
        self,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> int:
        return await asyncio.to_thread(
            self._create_task_sync,
            language,
            category,
            difficulty,
            prompt,
            test_cases,
        )

    async def get_task(self, task_id: int) -> TaskRecord | None:
        return await asyncio.to_thread(self._get_task_sync, task_id)

    async def list_tasks(self, language: str | None = None, limit: int = 20) -> list[TaskRecord]:
        return await asyncio.to_thread(self._list_tasks_sync, language, limit)

    async def update_task(
        self,
        task_id: int,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._update_task_sync,
            task_id,
            language,
            category,
            difficulty,
            prompt,
            test_cases,
        )

    async def delete_task(self, task_id: int) -> bool:
        return await asyncio.to_thread(self._delete_task_sync, task_id)

    def _ensure_schema_sync(self) -> None:
        if self._database_path != ":memory:":
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            connection.execute(CREATE_SCHEMA_MIGRATIONS_TABLE_SQLITE_SQL)

            if not self._is_sqlite_migration_applied(connection, "0001_create_users"):
                connection.execute(CREATE_USERS_TABLE_SQL)
                self._mark_sqlite_migration_applied(connection, "0001_create_users")

            if not self._is_sqlite_migration_applied(connection, "0002_add_preferred_language"):
                if not self._has_column(connection, "users", "preferred_language"):
                    connection.execute("ALTER TABLE users ADD COLUMN preferred_language TEXT;")
                self._mark_sqlite_migration_applied(connection, "0002_add_preferred_language")

            if not self._is_sqlite_migration_applied(connection, "0003_create_tasks"):
                connection.execute(CREATE_TASKS_TABLE_SQLITE_SQL)
                self._mark_sqlite_migration_applied(connection, "0003_create_tasks")

            if not self._is_sqlite_migration_applied(connection, "0004_create_submissions"):
                connection.execute(CREATE_SUBMISSIONS_TABLE_SQLITE_SQL)
                self._mark_sqlite_migration_applied(connection, "0004_create_submissions")

            if not self._is_sqlite_migration_applied(connection, "0005_add_task_id_to_submissions"):
                if not self._has_column(connection, "submissions", "task_id"):
                    connection.execute(ALTER_SUBMISSIONS_TABLE_SQLITE_ADD_TASK_ID_SQL)
                self._mark_sqlite_migration_applied(connection, "0005_add_task_id_to_submissions")

            if not self._is_sqlite_migration_applied(connection, "0006_create_indexes"):
                connection.execute(CREATE_USERS_CREATED_AT_INDEX_SQLITE_SQL)
                connection.execute(CREATE_TASKS_LANGUAGE_DIFFICULTY_INDEX_SQLITE_SQL)
                connection.execute(CREATE_TASKS_CATEGORY_INDEX_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSIONS_TELEGRAM_CREATED_AT_INDEX_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSIONS_TASK_CREATED_AT_INDEX_SQLITE_SQL)
                self._mark_sqlite_migration_applied(connection, "0006_create_indexes")

            if not self._is_sqlite_migration_applied(connection, "0007_add_skill_profile"):
                if not self._has_column(connection, "users", "skill_profile"):
                    connection.execute(ALTER_USERS_TABLE_SQLITE_ADD_SKILL_PROFILE_SQL)
                connection.execute(
                    BOOTSTRAP_USERS_SKILL_PROFILE_SQLITE_SQL,
                    (DEFAULT_SKILL_PROFILE_JSON,),
                )
                self._mark_sqlite_migration_applied(connection, "0007_add_skill_profile")

            if not self._is_sqlite_migration_applied(connection, "0008_create_submission_metrics"):
                connection.execute(CREATE_SUBMISSION_METRICS_TABLE_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSION_METRICS_SUBMISSION_ID_INDEX_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSION_METRICS_CREATED_AT_INDEX_SQLITE_SQL)
                self._mark_sqlite_migration_applied(connection, "0008_create_submission_metrics")

            if not self._is_sqlite_migration_applied(
                connection,
                "0009_create_submission_static_analysis",
            ):
                connection.execute(CREATE_SUBMISSION_STATIC_ANALYSIS_TABLE_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSION_STATIC_ANALYSIS_SUBMISSION_ID_INDEX_SQLITE_SQL)
                connection.execute(CREATE_SUBMISSION_STATIC_ANALYSIS_CREATED_AT_INDEX_SQLITE_SQL)
                self._mark_sqlite_migration_applied(
                    connection,
                    "0009_create_submission_static_analysis",
                )

            connection.execute(
                BOOTSTRAP_USERS_SKILL_PROFILE_SQLITE_SQL,
                (DEFAULT_SKILL_PROFILE_JSON,),
            )
            connection.commit()

    def _register_user_sync(self, telegram_id: int) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            insert_cursor = connection.execute(INSERT_USER_SQLITE_SQL, (telegram_id,))
            connection.execute(
                UPDATE_USER_SKILL_PROFILE_IF_MISSING_SQLITE_SQL,
                (DEFAULT_SKILL_PROFILE_JSON, telegram_id),
            )
            connection.commit()
        return insert_cursor.rowcount == 1

    def _get_preferred_language_sync(self, telegram_id: int) -> str | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(
                SELECT_PREFERRED_LANGUAGE_SQLITE_SQL,
                (telegram_id,),
            ).fetchone()

        if row is None:
            return None

        value = row[0]
        if value is None:
            return None
        return str(value)

    def _get_skill_profile_sync(self, telegram_id: int) -> dict[str, object] | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(
                SELECT_SKILL_PROFILE_SQLITE_SQL,
                (telegram_id,),
            ).fetchone()

        if row is None:
            return None

        return _decode_skill_profile(row[0])

    def _set_preferred_language_sync(self, telegram_id: int, preferred_language: str) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute(INSERT_USER_SQLITE_SQL, (telegram_id,))
            cursor = connection.execute(
                UPDATE_PREFERRED_LANGUAGE_SQLITE_SQL,
                (preferred_language, telegram_id, preferred_language),
            )
            connection.commit()

        return cursor.rowcount == 1

    def _save_submission_sync(self, telegram_id: int, code: str, task_id: int | None) -> bool:
        submission_id = self._save_submission_with_id_sync(telegram_id, code, task_id)
        return submission_id is not None

    def _save_submission_with_id_sync(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None,
    ) -> int | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            language = connection.execute(
                SELECT_PREFERRED_LANGUAGE_SQLITE_SQL,
                (telegram_id,),
            ).fetchone()
            if language is None or language[0] is None:
                return None

            if task_id is not None:
                task = connection.execute(SELECT_TASK_EXISTS_SQLITE_SQL, (task_id,)).fetchone()
                if task is None:
                    return None

            cursor = connection.execute(
                INSERT_SUBMISSION_SQLITE_SQL,
                (telegram_id, task_id, str(language[0]), code),
            )
            connection.commit()
        if cursor.lastrowid is None:
            return None
        return int(cursor.lastrowid)

    def _save_submission_metrics_sync(
        self,
        submission_id: int,
        runtime_ms: int,
        memory_usage_kb: int | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            submission = connection.execute(
                SELECT_SUBMISSION_EXISTS_SQLITE_SQL,
                (submission_id,),
            ).fetchone()
            if submission is None:
                return False
            cursor = connection.execute(
                INSERT_SUBMISSION_METRICS_SQLITE_SQL,
                (
                    submission_id,
                    runtime_ms,
                    memory_usage_kb,
                    exit_code,
                    stdout,
                    stderr,
                    int(timed_out),
                ),
            )
            connection.commit()
        return cursor.rowcount >= 1

    def _get_submission_metrics_sync(self, submission_id: int) -> SubmissionMetricsRecord | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(
                SELECT_SUBMISSION_METRICS_SQLITE_SQL,
                (submission_id,),
            ).fetchone()

        if row is None:
            return None

        return _row_to_submission_metrics(tuple(row))

    def _save_submission_static_analysis_sync(
        self,
        submission_id: int,
        language: str,
        pylint_score: float | None,
        complexity_score: float | None,
        security_warnings: list[dict[str, object]],
        pylint_warnings: list[dict[str, object]],
    ) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            submission = connection.execute(
                SELECT_SUBMISSION_EXISTS_SQLITE_SQL,
                (submission_id,),
            ).fetchone()
            if submission is None:
                return False

            cursor = connection.execute(
                INSERT_SUBMISSION_STATIC_ANALYSIS_SQLITE_SQL,
                (
                    submission_id,
                    language,
                    pylint_score,
                    complexity_score,
                    _encode_json_payload(security_warnings),
                    _encode_json_payload(pylint_warnings),
                ),
            )
            connection.commit()
        return cursor.rowcount >= 1

    def _get_submission_static_analysis_sync(
        self,
        submission_id: int,
    ) -> SubmissionStaticAnalysisRecord | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(
                SELECT_SUBMISSION_STATIC_ANALYSIS_SQLITE_SQL,
                (submission_id,),
            ).fetchone()

        if row is None:
            return None

        return _row_to_submission_static_analysis(tuple(row))

    def _create_task_sync(
        self,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> int:
        with closing(sqlite3.connect(self._database_path)) as connection:
            cursor = connection.execute(
                INSERT_TASK_SQLITE_SQL,
                (language, category, difficulty, prompt, test_cases),
            )
            connection.commit()

        if cursor.lastrowid is None:
            return 0
        return int(cursor.lastrowid)

    def _get_task_sync(self, task_id: int) -> TaskRecord | None:
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(SELECT_TASK_SQLITE_SQL, (task_id,)).fetchone()

        if row is None:
            return None

        return _row_to_task(tuple(row))

    def _list_tasks_sync(self, language: str | None, limit: int) -> list[TaskRecord]:
        normalized_limit = max(1, limit)

        with closing(sqlite3.connect(self._database_path)) as connection:
            if language is None:
                rows = connection.execute(SELECT_TASKS_SQLITE_SQL, (normalized_limit,)).fetchall()
            else:
                rows = connection.execute(
                    SELECT_TASKS_BY_LANGUAGE_SQLITE_SQL,
                    (language, normalized_limit),
                ).fetchall()

        return [_row_to_task(tuple(row)) for row in rows]

    def _update_task_sync(
        self,
        task_id: int,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            cursor = connection.execute(
                UPDATE_TASK_SQLITE_SQL,
                (language, category, difficulty, prompt, test_cases, task_id),
            )
            connection.commit()

        return cursor.rowcount == 1

    def _delete_task_sync(self, task_id: int) -> bool:
        with closing(sqlite3.connect(self._database_path)) as connection:
            cursor = connection.execute(DELETE_TASK_SQLITE_SQL, (task_id,))
            connection.commit()

        return cursor.rowcount == 1

    @staticmethod
    def _has_column(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        rows = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
        return any(str(row[1]) == column_name for row in rows)

    @staticmethod
    def _is_sqlite_migration_applied(connection: sqlite3.Connection, version: str) -> bool:
        row = connection.execute(SELECT_SCHEMA_MIGRATION_SQLITE_SQL, (version,)).fetchone()
        return row is not None

    @staticmethod
    def _mark_sqlite_migration_applied(connection: sqlite3.Connection, version: str) -> None:
        connection.execute(INSERT_SCHEMA_MIGRATION_SQLITE_SQL, (version,))


class PostgresUserRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = normalize_database_url(database_url)

    async def ensure_schema(self) -> None:
        await self._execute(CREATE_SCHEMA_MIGRATIONS_TABLE_POSTGRES_SQL)

        await self._apply_postgres_migration(
            "0001_create_users",
            (CREATE_USERS_TABLE_POSTGRES_SQL,),
        )
        await self._apply_postgres_migration(
            "0002_add_preferred_language",
            (ALTER_USERS_TABLE_POSTGRES_SQL,),
        )
        await self._apply_postgres_migration(
            "0003_create_tasks",
            (CREATE_TASKS_TABLE_POSTGRES_SQL,),
        )
        await self._apply_postgres_migration(
            "0004_create_submissions",
            (CREATE_SUBMISSIONS_TABLE_POSTGRES_SQL,),
        )
        await self._apply_postgres_migration(
            "0005_add_task_id_to_submissions",
            (ALTER_SUBMISSIONS_TABLE_POSTGRES_ADD_TASK_ID_SQL,),
        )
        await self._apply_postgres_migration(
            "0006_create_indexes",
            (
                CREATE_USERS_CREATED_AT_INDEX_POSTGRES_SQL,
                CREATE_TASKS_LANGUAGE_DIFFICULTY_INDEX_POSTGRES_SQL,
                CREATE_TASKS_CATEGORY_INDEX_POSTGRES_SQL,
                CREATE_SUBMISSIONS_TELEGRAM_CREATED_AT_INDEX_POSTGRES_SQL,
                CREATE_SUBMISSIONS_TASK_CREATED_AT_INDEX_POSTGRES_SQL,
            ),
        )
        await self._apply_postgres_migration(
            "0007_add_skill_profile",
            (
                ALTER_USERS_TABLE_POSTGRES_ADD_SKILL_PROFILE_SQL,
                BOOTSTRAP_USERS_SKILL_PROFILE_POSTGRES_SQL,
                ALTER_USERS_TABLE_POSTGRES_ENFORCE_SKILL_PROFILE_SQL,
                ALTER_USERS_TABLE_POSTGRES_ENFORCE_SKILL_PROFILE_NOT_NULL_SQL,
            ),
        )
        await self._apply_postgres_migration(
            "0008_create_submission_metrics",
            (
                CREATE_SUBMISSION_METRICS_TABLE_POSTGRES_SQL,
                CREATE_SUBMISSION_METRICS_SUBMISSION_ID_INDEX_POSTGRES_SQL,
                CREATE_SUBMISSION_METRICS_CREATED_AT_INDEX_POSTGRES_SQL,
            ),
        )
        await self._apply_postgres_migration(
            "0009_create_submission_static_analysis",
            (
                CREATE_SUBMISSION_STATIC_ANALYSIS_TABLE_POSTGRES_SQL,
                CREATE_SUBMISSION_STATIC_ANALYSIS_SUBMISSION_ID_INDEX_POSTGRES_SQL,
                CREATE_SUBMISSION_STATIC_ANALYSIS_CREATED_AT_INDEX_POSTGRES_SQL,
            ),
        )
        await self._execute(BOOTSTRAP_USERS_SKILL_PROFILE_POSTGRES_SQL)

    async def register_user(self, telegram_id: int) -> bool:
        status = await self._execute(INSERT_USER_POSTGRES_SQL, (telegram_id,))
        await self._execute(
            UPDATE_USER_SKILL_PROFILE_IF_MISSING_POSTGRES_SQL,
            (DEFAULT_SKILL_PROFILE_JSON, telegram_id),
        )
        return status == "INSERT 0 1"

    async def get_preferred_language(self, telegram_id: int) -> str | None:
        row = await self._fetch_one(SELECT_PREFERRED_LANGUAGE_POSTGRES_SQL, (telegram_id,))
        if row is None:
            return None

        value = row[0]
        if value is None:
            return None
        return str(value)

    async def get_skill_profile(self, telegram_id: int) -> dict[str, object] | None:
        row = await self._fetch_one(SELECT_SKILL_PROFILE_POSTGRES_SQL, (telegram_id,))
        if row is None:
            return None

        return _decode_skill_profile(row[0])

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool:
        await self._execute(INSERT_USER_POSTGRES_SQL, (telegram_id,))
        status = await self._execute(
            UPDATE_PREFERRED_LANGUAGE_POSTGRES_SQL,
            (preferred_language, telegram_id, preferred_language),
        )
        return status == "UPDATE 1"

    async def save_submission(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> bool:
        submission_id = await self.save_submission_with_id(
            telegram_id=telegram_id,
            code=code,
            task_id=task_id,
        )
        return submission_id is not None

    async def save_submission_with_id(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> int | None:
        preferred_language = await self.get_preferred_language(telegram_id)
        if preferred_language is None:
            return None

        if task_id is not None:
            task = await self._fetch_one(SELECT_TASK_EXISTS_POSTGRES_SQL, (task_id,))
            if task is None:
                return None

        row = await self._fetch_one(
            INSERT_SUBMISSION_POSTGRES_SQL,
            (telegram_id, task_id, preferred_language, code),
        )
        if row is None:
            return None
        return _coerce_int(row[0], "submission.id")

    async def save_submission_metrics(
        self,
        submission_id: int,
        runtime_ms: int,
        memory_usage_kb: int | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> bool:
        submission = await self._fetch_one(SELECT_SUBMISSION_EXISTS_POSTGRES_SQL, (submission_id,))
        if submission is None:
            return False

        status = await self._execute(
            INSERT_SUBMISSION_METRICS_POSTGRES_SQL,
            (
                submission_id,
                runtime_ms,
                memory_usage_kb,
                exit_code,
                stdout,
                stderr,
                timed_out,
            ),
        )
        return status in {"INSERT 0 1", "UPDATE 1"}

    async def get_submission_metrics(self, submission_id: int) -> SubmissionMetricsRecord | None:
        row = await self._fetch_one(SELECT_SUBMISSION_METRICS_POSTGRES_SQL, (submission_id,))
        if row is None:
            return None
        return _row_to_submission_metrics(row)

    async def save_submission_static_analysis(
        self,
        submission_id: int,
        language: str,
        pylint_score: float | None,
        complexity_score: float | None,
        security_warnings: list[dict[str, object]],
        pylint_warnings: list[dict[str, object]],
    ) -> bool:
        submission = await self._fetch_one(SELECT_SUBMISSION_EXISTS_POSTGRES_SQL, (submission_id,))
        if submission is None:
            return False

        status = await self._execute(
            INSERT_SUBMISSION_STATIC_ANALYSIS_POSTGRES_SQL,
            (
                submission_id,
                language,
                pylint_score,
                complexity_score,
                _encode_json_payload(security_warnings),
                _encode_json_payload(pylint_warnings),
            ),
        )
        return status in {"INSERT 0 1", "UPDATE 1"}

    async def get_submission_static_analysis(
        self,
        submission_id: int,
    ) -> SubmissionStaticAnalysisRecord | None:
        row = await self._fetch_one(
            SELECT_SUBMISSION_STATIC_ANALYSIS_POSTGRES_SQL,
            (submission_id,),
        )
        if row is None:
            return None
        return _row_to_submission_static_analysis(row)

    async def create_task(
        self,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> int:
        row = await self._fetch_one(
            INSERT_TASK_POSTGRES_SQL,
            (language, category, difficulty, prompt, test_cases),
        )
        if row is None:
            return 0
        return _coerce_int(row[0], "task.id")

    async def get_task(self, task_id: int) -> TaskRecord | None:
        row = await self._fetch_one(SELECT_TASK_POSTGRES_SQL, (task_id,))
        if row is None:
            return None
        return _row_to_task(row)

    async def list_tasks(self, language: str | None = None, limit: int = 20) -> list[TaskRecord]:
        normalized_limit = max(1, limit)
        if language is None:
            rows = await self._fetch_all(SELECT_TASKS_POSTGRES_SQL, (normalized_limit,))
        else:
            rows = await self._fetch_all(
                SELECT_TASKS_BY_LANGUAGE_POSTGRES_SQL,
                (language, normalized_limit),
            )
        return [_row_to_task(row) for row in rows]

    async def update_task(
        self,
        task_id: int,
        language: str,
        category: str,
        difficulty: int,
        prompt: str,
        test_cases: str,
    ) -> bool:
        status = await self._execute(
            UPDATE_TASK_POSTGRES_SQL,
            (language, category, difficulty, prompt, test_cases, task_id),
        )
        return status == "UPDATE 1"

    async def delete_task(self, task_id: int) -> bool:
        status = await self._execute(DELETE_TASK_POSTGRES_SQL, (task_id,))
        return status == "DELETE 1"

    async def _apply_postgres_migration(
        self,
        version: str,
        statements: tuple[str, ...],
    ) -> None:
        if await self._is_postgres_migration_applied(version):
            return

        for statement in statements:
            await self._execute(statement)
        await self._execute(INSERT_SCHEMA_MIGRATION_POSTGRES_SQL, (version,))

    async def _is_postgres_migration_applied(self, version: str) -> bool:
        row = await self._fetch_one(SELECT_SCHEMA_MIGRATION_POSTGRES_SQL, (version,))
        return row is not None

    async def _execute(self, query: str, params: tuple[object, ...] = ()) -> str:
        psycopg = _import_psycopg()

        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                status = cursor.statusmessage
            await connection.commit()

        return status

    async def _fetch_one(
        self,
        query: str,
        params: tuple[object, ...] = (),
    ) -> tuple[object, ...] | None:
        psycopg = _import_psycopg()

        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                row = await cursor.fetchone()

        if row is None:
            return None

        return tuple(row)

    async def _fetch_all(
        self,
        query: str,
        params: tuple[object, ...] = (),
    ) -> list[tuple[object, ...]]:
        psycopg = _import_psycopg()

        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                rows = await cursor.fetchall()

        return [tuple(row) for row in rows]


def _import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Add `psycopg[binary]` to run Postgres-backed bot storage."
        ) from exc

    return psycopg


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


def _coerce_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer-compatible value.")
    if isinstance(value, str):
        return int(value.strip())

    raise TypeError(f"{field_name} must be int/float/str, got {type(value).__name__}.")


def _coerce_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())

    raise TypeError(f"{field_name} must be int/float/str, got {type(value).__name__}.")


def _row_to_task(row: tuple[object, ...]) -> TaskRecord:
    return TaskRecord(
        id=_coerce_int(row[0], "task.id"),
        language=str(row[1]),
        category=str(row[2]),
        difficulty=_coerce_int(row[3], "task.difficulty"),
        prompt=str(row[4]),
        test_cases=str(row[5]),
        created_at=str(row[6]),
    )


def _row_to_submission_metrics(row: tuple[object, ...]) -> SubmissionMetricsRecord:
    return SubmissionMetricsRecord(
        submission_id=_coerce_int(row[0], "submission_metrics.submission_id"),
        runtime_ms=_coerce_int(row[1], "submission_metrics.runtime_ms"),
        memory_usage_kb=(
            _coerce_int(row[2], "submission_metrics.memory_usage_kb")
            if row[2] is not None
            else None
        ),
        exit_code=(
            _coerce_int(row[3], "submission_metrics.exit_code") if row[3] is not None else None
        ),
        stdout=str(row[4]),
        stderr=str(row[5]),
        timed_out=bool(row[6]),
        created_at=str(row[7]),
    )


def _row_to_submission_static_analysis(
    row: tuple[object, ...],
) -> SubmissionStaticAnalysisRecord:
    return SubmissionStaticAnalysisRecord(
        submission_id=_coerce_int(row[0], "submission_static_analysis.submission_id"),
        language=str(row[1]),
        pylint_score=(
            _coerce_float(row[2], "submission_static_analysis.pylint_score")
            if row[2] is not None
            else None
        ),
        complexity_score=(
            _coerce_float(row[3], "submission_static_analysis.complexity_score")
            if row[3] is not None
            else None
        ),
        security_warnings=_decode_json_array_payload(row[4]),
        pylint_warnings=_decode_json_array_payload(row[5]),
        created_at=str(row[6]),
    )


def _encode_json_payload(value: list[dict[str, object]]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_json_array_payload(value: object) -> list[dict[str, object]]:
    if value is None:
        return []

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    normalized_items: list[dict[str, object]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        normalized_items.append({str(key): item[key] for key in item})
    return normalized_items


def _decode_skill_profile(value: object) -> dict[str, object] | None:
    if value is None:
        return None

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return {str(key): parsed[key] for key in parsed}


def build_user_repository(database_url: str) -> UserRepository:
    normalized_url = normalize_database_url(database_url)
    if normalized_url.startswith("postgresql://") or normalized_url.startswith("postgres://"):
        return PostgresUserRepository(normalized_url)

    if normalized_url.startswith("sqlite:///"):
        database_path = normalized_url.removeprefix("sqlite:///")
        if database_path == "":
            raise ValueError("DATABASE_URL for sqlite must include a database path.")
        return SQLiteUserRepository(database_path)

    raise ValueError(
        "Unsupported DATABASE_URL scheme. Use postgresql://... or sqlite:///path/to/database.db."
    )
