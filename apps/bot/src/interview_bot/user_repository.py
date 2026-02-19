from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Protocol


CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id BIGINT NOT NULL UNIQUE,
    preferred_language TEXT,
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

UPDATE_PREFERRED_LANGUAGE_SQLITE_SQL = """
UPDATE users
SET preferred_language = ?
WHERE telegram_id = ?
  AND COALESCE(preferred_language, '') <> ?;
"""

CREATE_USERS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL UNIQUE,
    preferred_language TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

ALTER_USERS_TABLE_POSTGRES_SQL = """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS preferred_language TEXT;
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
    language TEXT NOT NULL,
    code TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_SUBMISSION_SQLITE_SQL = """
INSERT INTO submissions (telegram_id, language, code)
VALUES (?, ?, ?);
"""

CREATE_SUBMISSIONS_TABLE_POSTGRES_SQL = """
CREATE TABLE IF NOT EXISTS submissions (
    id BIGSERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL,
    language TEXT NOT NULL,
    code TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_SUBMISSION_POSTGRES_SQL = """
INSERT INTO submissions (telegram_id, language, code)
VALUES (%s, %s, %s);
"""

SELECT_PREFERRED_LANGUAGE_POSTGRES_SQL = """
SELECT preferred_language
FROM users
WHERE telegram_id = %s;
"""

UPDATE_PREFERRED_LANGUAGE_POSTGRES_SQL = """
UPDATE users
SET preferred_language = %s
WHERE telegram_id = %s
  AND preferred_language IS DISTINCT FROM %s;
"""


class UserRepository(Protocol):
    async def ensure_schema(self) -> None:
        ...

    async def register_user(self, telegram_id: int) -> bool:
        ...

    async def get_preferred_language(self, telegram_id: int) -> str | None:
        ...

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool:
        ...

    async def save_submission(self, telegram_id: int, code: str) -> bool:
        ...


class SQLiteUserRepository:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    async def ensure_schema(self) -> None:
        await asyncio.to_thread(self._ensure_schema_sync)

    async def register_user(self, telegram_id: int) -> bool:
        return await asyncio.to_thread(self._register_user_sync, telegram_id)

    async def get_preferred_language(self, telegram_id: int) -> str | None:
        return await asyncio.to_thread(self._get_preferred_language_sync, telegram_id)

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool:
        return await asyncio.to_thread(
            self._set_preferred_language_sync,
            telegram_id,
            preferred_language,
        )

    async def save_submission(self, telegram_id: int, code: str) -> bool:
        return await asyncio.to_thread(self._save_submission_sync, telegram_id, code)

    def _ensure_schema_sync(self) -> None:
        if self._database_path != ":memory:":
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._database_path) as connection:
            connection.execute(CREATE_USERS_TABLE_SQL)
            if not self._has_column(connection, "users", "preferred_language"):
                connection.execute("ALTER TABLE users ADD COLUMN preferred_language TEXT;")
            connection.execute(CREATE_SUBMISSIONS_TABLE_SQLITE_SQL)
            connection.commit()

    def _register_user_sync(self, telegram_id: int) -> bool:
        with sqlite3.connect(self._database_path) as connection:
            cursor = connection.execute(INSERT_USER_SQLITE_SQL, (telegram_id,))
            connection.commit()
        return cursor.rowcount == 1

    def _get_preferred_language_sync(self, telegram_id: int) -> str | None:
        with sqlite3.connect(self._database_path) as connection:
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

    def _set_preferred_language_sync(self, telegram_id: int, preferred_language: str) -> bool:
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(INSERT_USER_SQLITE_SQL, (telegram_id,))
            cursor = connection.execute(
                UPDATE_PREFERRED_LANGUAGE_SQLITE_SQL,
                (preferred_language, telegram_id, preferred_language),
            )
            connection.commit()

        return cursor.rowcount == 1

    def _save_submission_sync(self, telegram_id: int, code: str) -> bool:
        with sqlite3.connect(self._database_path) as connection:
            language = connection.execute(
                SELECT_PREFERRED_LANGUAGE_SQLITE_SQL,
                (telegram_id,),
            ).fetchone()
            if language is None or language[0] is None:
                return False

            cursor = connection.execute(
                INSERT_SUBMISSION_SQLITE_SQL,
                (telegram_id, str(language[0]), code),
            )
            connection.commit()
        return cursor.rowcount == 1

    @staticmethod
    def _has_column(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        rows = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
        return any(str(row[1]) == column_name for row in rows)


class PostgresUserRepository:
    def __init__(self, database_url: str) -> None:
        self._database_url = normalize_database_url(database_url)

    async def ensure_schema(self) -> None:
        await self._execute(CREATE_USERS_TABLE_POSTGRES_SQL)
        await self._execute(ALTER_USERS_TABLE_POSTGRES_SQL)
        await self._execute(CREATE_SUBMISSIONS_TABLE_POSTGRES_SQL)

    async def register_user(self, telegram_id: int) -> bool:
        status = await self._execute(INSERT_USER_POSTGRES_SQL, (telegram_id,))
        return status == "INSERT 0 1"

    async def get_preferred_language(self, telegram_id: int) -> str | None:
        row = await self._fetch_one(SELECT_PREFERRED_LANGUAGE_POSTGRES_SQL, (telegram_id,))
        if row is None:
            return None

        value = row[0]
        if value is None:
            return None
        return str(value)

    async def set_preferred_language(self, telegram_id: int, preferred_language: str) -> bool:
        await self._execute(INSERT_USER_POSTGRES_SQL, (telegram_id,))
        status = await self._execute(
            UPDATE_PREFERRED_LANGUAGE_POSTGRES_SQL,
            (preferred_language, telegram_id, preferred_language),
        )
        return status == "UPDATE 1"

    async def save_submission(self, telegram_id: int, code: str) -> bool:
        preferred_language = await self.get_preferred_language(telegram_id)
        if preferred_language is None:
            return False

        status = await self._execute(
            INSERT_SUBMISSION_POSTGRES_SQL,
            (telegram_id, preferred_language, code),
        )
        return status == "INSERT 0 1"

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
