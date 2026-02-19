from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from interview_bot.user_repository import (
    SQLiteUserRepository,
    build_user_repository,
    normalize_database_url,
)


class SQLiteUserRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_register_user_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            first_insert = await repository.register_user(telegram_id=100500)
            second_insert = await repository.register_user(telegram_id=100500)

            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE telegram_id = ?",
                    (100500,),
                ).fetchone()

        count = row[0] if row is not None else 0
        self.assertTrue(first_insert)
        self.assertFalse(second_insert)
        self.assertEqual(count, 1)

    async def test_preferred_language_is_persisted_and_repeatable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            await repository.register_user(telegram_id=42)

            current_language = await repository.get_preferred_language(telegram_id=42)
            first_save = await repository.set_preferred_language(
                telegram_id=42,
                preferred_language="python",
            )
            second_same_save = await repository.set_preferred_language(
                telegram_id=42,
                preferred_language="python",
            )
            switch_save = await repository.set_preferred_language(
                telegram_id=42,
                preferred_language="go",
            )
            updated_language = await repository.get_preferred_language(telegram_id=42)

        self.assertIsNone(current_language)
        self.assertTrue(first_save)
        self.assertFalse(second_same_save)
        self.assertTrue(switch_save)
        self.assertEqual(updated_language, "go")

    async def test_save_submission_requires_selected_language(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            await repository.register_user(telegram_id=777)

            is_saved = await repository.save_submission(
                telegram_id=777,
                code="print('hello')",
            )

            with sqlite3.connect(database_path) as connection:
                row = connection.execute("SELECT COUNT(*) FROM submissions").fetchone()

        count = row[0] if row is not None else 0
        self.assertFalse(is_saved)
        self.assertEqual(count, 0)

    async def test_save_submission_persists_code_in_submissions_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            await repository.register_user(telegram_id=888)
            await repository.set_preferred_language(
                telegram_id=888,
                preferred_language="python",
            )

            is_saved = await repository.save_submission(
                telegram_id=888,
                code="print('saved')",
            )

            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    """
                    SELECT telegram_id, language, code
                    FROM submissions
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()

        self.assertTrue(is_saved)
        self.assertIsNotNone(row)
        if row is None:
            self.fail("Expected one saved submission row.")

        self.assertEqual(row[0], 888)
        self.assertEqual(row[1], "python")
        self.assertEqual(row[2], "print('saved')")

    async def test_ensure_schema_migrates_existing_users_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"

            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        telegram_id BIGINT NOT NULL UNIQUE,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                connection.execute("INSERT INTO users (telegram_id) VALUES (?)", (7,))
                connection.commit()

            repository = SQLiteUserRepository(str(database_path))
            await repository.ensure_schema()

            with sqlite3.connect(database_path) as connection:
                columns = [
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(users);").fetchall()
                ]
                row = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE telegram_id = ?",
                    (7,),
                ).fetchone()

        user_count = row[0] if row is not None else 0
        self.assertIn("preferred_language", columns)
        self.assertEqual(user_count, 1)

    def test_build_repository_from_sqlite_url(self) -> None:
        repository = build_user_repository("sqlite:///:memory:")
        self.assertIsInstance(repository, SQLiteUserRepository)

    def test_normalize_database_url_for_postgres_driver(self) -> None:
        normalized = normalize_database_url(
            "postgresql+psycopg://postgres:postgres@localhost:5432/interview_assistant"
        )
        self.assertEqual(
            normalized,
            "postgresql://postgres:postgres@localhost:5432/interview_assistant",
        )


if __name__ == "__main__":
    unittest.main()
