from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
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

            with closing(sqlite3.connect(database_path)) as connection:
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

    async def test_register_user_bootstraps_skill_profile(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            await repository.register_user(telegram_id=4242)
            skill_profile = await repository.get_skill_profile(telegram_id=4242)

        self.assertIsNotNone(skill_profile)
        if skill_profile is None:
            self.fail("Expected a bootstrapped skill profile.")
        self.assertEqual(skill_profile.get("version"), 1)
        self.assertEqual(skill_profile.get("language_scores"), {})
        self.assertEqual(skill_profile.get("category_scores"), {})
        self.assertEqual(skill_profile.get("recent_scores"), [])

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

            with closing(sqlite3.connect(database_path)) as connection:
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
            task_id = await repository.create_task(
                language="python",
                category="arrays",
                difficulty=2,
                prompt="Implement reverse array.",
                test_cases='[{"input":"1 2 3","output":"3 2 1"}]',
            )

            is_saved = await repository.save_submission(
                telegram_id=888,
                code="print('saved')",
                task_id=task_id,
            )

            with closing(sqlite3.connect(database_path)) as connection:
                row = connection.execute(
                    """
                    SELECT telegram_id, task_id, language, code
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
        self.assertEqual(row[1], task_id)
        self.assertEqual(row[2], "python")
        self.assertEqual(row[3], "print('saved')")

    async def test_save_submission_metrics_persists_runtime_memory_and_streams(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            await repository.register_user(telegram_id=999)
            await repository.set_preferred_language(
                telegram_id=999,
                preferred_language="python",
            )
            submission_id = await repository.save_submission_with_id(
                telegram_id=999,
                code="print('metrics')",
            )

            self.assertIsNotNone(submission_id)
            if submission_id is None:
                self.fail("Expected saved submission id.")

            is_saved = await repository.save_submission_metrics(
                submission_id=submission_id,
                runtime_ms=1234,
                memory_usage_kb=5678,
                exit_code=0,
                stdout="hello",
                stderr="",
                timed_out=False,
            )
            metrics = await repository.get_submission_metrics(submission_id)

        self.assertTrue(is_saved)
        self.assertIsNotNone(metrics)
        if metrics is None:
            self.fail("Expected metrics row.")
        self.assertEqual(metrics.runtime_ms, 1234)
        self.assertEqual(metrics.memory_usage_kb, 5678)
        self.assertEqual(metrics.exit_code, 0)
        self.assertEqual(metrics.stdout, "hello")
        self.assertEqual(metrics.stderr, "")
        self.assertFalse(metrics.timed_out)

    async def test_save_submission_metrics_requires_existing_submission(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            is_saved = await repository.save_submission_metrics(
                submission_id=404,
                runtime_ms=100,
                memory_usage_kb=10,
                exit_code=1,
                stdout="",
                stderr="error",
                timed_out=True,
            )
            metrics = await repository.get_submission_metrics(404)

        self.assertFalse(is_saved)
        self.assertIsNone(metrics)

    async def test_ensure_schema_migrates_existing_users_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"

            with closing(sqlite3.connect(database_path)) as connection:
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

            with closing(sqlite3.connect(database_path)) as connection:
                columns = [
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(users);").fetchall()
                ]
                task_columns = [
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(submissions);").fetchall()
                ]
                task_table_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks';"
                ).fetchone()
                row = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE telegram_id = ?",
                    (7,),
                ).fetchone()
                migration_count_row = connection.execute(
                    "SELECT COUNT(*) FROM schema_migrations;"
                ).fetchone()
                skill_profile_row = connection.execute(
                    "SELECT skill_profile FROM users WHERE telegram_id = ?",
                    (7,),
                ).fetchone()
                metrics_table_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'submission_metrics';"
                ).fetchone()

        user_count = row[0] if row is not None else 0
        migration_count = migration_count_row[0] if migration_count_row is not None else 0
        self.assertIn("preferred_language", columns)
        self.assertIn("skill_profile", columns)
        self.assertIn("task_id", task_columns)
        self.assertIsNotNone(task_table_exists)
        self.assertIsNotNone(metrics_table_exists)
        self.assertEqual(user_count, 1)
        self.assertGreaterEqual(migration_count, 8)
        self.assertIsNotNone(skill_profile_row)
        if skill_profile_row is None:
            self.fail("Expected migrated skill profile value.")
        parsed_skill_profile = json.loads(str(skill_profile_row[0]))
        self.assertEqual(parsed_skill_profile.get("version"), 1)

    async def test_task_crud_works(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()
            task_id = await repository.create_task(
                language="python",
                category="strings",
                difficulty=1,
                prompt="Reverse a string.",
                test_cases='[{"input":"abc","output":"cba"}]',
            )

            created_task = await repository.get_task(task_id)
            python_tasks = await repository.list_tasks(language="python")
            is_updated = await repository.update_task(
                task_id=task_id,
                language="python",
                category="arrays",
                difficulty=3,
                prompt="Rotate an array.",
                test_cases='[{"input":"1 2 3","output":"2 3 1"}]',
            )
            updated_task = await repository.get_task(task_id)
            is_deleted = await repository.delete_task(task_id)
            deleted_task = await repository.get_task(task_id)

        self.assertIsNotNone(created_task)
        if created_task is None:
            self.fail("Expected created task.")
        self.assertEqual(created_task.language, "python")
        self.assertEqual(created_task.category, "strings")
        self.assertEqual(len(python_tasks), 1)
        self.assertEqual(python_tasks[0].id, task_id)
        self.assertTrue(is_updated)
        self.assertIsNotNone(updated_task)
        if updated_task is None:
            self.fail("Expected updated task.")
        self.assertEqual(updated_task.category, "arrays")
        self.assertEqual(updated_task.difficulty, 3)
        self.assertTrue(is_deleted)
        self.assertIsNone(deleted_task)

    async def test_ensure_schema_creates_expected_indexes_and_versions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            database_path = Path(tmp_dir) / "bot.db"
            repository = SQLiteUserRepository(str(database_path))

            await repository.ensure_schema()

            with closing(sqlite3.connect(database_path)) as connection:
                versions = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version;"
                    ).fetchall()
                }
                index_names = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index';"
                    ).fetchall()
                }

        self.assertSetEqual(
            versions,
            {
                "0001_create_users",
                "0002_add_preferred_language",
                "0003_create_tasks",
                "0004_create_submissions",
                "0005_add_task_id_to_submissions",
                "0006_create_indexes",
                "0007_add_skill_profile",
                "0008_create_submission_metrics",
            },
        )
        self.assertIn("idx_users_created_at", index_names)
        self.assertIn("idx_tasks_language_difficulty", index_names)
        self.assertIn("idx_tasks_category", index_names)
        self.assertIn("idx_submissions_telegram_created_at", index_names)
        self.assertIn("idx_submissions_task_created_at", index_names)
        self.assertIn("idx_submission_metrics_submission_id", index_names)
        self.assertIn("idx_submission_metrics_created_at", index_names)

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
