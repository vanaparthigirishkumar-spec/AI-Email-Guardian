"""Tests for Phase 4 SQLite persistence."""

from __future__ import annotations

import inspect
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import EmailMessage
from email_guardian.storage import (
    SCHEMA_VERSION,
    EmailStore,
    NotificationStatus,
    ProcessingResult,
    StorageMigrationError,
    StorageStateError,
    StoredEmail,
)
from email_guardian.summarizer import Classification


def make_settings(
    tmp: Path, *, threshold: int = 7, db_name: str = "emails.db"
) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / db_name),
        "LOG_DIR": str(tmp / "logs"),
        "IMPORTANCE_THRESHOLD": str(threshold),
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def classification_payload(
    *, account: str = "personal", message_id: str = "m1", score: int = 9
) -> dict[str, object]:
    return {
        "account_alias": account,
        "message_id": message_id,
        "thread_id": f"thread-{message_id}",
        "sender": "Example <sender@example.com>",
        "subject": "A safe subject",
        "email_date": datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
        "category": "IMPORTANT",
        "importance_score": score,
        "summary": "A safe summary.",
        "reason": "A safe reason.",
        "required_action": "Review the message.",
        "deadline": None,
        "confidence": 0.85,
    }


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.settings = make_settings(self.tmp)
        self.store = EmailStore(self.settings)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()


class SchemaTest(StorageTestCase):
    def test_initialize_creates_database_and_all_tables(self) -> None:
        self.store.initialize()

        self.assertTrue(Path(self.settings.database_path).is_file())
        self.assertEqual(self.store.schema_version(), SCHEMA_VERSION)
        with self.store.transaction() as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertTrue(
                {
                    "accounts",
                    "processed_emails",
                    "ai_classifications",
                    "notification_delivery",
                    "sync_checkpoints",
                    "schema_migrations",
                }.issubset(tables)
            )
            self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5000)
            self.assertEqual(
                str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "wal",
            )

    def test_migrations_upgrade_an_old_database(self) -> None:
        conn = sqlite3.connect(self.settings.database_path)
        try:
            conn.execute("PRAGMA user_version = 0")
            conn.execute("CREATE TABLE legacy_marker (value TEXT)")
            conn.commit()
        finally:
            conn.close()

        self.store.initialize()

        self.assertEqual(self.store.schema_version(), SCHEMA_VERSION)
        with self.store.transaction() as conn:
            versions = [
                int(row[0])
                for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]
            self.assertEqual(versions, list(range(1, SCHEMA_VERSION + 1)))
            self.assertIsNotNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'legacy_marker'"
                ).fetchone()
            )

    def test_migration_preserves_existing_rows(self) -> None:
        self.store.record_classification(**classification_payload())  # type: ignore[arg-type]
        self.store.close()

        conn = sqlite3.connect(self.settings.database_path)
        try:
            conn.execute("DROP TABLE notification_delivery")
            conn.execute("DROP TABLE sync_checkpoints")
            conn.execute("DELETE FROM schema_migrations WHERE version = 2")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        finally:
            conn.close()

        self.store.initialize()

        self.assertEqual(self.store.schema_version(), SCHEMA_VERSION)
        self.assertEqual(len(self.store.recent()), 1)

    def test_future_schema_version_is_rejected(self) -> None:
        conn = sqlite3.connect(self.settings.database_path)
        try:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(StorageMigrationError):
            self.store.initialize()


class DuplicateTest(StorageTestCase):
    def test_same_message_in_same_account_is_inserted_once(self) -> None:
        payload = classification_payload()
        first = self.store.record_classification(**payload)  # type: ignore[arg-type]
        changed = dict(payload)
        changed["summary"] = "A different summary."
        changed["reason"] = "A different reason."
        second = self.store.record_classification(**changed)  # type: ignore[arg-type]

        self.assertIs(first, ProcessingResult.INSERTED)
        self.assertIs(second, ProcessingResult.DUPLICATE)
        self.assertTrue(self.store.is_processed("m1", "personal"))
        self.assertEqual(len(self.store.recent()), 1)
        self.assertEqual(
            self.store.get_classification("personal", "m1")["summary"],
            "A safe summary.",
        )

    def test_same_message_in_different_accounts_is_allowed(self) -> None:
        personal = classification_payload(account="personal")
        work = classification_payload(account="work")

        self.assertIs(
            self.store.record_classification(**personal),  # type: ignore[arg-type]
            ProcessingResult.INSERTED,
        )
        self.assertIs(
            self.store.record_classification(**work),  # type: ignore[arg-type]
            ProcessingResult.INSERTED,
        )
        self.assertEqual(set(self.store.list_accounts()), {"personal", "work"})
        self.assertEqual(len(self.store.recent()), 2)
        self.assertEqual(len(self.store.recent(account_alias="work")), 1)

    def test_legacy_save_and_recent_interface_still_works(self) -> None:
        saved = StoredEmail(
            message_id="legacy-1",
            sender="Alice <alice@example.com>",
            subject="Legacy subject",
            score=6,
            summary="Legacy summary",
            notified=False,
            processed_at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
            account_alias="personal",
        )

        self.assertIs(self.store.save(saved), ProcessingResult.INSERTED)
        self.assertIs(self.store.save(saved), ProcessingResult.DUPLICATE)
        history = self.store.recent(limit=1)
        self.assertEqual(history[0].message_id, "legacy-1")
        self.assertEqual(history[0].summary, "Legacy summary")
        self.assertFalse(history[0].notified)

    def test_rollback_leaves_no_partial_data(self) -> None:
        self.store.initialize()
        with self.assertRaises(RuntimeError):
            with self.store.transaction() as conn:
                conn.execute(
                    "INSERT INTO accounts (alias, created_at, updated_at) "
                    "VALUES ('rollback', 'now', 'now')"
                )
                raise RuntimeError("force rollback")

        self.assertNotIn("rollback", self.store.list_accounts())
        self.assertFalse(self.store.is_processed("nothing"))

    def test_restart_preserves_data(self) -> None:
        self.store.record_classification(**classification_payload())  # type: ignore[arg-type]
        self.store.set_sync_checkpoint("personal", history_id="h1")
        self.store.close()

        reopened = EmailStore(self.settings)
        try:
            self.assertEqual(len(reopened.recent()), 1)
            self.assertEqual(reopened.get_sync_checkpoint("personal").history_id, "h1")
        finally:
            reopened.close()


class NotificationTest(StorageTestCase):
    def test_status_transitions_are_tracked(self) -> None:
        self.store.record_classification(**classification_payload())  # type: ignore[arg-type]
        self.assertIs(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.PENDING,
        )
        self.assertEqual(
            [task.message_id for task in self.store.pending_notifications()], ["m1"]
        )

        self.assertIs(
            self.store.mark_notification_failed("personal", "m1"),
            NotificationStatus.FAILED,
        )
        self.assertIs(
            self.store.mark_notification_pending("personal", "m1"),
            NotificationStatus.PENDING,
        )
        self.assertIs(
            self.store.mark_notification_sent(
                "personal", "m1", provider_message_id="telegram-123"
            ),
            NotificationStatus.SENT,
        )
        self.assertIs(
            self.store.mark_notification_sent("personal", "m1"),
            NotificationStatus.SENT,
        )
        self.assertEqual(self.store.pending_notifications(), [])

        with self.store.transaction() as conn:
            row = conn.execute(
                """
                SELECT attempts, sent_at, failed_at, provider_message_id
                FROM notification_delivery
                """
            ).fetchone()
            self.assertEqual(row["attempts"], 2)
            self.assertIsNotNone(row["sent_at"])
            self.assertIsNotNone(row["failed_at"])
            self.assertEqual(row["provider_message_id"], "telegram-123")

    def test_low_score_email_does_not_require_notification(self) -> None:
        payload = classification_payload(score=3)
        self.store.record_classification(**payload)  # type: ignore[arg-type]
        self.assertIs(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.NOT_REQUIRED,
        )
        with self.assertRaises(StorageStateError):
            self.store.mark_notification_sent("personal", "m1")

        self.assertIs(
            self.store.mark_notification_pending("personal", "m1"),
            NotificationStatus.PENDING,
        )
        self.assertIs(
            self.store.mark_notification_sent("personal", "m1"),
            NotificationStatus.SENT,
        )

    def test_prepare_notification_is_idempotent(self) -> None:
        self.store.record_classification(
            **classification_payload(score=2)  # type: ignore[arg-type]
        )
        self.assertIs(
            self.store.prepare_notification("personal", "m1", required=False),
            NotificationStatus.NOT_REQUIRED,
        )
        self.assertIs(
            self.store.prepare_notification("personal", "m1", required=True),
            NotificationStatus.NOT_REQUIRED,
        )


class SyncCheckpointTest(StorageTestCase):
    def test_checkpoint_persists_and_updates(self) -> None:
        first = self.store.set_sync_checkpoint("personal", history_id="100")
        self.assertEqual(first.account_alias, "personal")
        self.assertEqual(first.history_id, "100")

        updated = self.store.set_sync_checkpoint(
            "work", history_id="200", last_message_id="m9"
        )
        self.assertEqual(updated.last_message_id, "m9")
        self.assertIsNone(self.store.get_sync_checkpoint("personal").last_message_id)
        self.assertIsNone(self.store.get_sync_checkpoint("missing"))

        self.store.close()
        reopened = EmailStore(self.settings)
        try:
            self.assertEqual(reopened.get_sync_checkpoint("work").history_id, "200")
        finally:
            reopened.close()


class PrivacyTest(StorageTestCase):
    def test_schema_has_no_body_token_or_credential_columns(self) -> None:
        self.store.initialize()
        forbidden = (
            "body",
            "attachment",
            "oauth",
            "token",
            "credential",
            "password",
            "snippet",
            "raw_email",
            "full_message",
        )
        with self.store.transaction() as conn:
            tables = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            ]
            for table in tables:
                columns = [
                    str(row[1]).lower()
                    for row in conn.execute(f'PRAGMA table_info("{table}")')
                ]
                for column in columns:
                    for term in forbidden:
                        self.assertNotIn(term, column)

        parameters = inspect.signature(self.store.record_classification).parameters
        for name in ("body", "raw_email", "attachment", "token"):
            self.assertNotIn(name, parameters)

    def test_private_snippet_and_body_text_are_not_written(self) -> None:
        secret = "PRIVATE_BODY_SENTENCE_737463"
        message = EmailMessage(
            message_id="private-1",
            thread_id="thread-1",
            account="personal",
            sender_name="Alice",
            sender_email="alice@example.com",
            subject="A safe subject",
            snippet=secret,
            received_at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
        )
        analysis = Classification(
            category="IMPORTANT",
            score=8,
            summary="A safe summary.",
            reason="A safe reason.",
            required_action="Review the message.",
            deadline=None,
            confidence=0.9,
            model="llama3.2",
        )

        self.store.record_message(message, analysis)
        self.store.close()

        database = Path(self.settings.database_path)
        self.assertNotIn(secret.encode("utf-8"), database.read_bytes())
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(database) + suffix)
            if sidecar.exists():
                self.assertNotIn(secret.encode("utf-8"), sidecar.read_bytes())


class ConcurrencyAndCleanupTest(StorageTestCase):
    def test_concurrent_writes_keep_one_copy_per_account(self) -> None:
        closed: list[int] = []
        original_close = EmailStore.close

        def tracked_close(store: EmailStore) -> None:
            closed.append(id(store))
            original_close(store)

        with patch.object(EmailStore, "close", tracked_close):
            bootstrap = EmailStore(self.settings)
            try:
                bootstrap.initialize()
            finally:
                bootstrap.close()

            barrier = threading.Barrier(6)

            def write(index: int) -> ProcessingResult:
                store = EmailStore(self.settings)
                try:
                    store.initialize()
                    barrier.wait()
                    message_id = "shared" if index % 2 == 0 else f"unique-{index}"
                    return store.record_classification(
                        **classification_payload(message_id=message_id)  # type: ignore[arg-type]
                    )
                finally:
                    store.close()

            executor = ThreadPoolExecutor(max_workers=6)
            try:
                results = list(executor.map(write, range(6)))
            finally:
                executor.shutdown(wait=True)

            self.assertEqual(
                sum(1 for result in results if result is ProcessingResult.INSERTED), 4
            )
            self.assertEqual(
                sum(1 for result in results if result is ProcessingResult.DUPLICATE), 2
            )
            self.store.initialize()
            self.assertEqual(len(self.store.recent()), 4)
            self.store.close()

        self.assertEqual(len(closed), 8)

        database = Path(self.settings.database_path)
        database.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(database) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    def test_close_releases_wal_files_for_windows_cleanup(self) -> None:
        self.store.initialize()
        self.store.record_classification(**classification_payload())  # type: ignore[arg-type]
        database = Path(self.settings.database_path)

        self.store.close()
        self.store.close()
        self.assertIsNone(self.store._conn)

        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(database) + suffix)
            if sidecar.exists():
                sidecar.unlink()

        reopened = EmailStore(self.settings)
        try:
            self.assertEqual(len(reopened.recent()), 1)
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
