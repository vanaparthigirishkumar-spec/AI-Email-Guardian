"""SQLite persistence for processed emails, classifications and alerts.

Only notification metadata is stored. Email bodies, attachments, OAuth tokens
and Google credentials are never passed to this module or written to SQLite.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from email_guardian.config import Settings

SCHEMA_VERSION = 3
BUSY_TIMEOUT_MS = 5000


class NotificationStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"


class ProcessingResult(str, Enum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"


class StorageError(RuntimeError):
    """Base class for SQLite persistence failures."""


class StorageMigrationError(StorageError):
    """Raised when the database schema version cannot be upgraded."""


class StorageBusyError(StorageError):
    """Raised when SQLite remains locked after the configured busy timeout."""


class StorageStateError(StorageError):
    """Raised when a requested state transition is not valid."""


@dataclass(frozen=True)
class StoredEmail:
    """Metadata for a previously processed email, kept for history queries."""

    message_id: str
    sender: str
    subject: str
    score: int
    summary: str
    notified: bool
    processed_at: datetime
    account_alias: str = "default"
    thread_id: str = ""
    received_at: str | None = None
    category: str = ""
    reason: str = ""
    required_action: str | None = None
    deadline: str | None = None
    confidence: float = 0.0
    rfc822_message_id: str = ""
    account_email: str = ""


@dataclass(frozen=True)
class SyncCheckpoint:
    """Account-level position used to recover messages missed while offline."""

    account_alias: str
    history_id: str | None
    last_message_id: str | None
    last_synced_at: str


@dataclass(frozen=True)
class NotificationTask:
    """Minimal identifier for a pending alert; contains no email content."""

    account_alias: str
    message_id: str
    attempts: int


@dataclass(frozen=True)
class PendingNotification:
    """Read-only view of a queued alert; never contains an email body."""

    account_alias: str
    message_id: str
    sender: str
    subject: str
    category: str
    score: int
    email_date: str | None
    required_action: str | None
    attempts: int
    rfc822_message_id: str = ""
    account_email: str = ""


_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS accounts (
        alias TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS processed_emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_alias TEXT NOT NULL
            REFERENCES accounts(alias) ON DELETE CASCADE,
        gmail_message_id TEXT NOT NULL,
        gmail_thread_id TEXT NOT NULL DEFAULT '',
        sender TEXT NOT NULL DEFAULT '',
        subject TEXT NOT NULL DEFAULT '',
        email_date TEXT,
        processed_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (account_alias, gmail_message_id)
    );
    CREATE INDEX IF NOT EXISTS idx_processed_emails_account
        ON processed_emails (account_alias, processed_at DESC);
    CREATE TABLE IF NOT EXISTS ai_classifications (
        processed_email_id INTEGER PRIMARY KEY
            REFERENCES processed_emails(id) ON DELETE CASCADE,
        category TEXT NOT NULL,
        importance_score INTEGER NOT NULL,
        summary TEXT NOT NULL,
        reason TEXT NOT NULL,
        required_action TEXT,
        deadline TEXT,
        confidence REAL NOT NULL,
        classified_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS notification_delivery (
        processed_email_id INTEGER PRIMARY KEY
            REFERENCES processed_emails(id) ON DELETE CASCADE,
        status TEXT NOT NULL
            CHECK (status IN ('NOT_REQUIRED', 'PENDING', 'SENT', 'FAILED')),
        attempts INTEGER NOT NULL DEFAULT 0,
        last_attempt_at TEXT,
        sent_at TEXT,
        failed_at TEXT,
        provider_message_id TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_notification_status
        ON notification_delivery (status);
    CREATE TABLE IF NOT EXISTS sync_checkpoints (
        account_alias TEXT PRIMARY KEY
            REFERENCES accounts(alias) ON DELETE CASCADE,
        history_id TEXT,
        last_message_id TEXT,
        last_synced_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    "",
)



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_date(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value)


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _storage_error(exc: sqlite3.Error) -> StorageError:
    lowered = str(exc).lower()
    if "locked" in lowered or "busy" in lowered:
        return StorageBusyError(f"SQLite is busy or locked: {exc}")
    return StorageError(f"SQLite operation failed: {exc}")


class EmailStore:
    """Transaction-safe SQLite store with schema versioning and WAL mode."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.database_path)
        self._threshold = settings.importance_threshold
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._savepoints: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> EmailStore:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def initialize(self) -> None:
        with self._lock:
            if self._conn is not None:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = self._connect()

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            try:
                conn.close()
            finally:
                self._conn = None
                self._savepoints.clear()

    def schema_version(self) -> int:
        conn = self._require_connection()
        with self._lock:
            row = conn.execute("PRAGMA user_version").fetchone()
            return int(row[0])

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._require_connection()
        with self._transaction(conn):
            yield conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path),
            timeout=BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            self._migrate(conn)
        except Exception:
            conn.close()
            raise
        return conn

    def _require_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self.initialize()
            assert self._conn is not None
            return self._conn

    @contextmanager
    def _transaction(self, conn: sqlite3.Connection) -> Iterator[None]:
        with self._lock:
            savepoint = f"sp_{len(self._savepoints)}"
            try:
                if self._savepoints:
                    conn.execute(f"SAVEPOINT {savepoint}")
                else:
                    conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise _storage_error(exc) from exc
            self._savepoints.append(savepoint)
            try:
                yield
            except BaseException:
                try:
                    if len(self._savepoints) == 1:
                        conn.rollback()
                    else:
                        conn.execute(f"ROLLBACK TO {savepoint}")
                        conn.execute(f"RELEASE {savepoint}")
                except sqlite3.Error:
                    pass
                raise
            else:
                try:
                    if len(self._savepoints) == 1:
                        conn.commit()
                    else:
                        conn.execute(f"RELEASE {savepoint}")
                except sqlite3.Error as exc:
                    raise _storage_error(exc) from exc
            finally:
                self._savepoints.pop()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        with self._transaction(conn):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise StorageMigrationError(
                    f"Database schema version {current} is newer than supported "
                    f"version {SCHEMA_VERSION}."
                )
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[version - 1].split(";"):
                    if statement.strip():
                        conn.execute(statement)
                if version == 3:
                    self._ensure_email_link_columns(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations "
                    "(version, name, applied_at) VALUES (?, ?, ?)",
                    (version, f"schema_v{version}", _utc_now()),
                )
                conn.execute(f"PRAGMA user_version={version}")

    @staticmethod
    def _ensure_email_link_columns(conn: sqlite3.Connection) -> None:
        columns = {
            str(row[1]).lower()
            for row in conn.execute("PRAGMA table_info(processed_emails)")
        }
        if "rfc822_message_id" not in columns:
            conn.execute(
                "ALTER TABLE processed_emails "
                "ADD COLUMN rfc822_message_id TEXT NOT NULL DEFAULT ''"
            )
        if "account_email" not in columns:
            conn.execute(
                "ALTER TABLE processed_emails "
                "ADD COLUMN account_email TEXT NOT NULL DEFAULT ''"
            )

    def upsert_account(self, alias: str) -> None:
        if not alias or not alias.strip():
            raise StorageError("Account alias must not be empty.")
        with self.transaction() as conn:
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO accounts (alias, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (alias.strip(), now, now),
            )

    def list_accounts(self) -> tuple[str, ...]:
        conn = self._require_connection()
        with self._lock:
            rows = conn.execute("SELECT alias FROM accounts ORDER BY alias").fetchall()
            return tuple(str(row["alias"]) for row in rows)

    def is_processed(self, message_id: str, account_alias: str | None = None) -> bool:
        conn = self._require_connection()
        with self._lock:
            if account_alias:
                row = conn.execute(
                    """
                    SELECT 1 FROM processed_emails
                    WHERE account_alias = ? AND gmail_message_id = ?
                    """,
                    (account_alias, message_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM processed_emails WHERE gmail_message_id = ?",
                    (message_id,),
                ).fetchone()
            return row is not None

    def _remember_email(
        self,
        conn: sqlite3.Connection,
        *,
        account_alias: str,
        message_id: str,
        thread_id: str,
        sender: str,
        subject: str,
        email_date: datetime | str | None,
        processed_at: str,
        rfc822_message_id: str = "",
        account_email: str = "",
    ) -> int | None:
        now = processed_at
        conn.execute(
            """
            INSERT INTO accounts (alias, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(alias) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (account_alias, now, now),
        )
        cursor = conn.execute(
            """
            INSERT INTO processed_emails (
                account_alias, gmail_message_id, gmail_thread_id, sender,
                subject, email_date, processed_at, updated_at,
                rfc822_message_id, account_email
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_alias, gmail_message_id) DO NOTHING
            """,
            (
                account_alias,
                message_id,
                thread_id or "",
                sender or "",
                subject or "",
                _iso_date(email_date),
                now,
                now,
                rfc822_message_id or "",
                account_email or "",
            ),
        )
        if cursor.rowcount == 0:
            return None
        row = conn.execute(
            """
            SELECT id FROM processed_emails
            WHERE account_alias = ? AND gmail_message_id = ?
            """,
            (account_alias, message_id),
        ).fetchone()
        return int(row["id"])

    def record_classification(
        self,
        *,
        account_alias: str,
        message_id: str,
        thread_id: str = "",
        sender: str = "",
        subject: str = "",
        email_date: datetime | str | None = None,
        category: str,
        importance_score: int,
        summary: str,
        reason: str,
        required_action: str | None = None,
        deadline: str | None = None,
        confidence: float = 0.0,
        notification_required: bool | None = None,
        processed_at: datetime | str | None = None,
        rfc822_message_id: str = "",
        account_email: str = "",
    ) -> ProcessingResult:
        now = _iso_date(processed_at) or _utc_now()
        needs_alert = (
            importance_score >= self._threshold
            if notification_required is None
            else notification_required
        )
        with self.transaction() as conn:
            email_id = self._remember_email(
                conn,
                account_alias=account_alias,
                message_id=message_id,
                thread_id=thread_id,
                sender=sender,
                subject=subject,
                email_date=email_date,
                processed_at=now,
                rfc822_message_id=rfc822_message_id,
                account_email=account_email,
            )
            if email_id is None:
                return ProcessingResult.DUPLICATE
            conn.execute(
                """
                INSERT INTO ai_classifications (
                    processed_email_id, category, importance_score, summary,
                    reason, required_action, deadline, confidence, classified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    category,
                    int(importance_score),
                    summary,
                    reason,
                    required_action,
                    deadline,
                    float(confidence),
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO notification_delivery (
                    processed_email_id, status, attempts, updated_at
                ) VALUES (?, ?, 0, ?)
                """,
                (
                    email_id,
                    NotificationStatus.PENDING.value
                    if needs_alert
                    else NotificationStatus.NOT_REQUIRED.value,
                    now,
                ),
            )
        return ProcessingResult.INSERTED

    def record_message(
        self,
        email: Any,
        analysis: Any,
        *,
        account_alias: str | None = None,
        notification_required: bool | None = None,
        account_email: str = "",
    ) -> ProcessingResult:
        alias = account_alias or str(email.account)
        return self.record_classification(
            account_alias=alias,
            message_id=str(email.message_id),
            thread_id=str(email.thread_id or ""),
            sender=str(email.sender or ""),
            subject=str(email.subject or ""),
            email_date=email.received_at,
            category=str(analysis.category),
            importance_score=int(analysis.importance_score),
            summary=str(analysis.summary),
            reason=str(analysis.reason),
            required_action=analysis.required_action,
            deadline=analysis.deadline,
            confidence=float(analysis.confidence),
            notification_required=notification_required,
            rfc822_message_id=str(getattr(email, "rfc822_message_id", "") or ""),
            account_email=account_email,
        )

    def save(self, email: StoredEmail) -> ProcessingResult:
        return self.record_classification(
            account_alias=email.account_alias,
            message_id=email.message_id,
            thread_id=email.thread_id,
            sender=email.sender,
            subject=email.subject,
            email_date=email.received_at,
            category=email.category or "NORMAL",
            importance_score=email.score,
            summary=email.summary,
            reason=email.reason or "No classification reason was recorded.",
            required_action=email.required_action,
            deadline=email.deadline,
            confidence=email.confidence,
            notification_required=email.notified,
            processed_at=email.processed_at,
            rfc822_message_id=email.rfc822_message_id,
            account_email=email.account_email,
        )

    _EMAIL_COLUMNS = """
        SELECT p.account_alias, p.gmail_message_id, p.gmail_thread_id,
               p.sender, p.subject, p.email_date, p.processed_at,
               p.rfc822_message_id, p.account_email,
               c.category, c.importance_score, c.summary, c.reason,
               c.required_action, c.deadline, c.confidence, n.status
        FROM processed_emails p
        LEFT JOIN ai_classifications c
            ON c.processed_email_id = p.id
        LEFT JOIN notification_delivery n
            ON n.processed_email_id = p.id
    """

    @staticmethod
    def _stored_email(row: Any) -> StoredEmail:
        return StoredEmail(
            message_id=str(row["gmail_message_id"]),
            sender=str(row["sender"]),
            subject=str(row["subject"]),
            score=int(row["importance_score"] or 0),
            summary=str(row["summary"] or ""),
            notified=row["status"] == NotificationStatus.SENT.value,
            processed_at=_parse_date(row["processed_at"]),
            account_alias=str(row["account_alias"]),
            thread_id=str(row["gmail_thread_id"] or ""),
            received_at=row["email_date"],
            category=str(row["category"] or ""),
            reason=str(row["reason"] or ""),
            required_action=row["required_action"],
            deadline=row["deadline"],
            confidence=float(row["confidence"] or 0),
            rfc822_message_id=str(row["rfc822_message_id"] or ""),
            account_email=str(row["account_email"] or ""),
        )

    def get_email(self, account_alias: str, message_id: str) -> StoredEmail | None:
        """Return stored metadata for one email, or None when it is unknown."""

        conn = self._require_connection()
        with self._lock:
            row = conn.execute(
                self._EMAIL_COLUMNS
                + " WHERE p.account_alias = ? AND p.gmail_message_id = ? LIMIT 1",
                (account_alias, message_id),
            ).fetchone()
            return self._stored_email(row) if row is not None else None

    def recent(self, limit: int = 10, account_alias: str | None = None) -> list[StoredEmail]:
        conn = self._require_connection()
        with self._lock:
            params: tuple[Any, ...]
            where = ""
            if account_alias:
                where = "WHERE p.account_alias = ?"
                params = (account_alias, max(0, limit))
            else:
                params = (max(0, limit),)
            rows = conn.execute(
                self._EMAIL_COLUMNS
                + f" {where} ORDER BY p.processed_at DESC, p.id DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._stored_email(row) for row in rows]

    def get_classification(
        self, account_alias: str, message_id: str
    ) -> dict[str, Any] | None:
        conn = self._require_connection()
        with self._lock:
            row = conn.execute(
                """
                SELECT c.category, c.importance_score, c.summary, c.reason,
                       c.required_action, c.deadline, c.confidence, c.classified_at
                FROM processed_emails p
                JOIN ai_classifications c ON c.processed_email_id = p.id
                WHERE p.account_alias = ? AND p.gmail_message_id = ?
                """,
                (account_alias, message_id),
            ).fetchone()
            return dict(row) if row is not None else None

    def prepare_notification(
        self, account_alias: str, message_id: str, *, required: bool = True
    ) -> NotificationStatus:
        status = (
            NotificationStatus.PENDING
            if required
            else NotificationStatus.NOT_REQUIRED
        )
        with self.transaction() as conn:
            email_id = self._email_id(conn, account_alias, message_id)
            conn.execute(
                """
                INSERT INTO notification_delivery (
                    processed_email_id, status, attempts, updated_at
                ) VALUES (?, ?, 0, ?)
                ON CONFLICT(processed_email_id) DO NOTHING
                """,
                (email_id, status.value, _utc_now()),
            )
            row = conn.execute(
                "SELECT status FROM notification_delivery WHERE processed_email_id = ?",
                (email_id,),
            ).fetchone()
            return NotificationStatus(str(row["status"]))

    def get_notification_status(
        self, account_alias: str, message_id: str
    ) -> NotificationStatus | None:
        conn = self._require_connection()
        with self._lock:
            row = conn.execute(
                """
                SELECT n.status FROM notification_delivery n
                JOIN processed_emails p ON p.id = n.processed_email_id
                WHERE p.account_alias = ? AND p.gmail_message_id = ?
                """,
                (account_alias, message_id),
            ).fetchone()
            return NotificationStatus(str(row["status"])) if row else None

    def _email_id(
        self, conn: sqlite3.Connection, account_alias: str, message_id: str
    ) -> int:
        row = conn.execute(
            """
            SELECT id FROM processed_emails
            WHERE account_alias = ? AND gmail_message_id = ?
            """,
            (account_alias, message_id),
        ).fetchone()
        if row is None:
            raise StorageStateError(
                f"Email {message_id!r} for account {account_alias!r} was not recorded."
            )
        return int(row["id"])

    def mark_notification_pending(
        self, account_alias: str, message_id: str
    ) -> NotificationStatus:
        with self.transaction() as conn:
            email_id = self._email_id(conn, account_alias, message_id)
            row = conn.execute(
                """
                SELECT status FROM notification_delivery
                WHERE processed_email_id = ?
                """,
                (email_id,),
            ).fetchone()
            if row is None:
                return self.prepare_notification(
                    account_alias, message_id, required=True
                )
            status = NotificationStatus(str(row["status"]))
            if status is NotificationStatus.SENT:
                return status
            conn.execute(
                """
                UPDATE notification_delivery
                SET status = ?, updated_at = ?
                WHERE processed_email_id = ?
                """,
                (NotificationStatus.PENDING.value, _utc_now(), email_id),
            )
            return NotificationStatus.PENDING

    def suppress_notification(
        self, account_alias: str, message_id: str
    ) -> NotificationStatus:
        """Move one queued alert to the terminal NOT_REQUIRED state.

        Only the notification status row is touched: email metadata, the AI
        classification and sync checkpoints are left unchanged. A suppressed
        alert is excluded from `pending_notifications` and is never retried.
        """

        with self.transaction() as conn:
            email_id = self._email_id(conn, account_alias, message_id)
            row = conn.execute(
                """
                SELECT status FROM notification_delivery
                WHERE processed_email_id = ?
                """,
                (email_id,),
            ).fetchone()
            if row is None:
                raise StorageStateError("No notification was prepared for this email.")
            status = NotificationStatus(str(row["status"]))
            if status is NotificationStatus.SENT:
                raise StorageStateError(
                    "This notification was already delivered; it cannot be suppressed."
                )
            if status is not NotificationStatus.NOT_REQUIRED:
                conn.execute(
                    """
                    UPDATE notification_delivery
                    SET status = ?, updated_at = ?
                    WHERE processed_email_id = ?
                    """,
                    (
                        NotificationStatus.NOT_REQUIRED.value,
                        _utc_now(),
                        email_id,
                    ),
                )
            return NotificationStatus.NOT_REQUIRED

    def mark_notification_sent(
        self,
        account_alias: str,
        message_id: str,
        *,
        provider_message_id: str | None = None,
    ) -> NotificationStatus:
        with self.transaction() as conn:
            email_id = self._email_id(conn, account_alias, message_id)
            row = conn.execute(
                """
                SELECT status FROM notification_delivery
                WHERE processed_email_id = ?
                """,
                (email_id,),
            ).fetchone()
            if row is None:
                raise StorageStateError("No notification was prepared for this email.")
            status = NotificationStatus(str(row["status"]))
            if status is NotificationStatus.SENT:
                return status
            if status is NotificationStatus.NOT_REQUIRED:
                raise StorageStateError("This email is not marked for notification.")
            now = _utc_now()
            conn.execute(
                """
                UPDATE notification_delivery
                SET status = ?, attempts = attempts + 1, last_attempt_at = ?,
                    sent_at = ?, provider_message_id = ?, updated_at = ?
                WHERE processed_email_id = ?
                """,
                (
                    NotificationStatus.SENT.value,
                    now,
                    now,
                    provider_message_id or None,
                    now,
                    email_id,
                ),
            )
            return NotificationStatus.SENT

    def mark_notification_failed(
        self, account_alias: str, message_id: str
    ) -> NotificationStatus:
        with self.transaction() as conn:
            email_id = self._email_id(conn, account_alias, message_id)
            row = conn.execute(
                """
                SELECT status FROM notification_delivery
                WHERE processed_email_id = ?
                """,
                (email_id,),
            ).fetchone()
            if row is None:
                raise StorageStateError("No notification was prepared for this email.")
            status = NotificationStatus(str(row["status"]))
            if status is NotificationStatus.SENT:
                return status
            if status is NotificationStatus.NOT_REQUIRED:
                raise StorageStateError("This email is not marked for notification.")
            now = _utc_now()
            conn.execute(
                """
                UPDATE notification_delivery
                SET status = ?, attempts = attempts + 1, last_attempt_at = ?,
                    failed_at = ?, updated_at = ?
                WHERE processed_email_id = ?
                """,
                (
                    NotificationStatus.FAILED.value,
                    now,
                    now,
                    now,
                    email_id,
                ),
            )
            return NotificationStatus.FAILED

    def pending_notifications(self, limit: int = 50) -> list[NotificationTask]:
        conn = self._require_connection()
        with self._lock:
            rows = conn.execute(
                """
                SELECT p.account_alias, p.gmail_message_id, n.attempts
                FROM notification_delivery n
                JOIN processed_emails p ON p.id = n.processed_email_id
                WHERE n.status = ?
                ORDER BY n.updated_at, p.id
                LIMIT ?
                """,
                (NotificationStatus.PENDING.value, max(0, limit)),
            ).fetchall()
            return [
                NotificationTask(
                    account_alias=str(row["account_alias"]),
                    message_id=str(row["gmail_message_id"]),
                    attempts=int(row["attempts"]),
                )
                for row in rows
            ]

    def count_pending_notifications(self) -> int:
        """Count queued alerts without changing any row."""

        conn = self._require_connection()
        with self._lock:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM notification_delivery WHERE status = ?",
                (NotificationStatus.PENDING.value,),
            ).fetchone()
            return int(row["total"])

    def pending_notifications_detailed(self) -> list[PendingNotification]:
        """Return every queued alert with metadata only, in queue order."""

        conn = self._require_connection()
        with self._lock:
            rows = conn.execute(
                """
                SELECT p.account_alias, p.gmail_message_id, p.sender, p.subject,
                       p.email_date, p.rfc822_message_id, p.account_email,
                       c.category, c.importance_score, c.required_action, n.attempts
                FROM notification_delivery n
                JOIN processed_emails p ON p.id = n.processed_email_id
                LEFT JOIN ai_classifications c ON c.processed_email_id = p.id
                WHERE n.status = ?
                ORDER BY n.updated_at, p.id
                """,
                (NotificationStatus.PENDING.value,),
            ).fetchall()
            return [
                PendingNotification(
                    account_alias=str(row["account_alias"]),
                    message_id=str(row["gmail_message_id"]),
                    sender=str(row["sender"] or ""),
                    subject=str(row["subject"] or ""),
                    category=str(row["category"] or ""),
                    score=int(row["importance_score"] or 0),
                    email_date=row["email_date"],
                    required_action=row["required_action"],
                    attempts=int(row["attempts"]),
                    rfc822_message_id=str(row["rfc822_message_id"] or ""),
                    account_email=str(row["account_email"] or ""),
                )
                for row in rows
            ]

    def set_sync_checkpoint(
        self,
        account_alias: str,
        history_id: str | None = None,
        last_message_id: str | None = None,
        *,
        last_synced_at: datetime | str | None = None,
    ) -> SyncCheckpoint:
        synced = _iso_date(last_synced_at) or _utc_now()
        with self.transaction() as conn:
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO accounts (alias, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (account_alias, now, now),
            )
            conn.execute(
                """
                INSERT INTO sync_checkpoints (
                    account_alias, history_id, last_message_id, last_synced_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_alias) DO UPDATE SET
                    history_id = excluded.history_id,
                    last_message_id = excluded.last_message_id,
                    last_synced_at = excluded.last_synced_at,
                    updated_at = excluded.updated_at
                """,
                (account_alias, history_id, last_message_id, synced, now),
            )
        return SyncCheckpoint(account_alias, history_id, last_message_id, synced)

    def get_sync_checkpoint(self, account_alias: str) -> SyncCheckpoint | None:
        conn = self._require_connection()
        with self._lock:
            row = conn.execute(
                """
                SELECT account_alias, history_id, last_message_id, last_synced_at
                FROM sync_checkpoints WHERE account_alias = ?
                """,
                (account_alias,),
            ).fetchone()
            if row is None:
                return None
            return SyncCheckpoint(
                account_alias=str(row["account_alias"]),
                history_id=row["history_id"],
                last_message_id=row["last_message_id"],
                last_synced_at=str(row["last_synced_at"]),
            )
