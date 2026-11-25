"""One-cycle orchestration of Gmail, Ollama, SQLite and Telegram.

Only metadata and analysis output are persisted; email bodies stay in memory
for the duration of a single classification and are never logged.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from email_guardian.config import Settings
from email_guardian.gmail_client import EmailMessage, GmailClient, GmailError
from email_guardian.notifier import TelegramError, TelegramNotifier, gmail_message_url
from email_guardian.storage import (
    EmailStore,
    ProcessingResult,
    StorageError,
    StoredEmail,
)
from email_guardian.summarizer import (
    Classification,
    OllamaAnalyzer,
    OllamaError,
    OllamaUnavailableError,
)

LOGGER = logging.getLogger(__name__)

NOTIFIABLE_CATEGORIES = frozenset({"URGENT", "IMPORTANT"})
MAX_RECOVERY_NOTIFICATIONS = 50


def should_notify(category: str, score: int, threshold: int) -> bool:
    """Whether a stored classification is eligible for Telegram delivery."""

    if category not in NOTIFIABLE_CATEGORIES:
        return False
    return category == "URGENT" or score >= threshold


@dataclass
class CycleReport:
    """Counts for one processing cycle; contains no email content."""

    accounts_processed: int = 0
    fetched: int = 0
    analyzed: int = 0
    stored: int = 0
    duplicates: int = 0
    failed: int = 0
    account_failures: int = 0
    notifications_sent: int = 0
    notifications_failed: int = 0
    recovered: int = 0
    dry_run_pending: int = 0
    checkpoints_updated: int = 0

    @property
    def has_failures(self) -> bool:
        return bool(self.failed or self.account_failures or self.notifications_failed)


class EmailGuardianAgent:
    """Coordinates Gmail, Ollama, SQLite and Telegram for one cycle at a time."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: EmailStore | None = None,
        analyzer: OllamaAnalyzer | None = None,
        notifier: TelegramNotifier | None = None,
        client_factory: Callable[[Settings, str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._store = store if store is not None else EmailStore(settings)
        self._owns_store = store is None
        self._analyzer = analyzer if analyzer is not None else OllamaAnalyzer(settings)
        self._notifier = notifier if notifier is not None else TelegramNotifier(settings)
        self._client_factory = client_factory or GmailClient
        self._account_emails: dict[str, str] = {}
        self._ollama_unavailable = False

    def run_once(self, *, dry_run: bool = False) -> CycleReport:
        """Process every configured account once, then return the report."""

        report = CycleReport()
        self._ollama_unavailable = False
        self._store.initialize()
        try:
            self._recover_pending(report, dry_run=dry_run)
            self._analyzer.ensure_available()
            for account in self._settings.gmail_accounts:
                self._process_account(account, report, dry_run=dry_run)
            return report
        finally:
            if self._owns_store:
                self._store.close()

    def run_forever(
        self,
        *,
        stop_event: threading.Event | None = None,
        interval_seconds: int | None = None,
    ) -> None:
        """Poll both accounts until Ctrl+C (or ``stop_event``) ends the loop."""

        stop = stop_event if stop_event is not None else threading.Event()
        interval = (
            self._settings.poll_interval_seconds
            if interval_seconds is None
            else max(0, int(interval_seconds))
        )
        accounts = ", ".join(self._settings.gmail_accounts) or "none"
        LOGGER.info("Monitoring %s every %ss (read-only Gmail)", accounts, interval)
        try:
            while not stop.is_set():
                try:
                    report = self.run_once()
                    LOGGER.info(
                        "Cycle complete: %d stored, %d notified, %d failed",
                        report.stored,
                        report.notifications_sent,
                        report.failed
                        + report.account_failures
                        + report.notifications_failed,
                    )
                except KeyboardInterrupt:
                    raise
                except (OllamaError, StorageError, GmailError, TelegramError) as exc:
                    LOGGER.error(
                        "Monitor cycle failed (%s); retrying after the interval",
                        type(exc).__name__,
                    )
                except Exception as exc:
                    LOGGER.error(
                        "Monitor cycle failed unexpectedly (%s); "
                        "retrying after the interval",
                        type(exc).__name__,
                    )
                if stop.wait(interval):
                    break
        except KeyboardInterrupt:
            LOGGER.info("Monitoring stopped by user (Ctrl+C).")
        finally:
            LOGGER.info("Monitoring stopped for: %s", accounts)

    def _authenticated_email(self, account: str) -> str:
        """Resolve the receiving account's own address from its Gmail identity.

        Configured expectations and stored queue rows are deliberately ignored:
        both can be stale or wrong. No email content is involved.
        """

        cached = self._account_emails.get(account)
        if cached:
            return cached
        try:
            client = self._client_factory(self._settings, account)
        except GmailError as exc:
            LOGGER.warning("Could not open account %s: %s", account, exc)
            return ""

        read_local = getattr(client, "authenticated_email", lambda: "")
        try:
            local_email = read_local()
        except GmailError as exc:
            LOGGER.warning(
                "Could not read the stored address for account %s: %s", account, exc
            )
            local_email = ""
        email = local_email if isinstance(local_email, str) and local_email else ""

        if not email:
            try:
                profile_email = getattr(client.get_profile(), "email", None)
                if isinstance(profile_email, str):
                    email = profile_email
            except GmailError as exc:
                LOGGER.warning("Could not confirm address for account %s: %s", account, exc)
        if email:
            self._account_emails[account] = email
        return email

    def _recover_pending(self, report: CycleReport, *, dry_run: bool) -> None:
        try:
            tasks = self._store.pending_notifications(MAX_RECOVERY_NOTIFICATIONS)
        except StorageError as exc:
            LOGGER.warning("Could not load pending notifications: %s", exc)
            report.account_failures += 1
            return

        for task in tasks:
            try:
                stored = self._store.get_email(task.account_alias, task.message_id)
                analysis_data = self._store.get_classification(
                    task.account_alias, task.message_id
                )
                if stored is None or analysis_data is None:
                    LOGGER.warning(
                        "Pending alert for message %s (account %s) has no stored analysis",
                        task.message_id,
                        task.account_alias,
                    )
                    continue
                category = str(analysis_data["category"])
                score = int(analysis_data["importance_score"])
                if not self._should_notify_category(category, score):
                    self._store.prepare_notification(
                        task.account_alias, task.message_id, required=False
                    )
                    continue
                if dry_run:
                    report.dry_run_pending += 1
                    continue
                email = self._stored_email(stored)
                analysis = self._classification(analysis_data)
                account_email = self._authenticated_email(task.account_alias)
                link = gmail_message_url(account_email, stored.rfc822_message_id)
                if self._deliver(
                    task.account_alias, task.message_id, email, analysis, gmail_url=link
                ):
                    report.recovered += 1
                else:
                    report.notifications_failed += 1
            except (StorageError, KeyError, TypeError, ValueError) as exc:
                report.notifications_failed += 1
                LOGGER.warning(
                    "Pending alert for message %s (account %s) could not be retried: %s",
                    task.message_id,
                    task.account_alias,
                    exc,
                )

    def _process_account(
        self, account: str, report: CycleReport, *, dry_run: bool
    ) -> None:
        try:
            self._store.upsert_account(account)
            client = self._client_factory(self._settings, account)
            messages = client.fetch_recent(limit=self._settings.gmail_fetch_limit)
        except (GmailError, StorageError) as exc:
            LOGGER.error("Account %s could not be processed: %s", account, exc)
            report.account_failures += 1
            return

        try:
            profile_email = getattr(client.get_profile(), "email", None)
        except GmailError as exc:
            LOGGER.error(
                "Account %s could not confirm its Gmail identity: %s", account, exc
            )
            report.account_failures += 1
            return
        if not isinstance(profile_email, str) or not profile_email:
            LOGGER.error(
                "Account %s did not return an authenticated email address", account
            )
            report.account_failures += 1
            return
        account_email = profile_email
        self._account_emails[account] = account_email

        report.accounts_processed += 1
        report.fetched += len(messages)
        if self._ollama_unavailable:
            report.failed += len(messages)
            LOGGER.warning(
                "Ollama is unavailable; skipped %d message(s) for account %s "
                "without further inference attempts",
                len(messages),
                account,
            )
            return

        failures = 0
        remaining = len(messages)
        for message in messages:
            remaining -= 1
            try:
                if self._store.is_processed(message.message_id, account):
                    report.duplicates += 1
                    continue
                body = client.fetch_body(message.message_id).text
                analysis = self._analyzer.analyze(message, body)
                needs_alert = self._should_notify(analysis)
                outcome = self._store.record_message(
                    message,
                    analysis,
                    notification_required=needs_alert,
                    account_email=account_email,
                )
                if outcome is ProcessingResult.DUPLICATE:
                    report.duplicates += 1
                    continue
                report.stored += 1
                report.analyzed += 1
                if not needs_alert:
                    continue
                if dry_run:
                    report.dry_run_pending += 1
                elif self._deliver(
                    account,
                    message.message_id,
                    message,
                    analysis,
                    gmail_url=gmail_message_url(
                        account_email, getattr(message, "rfc822_message_id", "")
                    ),
                ):
                    report.notifications_sent += 1
                else:
                    report.notifications_failed += 1
            except OllamaUnavailableError as exc:
                skipped = 1 + remaining
                failures += skipped
                report.failed += skipped
                self._ollama_unavailable = True
                LOGGER.error(
                    "Ollama became unavailable for account %s (%s); "
                    "%d message(s) deferred to the next cycle",
                    account,
                    exc,
                    skipped,
                )
                break
            except (GmailError, OllamaError, StorageError) as exc:
                failures += 1
                report.failed += 1
                LOGGER.warning(
                    "Message %s for account %s was not processed: %s",
                    message.message_id,
                    account,
                    exc,
                )

        if failures == 0:
            self._save_checkpoint(account, messages, report)

    def _save_checkpoint(
        self, account: str, messages: list[EmailMessage], report: CycleReport
    ) -> None:
        try:
            existing = self._store.get_sync_checkpoint(account)
            history_id = existing.history_id if existing else None
            last_message_id = (
                messages[-1].message_id
                if messages
                else (existing.last_message_id if existing else None)
            )
            self._store.set_sync_checkpoint(
                account, history_id=history_id, last_message_id=last_message_id
            )
            report.checkpoints_updated += 1
        except StorageError as exc:
            LOGGER.warning(
                "Could not update the checkpoint for account %s: %s", account, exc
            )

    def _should_notify(self, analysis: Classification) -> bool:
        return self._should_notify_category(analysis.category, analysis.score)

    def _should_notify_category(self, category: str, score: int) -> bool:
        return should_notify(category, score, self._settings.importance_threshold)

    def _deliver(
        self,
        account: str,
        message_id: str,
        email: EmailMessage,
        analysis: Classification,
        *,
        gmail_url: str | None = None,
    ) -> bool:
        try:
            delivery = self._notifier.send_alert(
                email, analysis, gmail_url=gmail_url
            )
        except TelegramError as exc:
            LOGGER.warning(
                "Telegram delivery failed for message %s (account %s): %s",
                message_id,
                account,
                exc,
            )
            delivery = None
        if delivery is None:
            self._record_delivery_failure(account, message_id)
            return False
        try:
            self._store.mark_notification_sent(
                account,
                message_id,
                provider_message_id=str(delivery.message_id),
            )
        except StorageError as exc:
            LOGGER.warning(
                "Could not record confirmed delivery for message %s (account %s): %s",
                message_id,
                account,
                exc,
            )
            return False
        return True

    def _record_delivery_failure(self, account: str, message_id: str) -> None:
        try:
            self._store.mark_notification_failed(account, message_id)
        except StorageError as exc:
            LOGGER.warning(
                "Could not record failed delivery for message %s (account %s): %s",
                message_id,
                account,
                exc,
            )

    @staticmethod
    def _stored_email(stored: StoredEmail) -> EmailMessage:
        received: datetime | None = None
        if stored.received_at:
            try:
                received = datetime.fromisoformat(stored.received_at)
            except ValueError:
                received = None
        return EmailMessage(
            message_id=stored.message_id,
            thread_id=stored.thread_id,
            account=stored.account_alias,
            sender_name="",
            sender_email=stored.sender,
            subject=stored.subject,
            snippet="",
            received_at=received,
            rfc822_message_id=stored.rfc822_message_id,
        )

    @staticmethod
    def _classification(data: dict[str, Any]) -> Classification:
        return Classification(
            category=str(data["category"]),
            score=int(data["importance_score"]),
            summary=str(data["summary"]),
            reason=str(data["reason"]),
            required_action=(
                str(data["required_action"]) if data.get("required_action") else None
            ),
            deadline=str(data["deadline"]) if data.get("deadline") else None,
            confidence=float(data.get("confidence") or 0),
            model="",
        )
