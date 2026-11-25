"""Mocked end-to-end tests for the Gmail -> AI -> SQLite -> Telegram pipeline."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from email_guardian.agent import CycleReport, EmailGuardianAgent
from email_guardian.cli import main
from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import (
    EmailBody,
    EmailMessage,
    GmailApiError,
    GmailProfile,
)
from email_guardian.logging_setup import setup_logging, shutdown_logging
from email_guardian.notifier import TelegramDelivery, TelegramNotifier
from email_guardian.storage import EmailStore, NotificationStatus
from email_guardian.summarizer import Classification, OllamaError

FAKE_TOKEN = "333333:PIPELINE_FAKE_TOKEN_DO_NOT_PRINT_987654321"
CHAT_ID = "777001"
BODY_SECRET = "TOP SECRET BODY MUST NEVER BE SENT OR STORED"
THRESHOLD = 7


def make_settings(tmp: Path) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
        "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
        "TELEGRAM_CHAT_ID": CHAT_ID,
        "IMPORTANCE_THRESHOLD": str(THRESHOLD),
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def make_email(account: str, message_id: str, subject: str) -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        account=account,
        sender_name="Alice",
        sender_email="alice@example.com",
        subject=subject,
        snippet="PRIVATE SNIPPET MUST NOT BE SENT",
        received_at=datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc),
    )


def make_classification(category: str, score: int) -> Classification:
    return Classification(
        category=category,
        score=score,
        summary="A short factual summary.",
        reason="The message states a clear next step.",
        required_action="Review the message.",
        deadline=None,
        confidence=0.9,
        model="llama3.2",
    )


class FakeAnalyzer:
    def __init__(
        self,
        results: dict[str, Classification],
        *,
        fail_ids: tuple[str, ...] = (),
        unavailable: bool = False,
    ) -> None:
        self.results = results
        self.fail_ids = set(fail_ids)
        self.unavailable = unavailable
        self.analyzed: list[tuple[str, str, str]] = []

    def ensure_available(self) -> None:
        if self.unavailable:
            raise OllamaError("Ollama is not reachable")

    def analyze(self, email: EmailMessage, body: str) -> Classification:
        self.analyzed.append((email.account, email.message_id, body))
        if email.message_id in self.fail_ids:
            raise OllamaError("analysis failed")
        return self.results[email.message_id]


class FakeNotifier:
    def __init__(self, *, fail_ids: tuple[str, ...] = ()) -> None:
        self.fail_ids = set(fail_ids)
        self.sent: list[tuple[str, str]] = []
        self.counter = 0

    def send_alert(
        self,
        email: EmailMessage,
        analysis: Classification,
        *,
        gmail_url: str | None = None,
    ) -> Any:
        self.sent.append((email.account, email.message_id))
        if email.message_id in self.fail_ids:
            return None
        self.counter += 1
        return TelegramDelivery(message_id=self.counter, chat_id=CHAT_ID)


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._cleanup)
        self.settings = make_settings(self.tmp)
        self.store = EmailStore(self.settings)
        self.store.initialize()
        self.addCleanup(self.store.close)
        self.clients: list[MagicMock] = []

    def _cleanup(self) -> None:
        shutdown_logging()
        self._tmp.cleanup()

    @staticmethod
    def _raise_body_error(message_id: str) -> EmailBody:
        raise GmailApiError(f"body unavailable for {message_id}", status=500)

    def client_factory(
        self,
        messages: dict[str, list[EmailMessage]],
        *,
        body_failures: tuple[str, ...] = (),
        account_failures: tuple[str, ...] = (),
        body_text: str = BODY_SECRET,
    ) -> Any:
        def factory(settings: Settings, account: str) -> MagicMock:
            client = MagicMock()
            if account in account_failures:
                client.fetch_recent.side_effect = GmailApiError(
                    "Gmail API failed", status=500
                )
            else:
                client.get_profile.return_value = GmailProfile(
                    account=account,
                    email=f"{account}@example.com",
                    messages_total=0,
                    threads_total=0,
                )
                client.fetch_recent.return_value = list(messages.get(account, ()))
                client.fetch_body.side_effect = lambda message_id: (
                    self._raise_body_error(message_id)
                    if message_id in body_failures
                    else EmailBody(
                        message_id=message_id,
                        text=body_text,
                        content_type="text/plain",
                        truncated=False,
                    )
                )
            self.clients.append(client)
            return client

        return factory

    def agent(
        self,
        *,
        factory: Any,
        analyzer: FakeAnalyzer,
        notifier: Any,
    ) -> EmailGuardianAgent:
        return EmailGuardianAgent(
            self.settings,
            store=self.store,
            analyzer=analyzer,
            notifier=notifier,
            client_factory=factory,
        )


class FullPipelineTest(PipelineTestCase):
    def test_both_accounts_are_processed_with_selective_notifications(self) -> None:
        personal = [
            make_email("personal", "p1", "Security alert"),
            make_email("personal", "p2", "Interview invitation"),
            make_email("personal", "p3", "Low priority notice"),
            make_email("personal", "p4", "Family update"),
            make_email("personal", "p5", "Half price sale"),
            make_email("personal", "p6", "Prize winner"),
        ]
        work = [make_email("work", "w1", "Job alert: engineer openings")]
        messages = {"personal": personal, "work": work}
        results = {
            "p1": make_classification("URGENT", 9),
            "p2": make_classification("IMPORTANT", 8),
            "p3": make_classification("IMPORTANT", 5),
            "p4": make_classification("NORMAL", 2),
            "p5": make_classification("PROMOTIONAL", 3),
            "p6": make_classification("SPAM", 1),
            "w1": make_classification("IMPORTANT", THRESHOLD),
        }
        analyzer = FakeAnalyzer(results)
        notifier = FakeNotifier()
        factory = self.client_factory(messages)
        setup_logging(self.settings.log_dir, self.settings.log_level)

        report = self.agent(
            factory=factory, analyzer=analyzer, notifier=notifier
        ).run_once()

        self.assertFalse(report.has_failures)
        self.assertEqual(report.accounts_processed, 2)
        self.assertEqual(report.fetched, 7)
        self.assertEqual(report.stored, 7)
        self.assertEqual(report.analyzed, 7)
        self.assertEqual(
            notifier.sent, [("personal", "p1"), ("personal", "p2"), ("work", "w1")]
        )
        self.assertEqual(report.notifications_sent, 3)
        self.assertEqual(report.notifications_failed, 0)
        self.assertEqual(len(analyzer.analyzed), 7)
        self.assertTrue(all(body == BODY_SECRET for _, _, body in analyzer.analyzed))

        for account, message_id in (
            ("personal", "p1"),
            ("personal", "p2"),
            ("work", "w1"),
        ):
            self.assertEqual(
                self.store.get_notification_status(account, message_id),
                NotificationStatus.SENT,
            )
        for message_id in ("p3", "p4", "p5", "p6"):
            self.assertEqual(
                self.store.get_notification_status("personal", message_id),
                NotificationStatus.NOT_REQUIRED,
            )

        personal_checkpoint = self.store.get_sync_checkpoint("personal")
        work_checkpoint = self.store.get_sync_checkpoint("work")
        self.assertIsNotNone(personal_checkpoint)
        self.assertIsNotNone(work_checkpoint)
        assert personal_checkpoint is not None and work_checkpoint is not None
        self.assertEqual(personal_checkpoint.last_message_id, "p6")
        self.assertEqual(work_checkpoint.last_message_id, "w1")
        self.assertTrue(
            all(
                client.fetch_recent.call_args.kwargs["limit"]
                == self.settings.gmail_fetch_limit
                for client in self.clients
            )
        )

        shutdown_logging()
        log_text = (self.tmp / "logs" / "email_guardian.log").read_text(encoding="utf-8")
        self.assertNotIn(BODY_SECRET, log_text)
        self.assertNotIn(FAKE_TOKEN, log_text)
        self.assertNotIn("PRIVATE SNIPPET", log_text)

    def test_job_related_email_is_stored_as_notifiable(self) -> None:
        messages = {"personal": [make_email("personal", "j1", "Job alert")]}
        analyzer = FakeAnalyzer({"j1": make_classification("IMPORTANT", THRESHOLD)})
        notifier = FakeNotifier()

        report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=notifier,
        ).run_once()

        stored = self.store.get_email("personal", "j1")
        assert stored is not None
        self.assertEqual(stored.category, "IMPORTANT")
        self.assertEqual(stored.score, THRESHOLD)
        self.assertTrue(stored.notified)
        self.assertEqual(notifier.sent, [("personal", "j1")])
        self.assertEqual(report.notifications_sent, 1)


class DuplicateAndRecoveryTest(PipelineTestCase):
    def _single_message_setup(self) -> tuple[dict[str, list[EmailMessage]], FakeAnalyzer]:
        messages = {"personal": [make_email("personal", "m1", "Interview invitation")]}
        analyzer = FakeAnalyzer({"m1": make_classification("IMPORTANT", 8)})
        return messages, analyzer

    def test_second_cycle_skips_duplicates_and_does_not_notify_twice(self) -> None:
        messages, analyzer = self._single_message_setup()
        notifier = FakeNotifier()
        agent = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=notifier,
        )

        first = agent.run_once()
        second = agent.run_once()

        self.assertEqual(first.stored, 1)
        self.assertEqual(second.stored, 0)
        self.assertEqual(second.duplicates, 1)
        self.assertEqual(notifier.sent, [("personal", "m1")])
        self.assertEqual(second.notifications_sent, 0)
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.SENT,
        )

    def test_dry_run_stores_results_without_delivery_and_recovers_later(self) -> None:
        messages, analyzer = self._single_message_setup()
        dry_notifier = FakeNotifier()
        real_notifier = FakeNotifier()

        dry_report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=dry_notifier,
        ).run_once(dry_run=True)

        self.assertEqual(dry_report.stored, 1)
        self.assertEqual(dry_report.dry_run_pending, 1)
        self.assertEqual(dry_report.notifications_sent, 0)
        self.assertEqual(dry_notifier.sent, [])
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.PENDING,
        )

        recovery_report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=real_notifier,
        ).run_once()

        self.assertEqual(recovery_report.recovered, 1)
        self.assertEqual(real_notifier.sent, [("personal", "m1")])
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.SENT,
        )
        self.assertEqual(recovery_report.duplicates, 1)

    def test_failed_delivery_is_recorded_without_success(self) -> None:
        messages = {"personal": [make_email("personal", "m2", "Urgent alert")]}
        analyzer = FakeAnalyzer({"m2": make_classification("URGENT", 9)})
        notifier = FakeNotifier(fail_ids=("m2",))

        report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=notifier,
        ).run_once()

        self.assertTrue(report.has_failures)
        self.assertEqual(report.notifications_sent, 0)
        self.assertEqual(report.notifications_failed, 1)
        self.assertEqual(notifier.sent, [("personal", "m2")])
        self.assertEqual(
            self.store.get_notification_status("personal", "m2"),
            NotificationStatus.FAILED,
        )
    def test_suppressed_alert_is_never_retried_by_the_agent(self) -> None:
        messages, analyzer = self._single_message_setup()
        notifier = FakeNotifier()

        dry_report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=notifier,
        ).run_once(dry_run=True)
        self.assertEqual(dry_report.stored, 1)
        self.assertEqual(
            self.store.suppress_notification("personal", "m1"),
            NotificationStatus.NOT_REQUIRED,
        )

        next_report = self.agent(
            factory=self.client_factory(messages),
            analyzer=analyzer,
            notifier=notifier,
        ).run_once()

        self.assertEqual(next_report.recovered, 0)
        self.assertEqual(next_report.notifications_sent, 0)
        self.assertEqual(notifier.sent, [])
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.NOT_REQUIRED,
        )


class FailureHandlingTest(PipelineTestCase):
    def test_partial_failures_do_not_block_other_messages_or_accounts(self) -> None:
        messages = {
            "personal": [
                make_email("personal", "bad-body", "Body failure"),
                make_email("personal", "bad-ai", "AI failure"),
                make_email("personal", "good", "Good message"),
            ],
            "work": [make_email("work", "w1", "Job alert")],
        }
        analyzer = FakeAnalyzer(
            {
                "good": make_classification("IMPORTANT", 8),
                "w1": make_classification("IMPORTANT", THRESHOLD),
            },
            fail_ids=("bad-ai",),
        )
        notifier = FakeNotifier()

        report = self.agent(
            factory=self.client_factory(messages, body_failures=("bad-body",)),
            analyzer=analyzer,
            notifier=notifier,
        ).run_once()

        self.assertEqual(report.accounts_processed, 2)
        self.assertEqual(report.stored, 2)
        self.assertEqual(report.failed, 2)
        self.assertTrue(report.has_failures)
        self.assertEqual(notifier.sent, [("personal", "good"), ("work", "w1")])
        self.assertIsNone(self.store.get_email("personal", "bad-body"))
        self.assertIsNone(self.store.get_email("personal", "bad-ai"))
        self.assertIsNone(
            self.store.get_notification_status("personal", "bad-body")
        )

    def test_failed_account_keeps_its_existing_checkpoint(self) -> None:
        self.store.set_sync_checkpoint(
            "personal", history_id="history-1", last_message_id="old-message"
        )
        messages = {"personal": [make_email("personal", "p1", "Never fetched")]}
        analyzer = FakeAnalyzer({})

        report = self.agent(
            factory=self.client_factory(messages, account_failures=("personal",)),
            analyzer=analyzer,
            notifier=FakeNotifier(),
        ).run_once()

        self.assertEqual(report.account_failures, 1)
        self.assertEqual(report.accounts_processed, 1)
        checkpoint = self.store.get_sync_checkpoint("personal")
        assert checkpoint is not None
        self.assertEqual(checkpoint.history_id, "history-1")
        self.assertEqual(checkpoint.last_message_id, "old-message")
        self.assertFalse(analyzer.analyzed)
        self.assertEqual(report.checkpoints_updated, 1)

    def test_checkpoint_keeps_history_id_when_no_new_mail_arrives(self) -> None:
        self.store.set_sync_checkpoint("personal", history_id="history-7")
        analyzer = FakeAnalyzer({})

        report = self.agent(
            factory=self.client_factory({}),
            analyzer=analyzer,
            notifier=FakeNotifier(),
        ).run_once()

        self.assertEqual(report.checkpoints_updated, 2)
        checkpoint = self.store.get_sync_checkpoint("personal")
        assert checkpoint is not None
        self.assertEqual(checkpoint.history_id, "history-7")

    def test_unavailable_ollama_stops_before_touching_gmail(self) -> None:
        analyzer = FakeAnalyzer({}, unavailable=True)
        factory = self.client_factory({})

        with self.assertRaises(OllamaError):
            self.agent(
                factory=factory,
                analyzer=analyzer,
                notifier=FakeNotifier(),
            ).run_once()

        self.assertFalse(self.clients)
        self.assertEqual(self.store.list_accounts(), ())


class RealNotifierPayloadTest(PipelineTestCase):
    def test_notification_payload_contains_metadata_only(self) -> None:
        messages = {"personal": [make_email("personal", "p1", "Interview invitation")]}
        analyzer = FakeAnalyzer({"p1": make_classification("IMPORTANT", 8)})
        notifier = TelegramNotifier(self.settings)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": 91, "chat": {"id": int(CHAT_ID)}},
        }

        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            report = self.agent(
                factory=self.client_factory(messages),
                analyzer=analyzer,
                notifier=notifier,
            ).run_once()

        self.assertEqual(report.notifications_sent, 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], CHAT_ID)
        self.assertIn("Interview invitation", payload["text"])
        self.assertIn("A short factual summary.", payload["text"])
        self.assertNotIn(BODY_SECRET, payload["text"])
        self.assertNotIn("PRIVATE SNIPPET", payload["text"])
        self.assertNotIn(FAKE_TOKEN, str(payload))
        self.assertEqual(
            self.store.get_notification_status("personal", "p1"),
            NotificationStatus.SENT,
        )


class ProcessCliTest(PipelineTestCase):
    def run_cli(
        self, argv: list[str], *, accounts: str = "personal,work"
    ) -> tuple[int, str]:
        env = {
            "GMAIL_ACCOUNTS": accounts,
            "DATABASE_PATH": str(self.tmp / "data" / "emails.db"),
            "LOG_DIR": str(self.tmp / "logs"),
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue() + stderr.getvalue()

    def test_process_once_prints_cycle_summary_and_returns_zero(self) -> None:
        agent = MagicMock()
        agent.run_once.return_value = CycleReport(
            accounts_processed=2,
            fetched=3,
            analyzed=3,
            stored=3,
            notifications_sent=2,
            checkpoints_updated=2,
        )

        with patch("email_guardian.cli.EmailGuardianAgent", return_value=agent):
            code, output = self.run_cli(["process", "once"])

        self.assertEqual(code, 0)
        self.assertIn("Processing cycle complete", output)
        self.assertIn("Accounts: 2", output)
        self.assertIn("Sent: 2", output)
        agent.run_once.assert_called_once_with(dry_run=False)

    def test_process_once_dry_run_forwards_the_flag(self) -> None:
        agent = MagicMock()
        agent.run_once.return_value = CycleReport(dry_run_pending=1)

        with patch("email_guardian.cli.EmailGuardianAgent", return_value=agent):
            code, output = self.run_cli(["process", "once", "--dry-run"])

        self.assertEqual(code, 0)
        self.assertIn("Dry run complete", output)
        self.assertIn("no Telegram notifications were delivered", output)
        agent.run_once.assert_called_once_with(dry_run=True)

    def test_process_once_reports_partial_failures(self) -> None:
        agent = MagicMock()
        agent.run_once.return_value = CycleReport(failed=1)

        with patch("email_guardian.cli.EmailGuardianAgent", return_value=agent):
            code, output = self.run_cli(["process", "once"])

        self.assertEqual(code, 1)
        self.assertIn("Failed: 1", output)

    def test_process_once_requires_configured_accounts(self) -> None:
        with patch("email_guardian.cli.EmailGuardianAgent") as factory:
            code, output = self.run_cli(["process", "once"], accounts="")

        self.assertEqual(code, 1)
        self.assertIn("No Gmail accounts configured", output)
        factory.assert_not_called()

    def test_process_once_reports_ollama_failure_without_a_traceback(self) -> None:
        agent = MagicMock()
        agent.run_once.side_effect = OllamaError("Ollama is not reachable")

        with patch("email_guardian.cli.EmailGuardianAgent", return_value=agent):
            code, output = self.run_cli(["process", "once"])

        self.assertEqual(code, 1)
        self.assertIn("Processing cycle could not start: Ollama is not reachable", output)


if __name__ == "__main__":
    unittest.main()
