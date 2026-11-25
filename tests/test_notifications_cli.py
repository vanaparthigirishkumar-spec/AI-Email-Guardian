"""Tests for the read-only `notifications pending` queue command."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from email_guardian.cli import main
from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import EmailMessage
from email_guardian.logging_setup import shutdown_logging
from email_guardian.storage import (
    EmailStore,
    NotificationStatus,
    PendingNotification,
)
from email_guardian.summarizer import Classification

FAKE_TOKEN = "444444:QUEUE_FAKE_TOKEN_DO_NOT_PRINT_987654321"
BODY_SECRET = "PRIVATE BODY MUST NEVER BE SHOWN"
SNIPPET_SECRET = "PRIVATE SNIPPET MUST NEVER BE SHOWN"


def make_settings(tmp: Path) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
        "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
        "TELEGRAM_CHAT_ID": "888001",
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def make_email(account: str, message_id: str, sender: str, subject: str) -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        account=account,
        sender_name="",
        sender_email=sender,
        subject=subject,
        snippet=SNIPPET_SECRET,
        received_at=datetime(2026, 11, 25, 9, 30, tzinfo=timezone.utc),
    )


def make_classification(
    category: str = "IMPORTANT",
    score: int = 8,
    action: str | None = "Confirm the interview slot.",
) -> Classification:
    return Classification(
        category=category,
        score=score,
        summary="A short summary.",
        reason="The message asks for a reply.",
        required_action=action,
        deadline=None,
        confidence=0.9,
        model="llama3.2",
    )


class PendingQueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._cleanup)
        self.settings = make_settings(self.tmp)
        self.store = EmailStore(self.settings)
        self.store.initialize()
        self.addCleanup(self.store.close)

    def _cleanup(self) -> None:
        shutdown_logging()
        self._tmp.cleanup()

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        env = {
            "GMAIL_ACCOUNTS": "personal,work",
            "DATABASE_PATH": str(self.tmp / "data" / "emails.db"),
            "LOG_DIR": str(self.tmp / "logs"),
            "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
            "TELEGRAM_CHAT_ID": "888001",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue() + stderr.getvalue()

    def seed_queue(self) -> None:
        self.store.record_message(
            make_email("personal", "p1", "alice@example.com", "Interview invitation"),
            make_classification(),
            notification_required=True,
        )
        self.store.record_message(
            make_email("work", "w1", "recruiter@example.com", "Job alert"),
            make_classification(action=None),
            notification_required=True,
        )
        self.store.record_message(
            make_email("personal", "p2", "bob@example.com", "Family update"),
            make_classification("NORMAL", 2, action=None),
            notification_required=False,
        )
        self.store.record_message(
            make_email("personal", "p3", "carol@example.com", "Security alert"),
            make_classification("URGENT", 9),
            notification_required=True,
        )
        self.store.mark_notification_sent("personal", "p3", provider_message_id="55")
        self.store.set_sync_checkpoint(
            "personal", history_id="history-1", last_message_id="p3"
        )


class PendingQueueCliTest(PendingQueueTestCase):
    def test_pending_command_displays_all_requested_fields(self) -> None:
        self.seed_queue()

        code, output = self.run_cli(["notifications", "pending"])

        self.assertEqual(code, 0)
        self.assertIn("Pending notifications: 2", output)
        self.assertIn("Account: personal | Message id: p1", output)
        self.assertIn("Account: work | Message id: w1", output)
        self.assertIn("Sender : alice@example.com", output)
        self.assertIn("Subject: Interview invitation", output)
        self.assertIn("Subject: Job alert", output)
        self.assertIn("AI classification: IMPORTANT | Score: 8/10", output)
        self.assertIn("Date   : 2026-11-25T09:30:00+00:00", output)
        self.assertIn("Action : Confirm the interview slot.", output)
        self.assertIn("Action : (none recorded)", output)
        self.assertNotIn("Family update", output)
        self.assertNotIn("Security alert", output)
        self.assertNotIn(BODY_SECRET, output)
        self.assertNotIn(SNIPPET_SECRET, output)
        self.assertNotIn(FAKE_TOKEN, output)
        self.assertIn("no notifications were sent", output)

    def test_pending_command_with_empty_queue(self) -> None:
        code, output = self.run_cli(["notifications", "pending"])

        self.assertEqual(code, 0)
        self.assertIn("Pending notifications: 0", output)
        self.assertIn("No pending notifications.", output)
        self.assertNotIn(FAKE_TOKEN, output)

    def test_pending_command_is_read_only(self) -> None:
        self.seed_queue()
        before_statuses = [
            self.store.get_notification_status("personal", "p1"),
            self.store.get_notification_status("work", "w1"),
            self.store.get_notification_status("personal", "p3"),
        ]
        before_count = self.store.count_pending_notifications()
        before_checkpoint = self.store.get_sync_checkpoint("personal")
        before_recent = len(self.store.recent(limit=100))

        with patch("email_guardian.notifier.requests.get") as get, patch(
            "email_guardian.notifier.requests.post"
        ) as post, patch("email_guardian.cli.EmailGuardianAgent") as agent_factory:
            code, output = self.run_cli(["notifications", "pending"])

        self.assertEqual(code, 0)
        self.assertIn("Pending notifications: 2", output)
        get.assert_not_called()
        post.assert_not_called()
        agent_factory.assert_not_called()
        self.assertEqual(
            [
                self.store.get_notification_status("personal", "p1"),
                self.store.get_notification_status("work", "w1"),
                self.store.get_notification_status("personal", "p3"),
            ],
            before_statuses,
        )
        self.assertEqual(self.store.count_pending_notifications(), before_count)
        self.assertEqual(self.store.get_sync_checkpoint("personal"), before_checkpoint)
        self.assertEqual(len(self.store.recent(limit=100)), before_recent)
        self.assertEqual(
            self.store.get_notification_status("personal", "p3"),
            NotificationStatus.SENT,
        )


class PendingStorageTest(PendingQueueTestCase):
    def test_detailed_query_returns_metadata_without_body_fields(self) -> None:
        self.seed_queue()

        items = self.store.pending_notifications_detailed()

        self.assertEqual(self.store.count_pending_notifications(), 2)
        self.assertEqual(len(items), 2)
        self.assertTrue(all(isinstance(item, PendingNotification) for item in items))
        self.assertEqual([item.account_alias for item in items], ["personal", "work"])
        self.assertEqual([item.message_id for item in items], ["p1", "w1"])
        self.assertEqual(items[0].sender, "alice@example.com")
        self.assertEqual(items[0].subject, "Interview invitation")
        self.assertEqual(items[0].category, "IMPORTANT")
        self.assertEqual(items[0].score, 8)
        self.assertEqual(items[0].email_date, "2026-11-25T09:30:00+00:00")
        self.assertEqual(items[0].required_action, "Confirm the interview slot.")
        self.assertIsNone(items[1].required_action)
        self.assertNotIn("body", vars(items[0]))
        self.assertNotIn(BODY_SECRET, repr(items))


class SuppressionCliTest(PendingQueueTestCase):
    def snapshot(self) -> tuple[int, int, object, int]:
        return (
            self.store.count_pending_notifications(),
            len(self.store.recent(limit=100)),
            self.store.get_sync_checkpoint("personal"),
            len(self.store.pending_notifications_detailed()),
        )

    def test_suppress_removes_one_alert_and_persists_after_restart(self) -> None:
        self.seed_queue()
        before = self.snapshot()

        code, output = self.run_cli(
            ["notifications", "suppress", "work", "w1"]
        )

        self.assertEqual(code, 0)
        self.assertIn("[OK  ] Suppressed notification", output)
        self.assertIn("Status: NOT_REQUIRED", output)
        self.assertNotIn(FAKE_TOKEN, output)
        self.assertEqual(
            self.store.get_notification_status("work", "w1"),
            NotificationStatus.NOT_REQUIRED,
        )
        self.assertEqual(self.store.count_pending_notifications(), before[0] - 1)
        self.assertEqual(
            [item.message_id for item in self.store.pending_notifications_detailed()],
            ["p1"],
        )
        self.assertEqual(self.snapshot()[1:3], before[1:3])
        self.assertIsNotNone(self.store.get_classification("work", "w1"))

        reopened = EmailStore(self.settings)
        reopened.initialize()
        try:
            self.assertEqual(
                reopened.get_notification_status("work", "w1"),
                NotificationStatus.NOT_REQUIRED,
            )
        finally:
            reopened.close()

    def test_suppress_is_idempotent_and_creates_no_duplicate_rows(self) -> None:
        self.seed_queue()
        before = self.snapshot()

        first_code, _ = self.run_cli(["notifications", "suppress", "personal", "p1"])
        second_code, _ = self.run_cli(["notifications", "suppress", "personal", "p1"])

        self.assertEqual(first_code, 0)
        self.assertEqual(second_code, 0)
        self.assertEqual(
            self.store.get_notification_status("personal", "p1"),
            NotificationStatus.NOT_REQUIRED,
        )
        self.assertEqual(
            self.snapshot(), (before[0] - 1, before[1], before[2], before[3] - 1)
        )
        self.assertIsNotNone(self.store.get_classification("personal", "p1"))

    def test_suppress_unknown_or_delivered_alert_is_refused(self) -> None:
        self.seed_queue()
        before = self.snapshot()

        missing_code, missing_output = self.run_cli(
            ["notifications", "suppress", "work", "does-not-exist"]
        )
        sent_code, sent_output = self.run_cli(
            ["notifications", "suppress", "personal", "p3"]
        )

        self.assertEqual(missing_code, 1)
        self.assertIn("No notification found", missing_output)
        self.assertEqual(sent_code, 1)
        self.assertIn("already delivered", sent_output)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(
            self.store.get_notification_status("personal", "p3"),
            NotificationStatus.SENT,
        )

    def test_unsuppress_restores_an_eligible_alert_to_pending(self) -> None:
        self.seed_queue()
        before = self.snapshot()
        self.assertEqual(
            self.run_cli(["notifications", "suppress", "personal", "p1"])[0], 0
        )

        code, output = self.run_cli(
            ["notifications", "unsuppress", "personal", "p1"]
        )

        self.assertEqual(code, 0)
        self.assertIn("[OK  ] Restored notification", output)
        self.assertEqual(
            self.store.get_notification_status("personal", "p1"),
            NotificationStatus.PENDING,
        )
        self.assertEqual(self.snapshot(), before)

    def test_unsuppress_refuses_an_ineligible_email(self) -> None:
        self.seed_queue()
        before = self.snapshot()

        code, output = self.run_cli(
            ["notifications", "unsuppress", "personal", "p2"]
        )

        self.assertEqual(code, 1)
        self.assertIn("not eligible for notification", output)
        self.assertEqual(
            self.store.get_notification_status("personal", "p2"),
            NotificationStatus.NOT_REQUIRED,
        )
        self.assertEqual(self.snapshot(), before)

    def test_suppression_commands_do_not_send_or_touch_gmail(self) -> None:
        self.seed_queue()
        before = self.snapshot()

        with patch("email_guardian.notifier.requests.get") as get, patch(
            "email_guardian.notifier.requests.post"
        ) as post, patch("email_guardian.cli.GmailClient") as gmail, patch(
            "email_guardian.cli.EmailGuardianAgent"
        ) as agent_factory:
            self.assertEqual(
                self.run_cli(["notifications", "suppress", "personal", "p1"])[0], 0
            )
            self.assertEqual(
                self.run_cli(["notifications", "unsuppress", "personal", "p1"])[0], 0
            )

        get.assert_not_called()
        post.assert_not_called()
        gmail.assert_not_called()
        agent_factory.assert_not_called()
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
