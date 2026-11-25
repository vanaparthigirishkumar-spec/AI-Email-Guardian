"""Bounded, mocked tests for continuous monitoring."""

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import unquote

from email_guardian.agent import CycleReport, EmailGuardianAgent
from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import (
    EmailBody,
    EmailMessage,
    GmailApiError,
    GmailClient,
    GmailProfile,
)
from email_guardian.logging_setup import shutdown_logging
from email_guardian.notifier import TelegramDelivery, TelegramNotifier, gmail_message_url
from email_guardian.storage import EmailStore, NotificationStatus
from email_guardian.summarizer import Classification, OllamaError

FAKE_TOKEN = "555555:WATCH_FAKE_TOKEN_DO_NOT_PRINT_987654321"
CHAT_ID = 999001
BODY_SECRET = "PRIVATE WATCH BODY MUST NOT BE LOGGED"
PERSONAL_EMAIL = "vanaparthigirishkumar@gmail.com"
WORK_EMAIL = "vanaparthigirish254@gmail.com"


class RecordingEvent(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self.is_set()


def make_settings(tmp: Path) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
        "TELEGRAM_BOT_TOKEN": FAKE_TOKEN,
        "TELEGRAM_CHAT_ID": str(CHAT_ID),
        "IMPORTANCE_THRESHOLD": "7",
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def make_email(account: str, message_id: str, rfc822_id: str) -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        account=account,
        sender_name="Alice",
        sender_email="alice@example.com",
        subject="Interview invitation",
        snippet="PRIVATE SNIPPET",
        received_at=datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc),
        rfc822_message_id=rfc822_id,
    )


def make_analysis() -> Classification:
    return Classification(
        category="IMPORTANT",
        score=8,
        summary="An interview is scheduled.",
        reason="The email invites the recipient to an interview.",
        required_action="Confirm the interview slot.",
        deadline=None,
        confidence=0.9,
        model="llama3.2",
    )


class FakeAnalyzer:
    def __init__(self, results: dict[str, Classification]) -> None:
        self.results = results

    def ensure_available(self) -> None:
        return None

    def analyze(self, email: EmailMessage, body: str) -> Classification:
        return self.results[email.message_id]


class FakeNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str | None]] = []

    def send_alert(
        self,
        email: EmailMessage,
        analysis: Classification,
        *,
        gmail_url: str | None = None,
    ) -> TelegramDelivery | None:
        self.sent.append((email.account, email.message_id, gmail_url))
        return TelegramDelivery(message_id=len(self.sent), chat_id=str(CHAT_ID))


class WatchTestCase(unittest.TestCase):
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

    def make_agent(
        self,
        *,
        messages: dict[str, list[EmailMessage]] | None = None,
        profiles: dict[str, str] | None = None,
        failing_accounts: tuple[str, ...] = (),
        notifier: Any = None,
        authenticated_emails: dict[str, str] | None = None,
    ) -> EmailGuardianAgent:
        messages = messages or {}
        profiles = {
            "personal": PERSONAL_EMAIL,
            "work": WORK_EMAIL,
            **(profiles or {}),
        }
        authenticated_emails = authenticated_emails or {}

        def factory(settings: Settings, account: str) -> MagicMock:
            client = MagicMock()
            if account in failing_accounts:
                client.get_profile.side_effect = GmailApiError("Gmail failed", status=500)
            else:
                email = profiles.get(account, "")
                client.authenticated_email.return_value = authenticated_emails.get(
                    account, ""
                )
                client.get_profile.return_value = GmailProfile(
                    account=account,
                    email=email,
                    messages_total=len(messages.get(account, ())),
                    threads_total=0,
                )
            client.fetch_recent.return_value = list(messages.get(account, ()))
            client.fetch_body.side_effect = lambda message_id: EmailBody(
                message_id=message_id,
                text=BODY_SECRET,
                content_type="text/plain",
                truncated=False,
            )
            return client

        return EmailGuardianAgent(
            self.settings,
            store=self.store,
            analyzer=FakeAnalyzer({"m1": make_analysis(), "w1": make_analysis()}),
            notifier=notifier or FakeNotifier(),
            client_factory=factory,
        )


class WatchLoopTest(WatchTestCase):
    def test_polls_repeatedly_until_stop_event(self) -> None:
        agent = self.make_agent()
        event = threading.Event()
        calls: list[int] = []

        def cycle(*, dry_run: bool = False) -> CycleReport:
            calls.append(1)
            if len(calls) == 3:
                event.set()
            return CycleReport()

        with patch.object(agent, "run_once", side_effect=cycle):
            agent.run_forever(stop_event=event, interval_seconds=0)

        self.assertEqual(len(calls), 3)

    def test_uses_the_configured_poll_interval(self) -> None:
        agent = self.make_agent()
        event = RecordingEvent()
        calls: list[int] = []

        def cycle(*, dry_run: bool = False) -> CycleReport:
            calls.append(1)
            if len(calls) == 2:
                event.set()
            return CycleReport()

        with patch.object(agent, "run_once", side_effect=cycle):
            agent.run_forever(stop_event=event)

        self.assertEqual(event.waits, [self.settings.poll_interval_seconds] * 2)
        self.assertEqual(self.settings.poll_interval_seconds, 60)

    def test_service_and_unexpected_errors_do_not_stop_polling(self) -> None:
        agent = self.make_agent()
        event = threading.Event()
        calls: list[int] = []

        def cycle(*, dry_run: bool = False) -> CycleReport:
            calls.append(1)
            if len(calls) == 1:
                raise OllamaError("Ollama is not reachable")
            if len(calls) == 2:
                raise RuntimeError("unexpected")
            event.set()
            return CycleReport()

        with patch.object(agent, "run_once", side_effect=cycle):
            agent.run_forever(stop_event=event, interval_seconds=0)

        self.assertEqual(len(calls), 3)

    def test_ctrl_c_stops_the_loop_cleanly(self) -> None:
        agent = self.make_agent()

        with patch.object(agent, "run_once", side_effect=KeyboardInterrupt):
            agent.run_forever(interval_seconds=0)

    def test_one_failing_account_does_not_block_the_other(self) -> None:
        messages = {"work": [make_email("work", "w1", "<w1@example.com>")]}
        notifier = FakeNotifier()
        agent = self.make_agent(
            messages=messages,
            profiles={"work": "work@example.com"},
            failing_accounts=("personal",),
            notifier=notifier,
        )

        report = agent.run_once()

        self.assertEqual(report.account_failures, 1)
        self.assertEqual(report.accounts_processed, 1)
        self.assertEqual(len(notifier.sent), 1)
        self.assertEqual(notifier.sent[0][:2], ("work", "w1"))
        self.assertIn("rfc822msgid", notifier.sent[0][2] or "")
        self.assertEqual(
            self.store.get_notification_status("work", "w1"),
            NotificationStatus.SENT,
        )


class WatchDeliveryTest(WatchTestCase):
    def test_alert_contains_account_label_and_rfc822_gmail_link(self) -> None:
        messages = {
            "personal": [make_email("personal", "m1", "<p1@example.com>")],
            "work": [make_email("work", "w1", "<w1@example.com>")],
        }
        notifier = TelegramNotifier(self.settings)
        responses = []
        for message_id in (1, 2):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "ok": True,
                "result": {
                    "message_id": message_id,
                    "chat": {"id": CHAT_ID},
                },
            }
            responses.append(response)

        with patch(
            "email_guardian.notifier.requests.post", side_effect=responses
        ) as post:
            report = self.make_agent(
                messages=messages,
                profiles={
                    "personal": "personal@example.com",
                    "work": "work@example.com",
                },
                notifier=notifier,
            ).run_once()

        self.assertEqual(report.notifications_sent, 2)
        texts = [call.kwargs["json"]["text"] for call in post.call_args_list]
        self.assertTrue(any("Account: Personal" in text for text in texts))
        self.assertTrue(any("Account: Work" in text for text in texts))
        raw_urls = []
        urls = []
        for call in post.call_args_list:
            payload = call.kwargs["json"]
            button = payload["reply_markup"]["inline_keyboard"][0][0]
            self.assertEqual(button["text"], "Open in Gmail")
            raw_urls.append(button["url"])
            urls.append(unquote(button["url"]))
        self.assertTrue(
            all(
                url.startswith("https://mail.google.com/mail/u/0/?authuser=")
                for url in raw_urls
            )
        )
        self.assertTrue(
            any(
                "authuser=personal%40example.com" in raw
                and "/u/0/?authuser=personal@example.com"
                "#search/rfc822msgid:p1@example.com" in decoded
                for raw, decoded in zip(raw_urls, urls)
            )
        )
        self.assertTrue(
            any(
                "authuser=work%40example.com" in raw
                and "/u/0/?authuser=work@example.com"
                "#search/rfc822msgid:w1@example.com" in decoded
                for raw, decoded in zip(raw_urls, urls)
            )
        )
        stored = self.store.get_email("work", "w1")
        assert stored is not None
        self.assertEqual(stored.rfc822_message_id, "<w1@example.com>")
        self.assertEqual(stored.account_email, "work@example.com")

    def test_pending_alert_recovered_after_restart_keeps_its_link(self) -> None:
        messages = {"personal": [make_email("personal", "m1", "<p1@example.com>")]}
        profiles = {"personal": "personal@example.com"}

        dry_report = self.make_agent(
            messages=messages, profiles=profiles
        ).run_once(dry_run=True)
        self.assertEqual(dry_report.dry_run_pending, 1)

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": 12, "chat": {"id": CHAT_ID}},
        }
        restarted = self.make_agent(
            messages=messages,
            profiles=profiles,
            notifier=TelegramNotifier(self.settings),
        )
        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            report = restarted.run_once()

        self.assertEqual(report.recovered, 1)
        raw_url = post.call_args.kwargs["json"]["reply_markup"]["inline_keyboard"][0][
            0
        ]["url"]
        self.assertIn("authuser=personal%40example.com", raw_url)
        self.assertEqual(
            unquote(raw_url),
            "https://mail.google.com/mail/u/0/?authuser=personal@example.com"
            "#search/rfc822msgid:p1@example.com",
        )
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.SENT,
        )

    def test_missing_message_id_omits_the_message_button(self) -> None:
        messages = {"work": [make_email("work", "w1", "")]}
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": 4, "chat": {"id": CHAT_ID}},
        }

        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            report = self.make_agent(
                messages=messages,
                profiles={"work": "work@example.com"},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        self.assertEqual(report.notifications_sent, 1)
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("reply_markup", payload)
        self.assertIn("Account: Work", payload["text"])

    def test_notification_text_and_link_never_contain_bodies_or_tokens(self) -> None:
        messages = {"work": [make_email("work", "w1", "<w1@example.com>")]}
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": 3, "chat": {"id": CHAT_ID}},
        }

        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            self.make_agent(
                messages=messages,
                profiles={"work": "work@example.com"},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        payload = post.call_args.kwargs["json"]
        self.assertNotIn(BODY_SECRET, str(payload))
        self.assertNotIn("PRIVATE SNIPPET", str(payload))
        self.assertNotIn(FAKE_TOKEN, str(payload))


class AccountIdentityTest(WatchTestCase):
    @staticmethod
    def _response(message_id: int = 30) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": message_id, "chat": {"id": CHAT_ID}},
        }
        return response

    def test_personal_notification_uses_only_the_personal_address(self) -> None:
        messages = {"personal": [make_email("personal", "m1", "<p1@example.com>")]}

        with patch(
            "email_guardian.notifier.requests.post", return_value=self._response()
        ) as post:
            report = self.make_agent(
                messages=messages,
                profiles={"personal": PERSONAL_EMAIL},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        self.assertEqual(report.notifications_sent, 1)
        payload = post.call_args.kwargs["json"]
        raw_url = payload["reply_markup"]["inline_keyboard"][0][0]["url"]
        decoded = unquote(raw_url)
        self.assertIn("Account: Personal", payload["text"])
        self.assertIn(f"authuser={PERSONAL_EMAIL.replace('@', '%40')}", raw_url)
        self.assertTrue(raw_url.startswith("https://mail.google.com/mail/u/0/?authuser="))
        self.assertNotIn(WORK_EMAIL, decoded)
        self.assertTrue(decoded.endswith("#search/rfc822msgid:p1@example.com"))

    def test_work_notification_uses_only_the_work_address(self) -> None:
        messages = {"work": [make_email("work", "w1", "<w1@example.com>")]}

        with patch(
            "email_guardian.notifier.requests.post", return_value=self._response()
        ) as post:
            report = self.make_agent(
                messages=messages,
                profiles={"work": WORK_EMAIL},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        self.assertEqual(report.notifications_sent, 1)
        payload = post.call_args.kwargs["json"]
        raw_url = payload["reply_markup"]["inline_keyboard"][0][0]["url"]
        decoded = unquote(raw_url)
        self.assertIn("Account: Work", payload["text"])
        self.assertIn(f"authuser={WORK_EMAIL.replace('@', '%40')}", raw_url)
        self.assertTrue(raw_url.startswith("https://mail.google.com/mail/u/0/?authuser="))
        self.assertNotIn(PERSONAL_EMAIL, decoded)
        self.assertTrue(decoded.endswith("#search/rfc822msgid:w1@example.com"))

    def test_recovery_ignores_a_wrong_email_stored_in_the_queue(self) -> None:
        message = make_email("personal", "m1", "<p1@example.com>")
        self.store.record_message(
            message,
            make_analysis(),
            notification_required=True,
            account_email=WORK_EMAIL,
        )
        before = self.store.get_email("personal", "m1")
        assert before is not None

        with patch(
            "email_guardian.notifier.requests.post", return_value=self._response()
        ) as post:
            report = self.make_agent(
                messages={},
                profiles={"personal": PERSONAL_EMAIL},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        self.assertEqual(report.recovered, 1)
        raw_url = post.call_args.kwargs["json"]["reply_markup"]["inline_keyboard"][0][
            0
        ]["url"]
        self.assertIn(f"authuser={PERSONAL_EMAIL.replace('@', '%40')}", raw_url)
        self.assertNotIn(WORK_EMAIL, unquote(raw_url))
        self.assertTrue(unquote(raw_url).endswith("#search/rfc822msgid:p1@example.com"))

        after = self.store.get_email("personal", "m1")
        assert after is not None
        self.assertEqual(after.account_email, WORK_EMAIL)
        self.assertEqual(
            self.store.get_notification_status("personal", "m1"),
            NotificationStatus.SENT,
        )

    def test_recovery_prefers_authenticated_token_identity_over_profile(self) -> None:
        self.store.record_message(
            make_email("personal", "m1", "<p1@example.com>"),
            make_analysis(),
            notification_required=True,
            account_email="",
        )

        with patch(
            "email_guardian.notifier.requests.post", return_value=self._response()
        ) as post:
            report = self.make_agent(
                messages={},
                authenticated_emails={"personal": PERSONAL_EMAIL},
                profiles={"personal": WORK_EMAIL},
                notifier=TelegramNotifier(self.settings),
            ).run_once()

        self.assertEqual(report.recovered, 1)
        raw_url = post.call_args.kwargs["json"]["reply_markup"]["inline_keyboard"][0][
            0
        ]["url"]
        self.assertIn(f"authuser={PERSONAL_EMAIL.replace('@', '%40')}", raw_url)
        self.assertNotIn(WORK_EMAIL, unquote(raw_url))


class GmailLinkTest(unittest.TestCase):
    def test_personal_account_url_uses_numeric_selector_and_authuser(self) -> None:
        url = gmail_message_url("personal@example.com", "<p1@example.com>")

        self.assertTrue(
            url.startswith("https://mail.google.com/mail/u/0/?authuser=")
        )
        self.assertIn("authuser=personal%40example.com", url)
        self.assertEqual(
            unquote(url),
            "https://mail.google.com/mail/u/0/?authuser=personal@example.com"
            "#search/rfc822msgid:p1@example.com",
        )
        self.assertNotIn("/u/personal@example.com/", url)

    def test_work_account_url_selects_the_work_address(self) -> None:
        url = gmail_message_url("work@example.com", "<w1@example.com>")

        self.assertIn("authuser=work%40example.com", url)
        self.assertEqual(
            unquote(url),
            "https://mail.google.com/mail/u/0/?authuser=work@example.com"
            "#search/rfc822msgid:w1@example.com",
        )

    def test_numeric_u0_selector_is_used_for_both_accounts(self) -> None:
        for email in ("personal@example.com", "work@example.com"):
            url = gmail_message_url(email, "<abc@example.com>")
            self.assertIn("/u/0/", url)
            self.assertNotIn(f"/u/{email}/", url)

    def test_search_query_uses_stripped_rfc822_message_id_only(self) -> None:
        url = gmail_message_url("work@example.com", "  <abc123@example.com>  ")
        fragment = unquote(url).split("#search/", 1)[1]

        self.assertEqual(fragment, "rfc822msgid:abc123@example.com")
        self.assertNotIn("gmail-api-id", url)
        self.assertNotIn("thread-1", url)
        self.assertNotIn("<", url)
        self.assertNotIn(">", url)

    def test_special_characters_are_url_encoded(self) -> None:
        url = gmail_message_url(
            "work+alerts@example.com", "<id/with+chars@example.com>"
        )

        self.assertIn("authuser=work%2Balerts%40example.com", url)
        self.assertIn("rfc822msgid%3Aid%2Fwith%2Bchars%40example.com", url)
        self.assertEqual(
            unquote(url),
            "https://mail.google.com/mail/u/0/?authuser=work+alerts@example.com"
            "#search/rfc822msgid:id/with+chars@example.com",
        )

    def test_missing_message_id_or_account_returns_no_url(self) -> None:
        self.assertEqual(gmail_message_url("work@example.com", ""), "")
        self.assertEqual(gmail_message_url("work@example.com", "   "), "")
        self.assertEqual(gmail_message_url("work@example.com", "<>"), "")
        self.assertEqual(gmail_message_url("", "<abc@example.com>"), "")

    def test_gmail_metadata_parser_keeps_rfc822_message_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(Path(tmp))
            client = GmailClient(settings, "personal", service=MagicMock())
            payload = {
                "id": "gmail-id",
                "threadId": "thread",
                "internalDate": "1764048000000",
                "snippet": "snippet",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Alice <alice@example.com>"},
                        {"name": "Subject", "value": "Interview"},
                        {"name": "Date", "value": "Wed, 25 Nov 2026 09:00:00 +0000"},
                        {"name": "Message-ID", "value": "<abc123@example.com>"},
                    ]
                },
            }

            message = client._parse_message("gmail-id", payload)

        self.assertEqual(message.rfc822_message_id, "<abc123@example.com>")
        self.assertEqual(message.message_id, "gmail-id")


if __name__ == "__main__":
    unittest.main()
