"""Tests for token-safe Telegram integration."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import requests

from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import EmailMessage
from email_guardian.notifier import (
    API_BASE_URL,
    TEST_MESSAGE_TEXT,
    TelegramBotIdentity,
    TelegramDelivery,
    TelegramError,
    TelegramNotifier,
)
from email_guardian.summarizer import Classification

FAKE_TOKEN = "111111:FAKE_TOKEN_DO_NOT_PRINT_987654321"
FAKE_CHAT_ID = "424242"


def make_settings(tmp: Path, *, token: str = FAKE_TOKEN, chat_id: str = FAKE_CHAT_ID) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
        "TELEGRAM_BOT_TOKEN": token,
        "TELEGRAM_CHAT_ID": chat_id,
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


class FakeResponse:
    def __init__(self, status_code: int, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def make_message() -> EmailMessage:
    return EmailMessage(
        message_id="m1",
        thread_id="t1",
        account="personal",
        sender_name="Alice",
        sender_email="alice@example.com",
        subject="Interview invitation",
        snippet="PRIVATE SNIPPET MUST NOT BE SENT",
        received_at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
    )


def make_analysis() -> Classification:
    return Classification(
        category="IMPORTANT",
        score=8,
        summary="An interview is scheduled.",
        reason="The email invites the recipient to an interview.",
        required_action="Confirm the interview slot.",
        deadline="2026-11-25",
        confidence=0.9,
        model="llama3.2",
    )


class TelegramNotifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_check_availability_uses_configured_token_without_logging_it(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200, {"ok": True, "result": {"is_bot": True, "username": "guardian_bot"}}
        )
        output = io.StringIO()

        with patch("email_guardian.notifier.requests.get", return_value=response) as get:
            with redirect_stdout(output):
                self.assertTrue(notifier.check_availability())

        self.assertEqual(
            get.call_args.args[0], f"{API_BASE_URL}/bot{FAKE_TOKEN}/getMe"
        )
        self.assertNotIn(FAKE_TOKEN, output.getvalue())

    def test_connection_error_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        error = requests.ConnectionError(
            f"connection to /bot{FAKE_TOKEN}/getMe failed"
        )

        with patch("email_guardian.notifier.requests.get", side_effect=error):
            with self.assertLogs("email_guardian.notifier", level="WARNING") as logs:
                self.assertFalse(notifier.check_availability())

        logged = "\n".join(logs.output)
        self.assertNotIn(FAKE_TOKEN, logged)
        self.assertIn("<redacted>", logged)

    def test_bot_identity_verifies_getme_without_exposing_the_token(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {"ok": True, "result": {"is_bot": True, "username": "guardian_bot"}},
        )
        output = io.StringIO()

        with patch("email_guardian.notifier.requests.get", return_value=response) as get:
            with redirect_stdout(output):
                identity = notifier.bot_identity()

        self.assertEqual(identity, TelegramBotIdentity(username="guardian_bot"))
        self.assertEqual(
            get.call_args.args[0], f"{API_BASE_URL}/bot{FAKE_TOKEN}/getMe"
        )
        self.assertNotIn(FAKE_TOKEN, output.getvalue())
        self.assertNotIn(FAKE_TOKEN, repr(identity))

    def test_bot_identity_rejects_response_without_a_bot(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(200, {"ok": True, "result": {"is_bot": False}})

        with patch("email_guardian.notifier.requests.get", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.bot_identity()

        self.assertIn("did not return a bot identity", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_send_test_message_posts_exact_payload_and_confirms_delivery(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {
                "ok": True,
                "result": {
                    "message_id": 42,
                    "chat": {"id": int(FAKE_CHAT_ID), "type": "private"},
                    "text": TEST_MESSAGE_TEXT,
                },
            },
        )
        output = io.StringIO()

        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            with redirect_stdout(output):
                delivery = notifier.send_test_message()

        self.assertEqual(
            post.call_args.args[0],
            f"{API_BASE_URL}/bot{FAKE_TOKEN}/sendMessage",
        )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], FAKE_CHAT_ID)
        self.assertEqual(payload["text"], TEST_MESSAGE_TEXT)
        self.assertNotIn("parse_mode", payload)
        self.assertTrue(payload["disable_web_page_preview"])
        self.assertEqual(payload["text"].splitlines()[0], "AI Email Guardian")
        self.assertEqual(delivery, TelegramDelivery(message_id=42, chat_id=FAKE_CHAT_ID))
        self.assertNotIn(FAKE_TOKEN, output.getvalue())
        self.assertNotIn(FAKE_TOKEN, repr(delivery))

    def test_send_test_message_without_chat_id_skips_network(self) -> None:
        notifier = TelegramNotifier(make_settings(Path(self._tmp.name), chat_id=""))

        with patch("email_guardian.notifier.requests.post") as post:
            with self.assertRaises(TelegramError) as ctx:
                notifier.send_test_message()

        self.assertIn("TELEGRAM_CHAT_ID is not configured", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        post.assert_not_called()

    def test_send_test_message_rejects_response_without_message_id(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {"ok": True, "result": {"chat": {"id": int(FAKE_CHAT_ID)}}},
        )

        with patch("email_guardian.notifier.requests.post", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.send_test_message()

        self.assertIn("did not return a message id", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_send_test_message_rejects_mismatched_chat(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {"ok": True, "result": {"message_id": 9, "chat": {"id": 999999}}},
        )

        with patch("email_guardian.notifier.requests.post", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.send_test_message()

        self.assertIn("different chat", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_send_test_message_timeout_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        error = requests.Timeout(f"POST /bot{FAKE_TOKEN}/sendMessage timed out")

        with patch("email_guardian.notifier.requests.post", side_effect=error):
            with self.assertRaises(TelegramError) as ctx:
                notifier.send_test_message()

        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))

    def test_send_test_message_api_failure_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            400,
            {"ok": False, "description": f"Bad chat id for bot {FAKE_TOKEN}"},
        )

        with patch("email_guardian.notifier.requests.post", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.send_test_message()

        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))

    def test_send_alert_posts_only_metadata(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(200, {"ok": True, "result": {"message_id": 7}})

        with patch("email_guardian.notifier.requests.post", return_value=response) as post:
            self.assertTrue(notifier.send_alert(make_message(), make_analysis()))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], FAKE_CHAT_ID)
        for expected in (
            "IMPORTANT",
            "Interview invitation",
            "An interview is scheduled.",
            "Confirm the interview slot.",
            "2026-11-25",
        ):
            self.assertIn(expected, payload["text"])
        self.assertNotIn("PRIVATE SNIPPET", payload["text"])
        self.assertTrue(payload["disable_web_page_preview"])

    def test_send_alert_without_chat_id_skips_network(self) -> None:
        notifier = TelegramNotifier(make_settings(Path(self._tmp.name), chat_id=""))

        with patch("email_guardian.notifier.requests.post") as post:
            self.assertFalse(notifier.send_alert(make_message(), make_analysis()))

        post.assert_not_called()

    def test_send_alert_failure_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        error = requests.ConnectionError(
            f"post to /bot{FAKE_TOKEN}/sendMessage failed"
        )

        with patch("email_guardian.notifier.requests.post", side_effect=error):
            with self.assertLogs("email_guardian.notifier", level="WARNING") as logs:
                self.assertFalse(notifier.send_alert(make_message(), make_analysis()))

        self.assertNotIn(FAKE_TOKEN, "\n".join(logs.output))

    def test_recent_chats_parses_and_deduplicates_updates(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {
                "ok": True,
                "result": [
                    {"update_id": 1, "message": {"chat": {"id": 111, "type": "private", "first_name": "Alice"}}},
                    {"update_id": 2, "channel_post": {"chat": {"id": -222, "type": "channel", "title": "Alerts"}}},
                    {"update_id": 3, "message": {"chat": {"id": 111, "type": "private", "first_name": "Alice"}}},
                    {"update_id": 4},
                ],
            },
        )

        with patch("email_guardian.notifier.requests.get", return_value=response) as get:
            chats = notifier.recent_chats(limit=5)

        self.assertEqual([chat.chat_id for chat in chats], ["111", "-222"])
        self.assertEqual([chat.title for chat in chats], ["Alice", "Alerts"])
        self.assertEqual(get.call_args.kwargs["params"], {"offset": -5, "limit": 5})
        self.assertNotIn(FAKE_TOKEN, repr(chats))

    def test_recent_chats_without_token_raises_generic_error(self) -> None:
        notifier = TelegramNotifier(make_settings(Path(self._tmp.name), token=""))

        with self.assertRaises(TelegramError) as ctx:
            notifier.recent_chats()

        self.assertIn("TELEGRAM_BOT_TOKEN is not configured", str(ctx.exception))

    def test_http_error_description_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            401,
            {"ok": False, "description": f"Unauthorized for token {FAKE_TOKEN}"},
        )

        with patch("email_guardian.notifier.requests.get", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.recent_chats()

        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))

    def test_api_rejection_is_redacted(self) -> None:
        notifier = TelegramNotifier(self.settings)
        response = FakeResponse(
            200,
            {"ok": False, "description": f"Bad token {FAKE_TOKEN}"},
        )

        with patch("email_guardian.notifier.requests.get", return_value=response):
            with self.assertRaises(TelegramError) as ctx:
                notifier.recent_chats()

        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
