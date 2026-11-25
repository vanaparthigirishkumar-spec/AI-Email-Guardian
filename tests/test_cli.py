"""Smoke tests for the CLI and module imports."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from email_guardian.cli import main
from email_guardian.gmail_client import (
    READONLY_SCOPE,
    GmailAuthError,
    GmailProfile,
    TokenStatus,
)
from email_guardian.logging_setup import shutdown_logging
from email_guardian.storage import EmailStore, StorageError


def _file_handlers_for(log_file: Path) -> list[logging.Handler]:
    expected = os.path.abspath(str(log_file))
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "baseFilename", None) == expected
    ]


class CliSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(shutdown_logging)

    def test_version_flag_exits_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_doctor_reports_core_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "DATABASE_PATH": str(Path(tmp) / "data" / "emails.db"),
                "LOG_DIR": str(Path(tmp) / "logs"),
            }
            buffer = io.StringIO()
            with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(buffer):
                code = main(["doctor"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Python version", output)
        self.assertIn("Database schema", output)
        self.assertIn("Telegram bot token", output)

    def test_run_is_deferred_to_later_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "DATABASE_PATH": str(Path(tmp) / "data" / "emails.db"),
                "LOG_DIR": str(Path(tmp) / "logs"),
            }
            with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(io.StringIO()):
                code = main(["run"])

        self.assertEqual(code, 2)

    def test_run_releases_log_file_after_command(self) -> None:
        """The log handle must be closed before the temp dir is cleaned up.

        On Windows an open handle makes TemporaryDirectory cleanup fail with
        PermissionError [WinError 32]; on every platform the file must be
        detachable and deletable afterwards.
        """

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            log_file = log_dir / "email_guardian.log"
            env = {
                "DATABASE_PATH": str(Path(tmp) / "data" / "emails.db"),
                "LOG_DIR": str(log_dir),
            }

            with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run"]), 2)

            self.assertTrue(log_file.is_file())
            self.assertEqual(_file_handlers_for(log_file), [])

            with log_file.open("a", encoding="utf-8") as stream:
                stream.write("")
            os.remove(log_file)
            self.assertFalse(log_file.exists())


class DoctorGmailTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._cleanup)
        self.env = {
            "GMAIL_ACCOUNTS": "personal",
            "GMAIL_CREDENTIALS_FILE": str(self.tmp / "credentials.json"),
            "GMAIL_TOKEN_DIR": str(self.tmp / "tokens"),
            "DATABASE_PATH": str(self.tmp / "data" / "emails.db"),
            "LOG_DIR": str(self.tmp / "logs"),
        }
        (self.tmp / "credentials.json").write_text("{}", encoding="utf-8")

    def _cleanup(self) -> None:
        shutdown_logging()
        self._tmp.cleanup()

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        with patch.dict("os.environ", self.env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(argv)
        return code, output.getvalue()

    def test_offline_doctor_marks_missing_token_but_still_passes(self) -> None:
        code, output = self.run_cli(["doctor"])

        self.assertEqual(code, 0)
        self.assertIn("Gmail:", output)
        self.assertIn("python main.py gmail auth personal", output)
        self.assertIn("Token directory writable", output)
        self.assertIn("[WARN] Telegram bot token", output)
        self.assertNotIn("[FAIL] Database schema", output)
        self.assertIn("Account status: personal: not authenticated", output)
        self.assertNotIn("Authenticate each account", output)

    def test_doctor_closes_database_connection(self) -> None:
        closed: list[int] = []
        original_close = EmailStore.close

        def tracked_close(store: EmailStore) -> None:
            closed.append(id(store))
            original_close(store)

        with patch.object(EmailStore, "close", tracked_close):
            code, output = self.run_cli(["doctor"])

        self.assertEqual(code, 0)
        self.assertEqual(len(closed), 1)
        self.assertIn("Database schema", output)

    def test_doctor_closes_database_after_initialization_error(self) -> None:
        closed: list[int] = []
        original_close = EmailStore.close

        def tracked_close(store: EmailStore) -> None:
            closed.append(id(store))
            original_close(store)

        with patch.object(
            EmailStore, "initialize", side_effect=StorageError("database failed")
        ), patch.object(EmailStore, "close", tracked_close):
            code, output = self.run_cli(["doctor"])

        self.assertEqual(code, 1)
        self.assertEqual(len(closed), 1)
        self.assertIn("[FAIL] Database schema", output)
        self.assertIn("database failed", output)

    def test_online_doctor_reports_live_profile_failure(self) -> None:
        instance = MagicMock()
        instance.token_status.return_value = TokenStatus(
            True, "owner@example.com", (READONLY_SCOPE,), False, True, None
        )
        instance.get_profile.side_effect = GmailAuthError(
            "Google rejected the stored token.", account="personal"
        )

        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["doctor", "--online"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Live profile", output)

    def test_online_doctor_passes_with_live_profile(self) -> None:
        instance = MagicMock()
        instance.token_status.return_value = TokenStatus(
            True, "owner@example.com", (READONLY_SCOPE,), False, True, None
        )
        instance.get_profile.return_value = GmailProfile(
            account="personal",
            email="owner@example.com",
            messages_total=5,
            threads_total=3,
        )

        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["doctor", "--online"])

        self.assertEqual(code, 0)
        self.assertIn("[OK  ] Live profile", output)
        self.assertIn("owner@example.com (5 messages, 3 threads)", output)
        self.assertIn(
            "Account status: personal: authenticated as owner@example.com", output
        )


class TelegramCliTest(unittest.TestCase):
    TOKEN = "222222:CLI_FAKE_TOKEN_DO_NOT_PRINT_987654321"
    CHAT_ID = "555001"
    TEST_TEXT = (
        "AI Email Guardian\n"
        "Telegram notification test successful!\n"
        "Your bot is connected and can deliver notifications to your Telegram account."
    )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(shutdown_logging)
        self.addCleanup(self._tmp.cleanup)
        self.env = {
            "GMAIL_ACCOUNTS": "personal",
            "GMAIL_CREDENTIALS_FILE": str(Path(self._tmp.name) / "credentials.json"),
            "GMAIL_TOKEN_DIR": str(Path(self._tmp.name) / "tokens"),
            "DATABASE_PATH": str(Path(self._tmp.name) / "data" / "emails.db"),
            "LOG_DIR": str(Path(self._tmp.name) / "logs"),
            "TELEGRAM_BOT_TOKEN": self.TOKEN,
            "TELEGRAM_CHAT_ID": self.CHAT_ID,
        }

    def identity_response(self) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {"is_bot": True, "username": "guardian_bot"},
        }
        return response

    def delivered_response(self, message_id: int = 42) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": {
                "message_id": message_id,
                "chat": {"id": int(self.CHAT_ID), "type": "private"},
            },
        }
        return response

    def run_cli(
        self,
        argv: list[str],
        *,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> tuple[int, str]:
        env = dict(self.env)
        if token is not None:
            env["TELEGRAM_BOT_TOKEN"] = token
        if chat_id is not None:
            env["TELEGRAM_CHAT_ID"] = chat_id
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict("os.environ", env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue() + stderr.getvalue()

    def test_chats_prints_chat_ids_without_the_token(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "chat": {
                            "id": 555001,
                            "type": "private",
                            "first_name": "Alice",
                        }
                    },
                }
            ],
        }

        with patch("email_guardian.notifier.requests.get", return_value=response) as get:
            code, output = self.run_cli(["telegram", "chats", "--limit", "3"])

        self.assertEqual(code, 0)
        self.assertIn("555001", output)
        self.assertIn("Alice", output)
        self.assertNotIn(self.TOKEN, output)
        self.assertEqual(get.call_args.kwargs["params"], {"offset": -3, "limit": 3})

    def test_test_command_delivers_a_real_message_after_getme(self) -> None:
        with patch(
            "email_guardian.notifier.requests.get", return_value=self.identity_response()
        ) as get, patch(
            "email_guardian.notifier.requests.post",
            return_value=self.delivered_response(),
        ) as post:
            code, output = self.run_cli(["telegram", "test"])

        self.assertEqual(code, 0)
        self.assertIn("Telegram bot is reachable", output)
        self.assertIn("@guardian_bot", output)
        self.assertIn("Test message delivered and confirmed by Telegram", output)
        self.assertIn(f"Chat id: {self.CHAT_ID} | Message id: 42", output)
        self.assertNotIn(self.TOKEN, output)
        self.assertIn("/getMe", get.call_args.args[0])
        self.assertIn("/sendMessage", post.call_args.args[0])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_id"], self.CHAT_ID)
        self.assertEqual(payload["text"], self.TEST_TEXT)
        self.assertNotIn("parse_mode", payload)
        self.assertTrue(payload["disable_web_page_preview"])

    def test_test_command_stops_with_clear_error_when_chat_id_is_missing(self) -> None:
        with patch(
            "email_guardian.notifier.requests.get", return_value=self.identity_response()
        ), patch("email_guardian.notifier.requests.post") as post:
            code, output = self.run_cli(["telegram", "test"], chat_id="")

        self.assertEqual(code, 1)
        self.assertIn("TELEGRAM_CHAT_ID is not configured", output)
        self.assertNotIn("Test message delivered", output)
        self.assertNotIn(self.TOKEN, output)
        post.assert_not_called()

    def test_test_command_reports_no_success_when_send_is_rejected(self) -> None:
        rejected = MagicMock()
        rejected.status_code = 400
        rejected.json.return_value = {
            "ok": False,
            "description": f"Bad chat id for bot {self.TOKEN}",
        }

        with patch(
            "email_guardian.notifier.requests.get", return_value=self.identity_response()
        ), patch("email_guardian.notifier.requests.post", return_value=rejected):
            code, output = self.run_cli(["telegram", "test"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Telegram test message was not delivered:", output)
        self.assertNotIn("Test message delivered and confirmed", output)
        self.assertNotIn(self.TOKEN, output)
        self.assertIn("<redacted>", output)

    def test_test_command_handles_send_timeout_safely(self) -> None:
        error = requests.Timeout(f"POST /bot{self.TOKEN}/sendMessage timed out")

        with patch(
            "email_guardian.notifier.requests.get", return_value=self.identity_response()
        ), patch("email_guardian.notifier.requests.post", side_effect=error):
            code, output = self.run_cli(["telegram", "test"])

        self.assertEqual(code, 1)
        self.assertNotIn(self.TOKEN, output)
        self.assertIn("<redacted>", output)
        self.assertNotIn("Test message delivered and confirmed", output)

    def test_test_command_does_not_report_delivery_for_unconfirmed_response(self) -> None:
        unconfirmed = MagicMock()
        unconfirmed.status_code = 200
        unconfirmed.json.return_value = {"ok": True, "result": {}}

        with patch(
            "email_guardian.notifier.requests.get", return_value=self.identity_response()
        ), patch("email_guardian.notifier.requests.post", return_value=unconfirmed):
            code, output = self.run_cli(["telegram", "test"])

        self.assertEqual(code, 1)
        self.assertIn("did not return a message id", output)
        self.assertNotIn("Test message delivered and confirmed", output)
        self.assertNotIn(self.TOKEN, output)

    def test_test_command_reports_bot_verification_failure(self) -> None:
        error = requests.ConnectionError(f"get /bot{self.TOKEN}/getMe failed")

        with patch("email_guardian.notifier.requests.get", side_effect=error), patch(
            "email_guardian.notifier.requests.post"
        ) as post:
            code, output = self.run_cli(["telegram", "test"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Telegram bot verification failed:", output)
        self.assertNotIn(self.TOKEN, output)
        self.assertIn("<redacted>", output)
        post.assert_not_called()

    def test_chats_error_is_redacted(self) -> None:
        error = requests.ConnectionError(
            f"get /bot{self.TOKEN}/getUpdates failed"
        )

        with patch("email_guardian.notifier.requests.get", side_effect=error):
            code, output = self.run_cli(["telegram", "chats"])

        self.assertEqual(code, 1)
        self.assertNotIn(self.TOKEN, output)
        self.assertIn("<redacted>", output)

    def test_missing_token_is_reported_without_a_secret(self) -> None:
        code, output = self.run_cli(["telegram", "chats"], token="")

        self.assertEqual(code, 1)
        self.assertIn("TELEGRAM_BOT_TOKEN is not configured", output)
        self.assertNotIn(self.TOKEN, output)


if __name__ == "__main__":
    unittest.main()
