"""Tests for the Gmail CLI commands."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from email_guardian.cli import main
from email_guardian.config import load_settings
from email_guardian.gmail_client import (
    READONLY_SCOPE,
    SCOPES,
    EmailMessage,
    GmailApiError,
    GmailAuthError,
    GmailClient,
    GmailProfile,
    TokenStatus,
)
from email_guardian.logging_setup import shutdown_logging
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


class GmailCliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._cleanup)
        self.env = {
            "GMAIL_ACCOUNTS": "personal,work",
            "GMAIL_CREDENTIALS_FILE": str(self.tmp / "credentials.json"),
            "GMAIL_TOKEN_DIR": str(self.tmp / "tokens"),
            "DATABASE_PATH": str(self.tmp / "data" / "emails.db"),
            "LOG_DIR": str(self.tmp / "logs"),
        }

    def _cleanup(self) -> None:
        shutdown_logging()
        self._tmp.cleanup()

    def run_cli(self, argv: list[str]) -> tuple[int, str]:
        with patch.dict("os.environ", self.env, clear=True), patch(
            "email_guardian.config.load_dotenv"
        ):
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                code = main(argv)
        return code, output.getvalue()

    def write_token(self, account: str = "personal") -> None:
        tokens = Path(self.env["GMAIL_TOKEN_DIR"])
        tokens.mkdir(parents=True, exist_ok=True)
        (tokens / f"{account}.json").write_text(
            json.dumps(
                {
                    "token": "access-token",
                    "refresh_token": "refresh-token",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "client_id": "test-client-id",
                    "client_secret": "test-client-secret",
                    "scopes": [READONLY_SCOPE],
                }
            ),
            encoding="utf-8",
        )
        (tokens / f"{account}.meta.json").write_text(
            json.dumps(
                {
                    "account": account,
                    "email": "owner@example.com",
                    "scopes": [READONLY_SCOPE],
                    "authenticated_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    def settings(self):
        with patch.dict("os.environ", self.env, clear=True):
            return load_settings(env_file=self.tmp / "absent.env")


class AccountsCommandTest(GmailCliTestCase):
    def test_reports_when_no_accounts_are_configured(self) -> None:
        self.env["GMAIL_ACCOUNTS"] = ""

        code, output = self.run_cli(["gmail", "accounts"])

        self.assertEqual(code, 1)
        self.assertIn("No Gmail accounts configured", output)

    def test_lists_configured_accounts_and_token_state(self) -> None:
        self.write_token("personal")

        code, output = self.run_cli(["gmail", "accounts"])

        self.assertEqual(code, 0)
        self.assertIn("personal: authenticated as owner@example.com", output)
        self.assertIn("work: not authenticated", output)

    def test_describes_expired_token(self) -> None:
        settings = self.settings()
        tokens = settings.gmail_token_dir
        tokens.mkdir(parents=True, exist_ok=True)
        expired = datetime(2020, 1, 1, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (tokens / "personal.json").write_text(
            json.dumps(
                {
                    "token": "access-token",
                    "refresh_token": "refresh-token",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "client_id": "test-client-id",
                    "client_secret": "test-client-secret",
                    "scopes": [READONLY_SCOPE],
                    "expiry": expired,
                }
            ),
            encoding="utf-8",
        )
        (tokens / "personal.meta.json").write_text(
            json.dumps({"account": "personal", "email": "owner@example.com"}),
            encoding="utf-8",
        )

        code, output = self.run_cli(["gmail", "accounts"])

        self.assertEqual(code, 0)
        self.assertIn("access token expired (will refresh)", output)


class AuthCommandTest(GmailCliTestCase):
    def test_rejects_unconfigured_account(self) -> None:
        code, output = self.run_cli(["gmail", "auth", "missing"])

        self.assertEqual(code, 2)
        self.assertIn("is not configured", output)

    def test_authenticates_one_account(self) -> None:
        instance = MagicMock()
        instance.authenticate.return_value = "owner@example.com"
        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["gmail", "auth", "personal"])

        self.assertEqual(code, 0)
        self.assertIn("Authenticated personal as owner@example.com", output)
        instance.authenticate.assert_called_once_with(force=False)

    def test_force_flag_is_forwarded(self) -> None:
        instance = MagicMock()
        instance.authenticate.return_value = "owner@example.com"
        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, _ = self.run_cli(["gmail", "auth", "personal", "--force"])

        self.assertEqual(code, 0)
        instance.authenticate.assert_called_once_with(force=True)

    def test_reports_auth_failure(self) -> None:
        instance = MagicMock()
        instance.authenticate.side_effect = GmailAuthError(
            "Google rejected the token.", account="personal"
        )
        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["gmail", "auth", "personal"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL]", output)
        self.assertIn("gmail auth personal", output)


class AuthPersistenceTest(GmailCliTestCase):
    def test_auth_writes_the_account_token_file(self) -> None:
        Path(self.env["GMAIL_CREDENTIALS_FILE"]).write_text("{}", encoding="utf-8")
        credentials = Credentials(
            token="new-access-token",
            refresh_token="new-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=list(SCOPES),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        flow = MagicMock()
        flow.run_local_server.return_value = credentials

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            code, output = self.run_cli(["gmail", "auth", "personal"])

        token_path = Path(self.env["GMAIL_TOKEN_DIR"]) / "personal.json"
        self.assertEqual(code, 0)
        self.assertTrue(token_path.is_file())
        self.assertIn("Authenticated personal as owner@example.com", output)
        self.assertIn("Token saved to", output)
        stored = json.loads(token_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["scopes"], list(SCOPES))

    def test_auth_keeps_the_token_when_the_configured_email_does_not_match(
        self,
    ) -> None:
        self.env["GMAIL_ACCOUNT_PERSONAL_EMAIL"] = "placeholder@example.com"
        Path(self.env["GMAIL_CREDENTIALS_FILE"]).write_text("{}", encoding="utf-8")
        credentials = Credentials(
            token="new-access-token",
            refresh_token="new-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=list(SCOPES),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        flow = MagicMock()
        flow.run_local_server.return_value = credentials

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            code, output = self.run_cli(["gmail", "auth", "personal"])

        token_path = Path(self.env["GMAIL_TOKEN_DIR"]) / "personal.json"
        self.assertEqual(code, 1)
        self.assertTrue(token_path.is_file())
        self.assertIn("GMAIL_ACCOUNT_PERSONAL_EMAIL", output)
        metadata = json.loads(
            (Path(self.env["GMAIL_TOKEN_DIR"]) / "personal.meta.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["email"], "owner@example.com")


class TestCommandTest(GmailCliTestCase):
    def test_rejects_unconfigured_account(self) -> None:
        code, output = self.run_cli(["gmail", "test", "missing"])

        self.assertEqual(code, 2)
        self.assertIn("is not configured", output)

    def test_prints_profile_and_recent_messages(self) -> None:
        instance = MagicMock()
        instance.get_profile.return_value = GmailProfile(
            account="personal",
            email="owner@example.com",
            messages_total=12,
            threads_total=7,
        )
        instance.fetch_recent.return_value = [
            EmailMessage(
                message_id="m1",
                thread_id="t1",
                account="personal",
                sender_name="Jane Doe",
                sender_email="jane@example.com",
                subject="Quarterly report",
                snippet="hidden",
                received_at=datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc),
            )
        ]

        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["gmail", "test", "personal", "--limit", "1"])

        self.assertEqual(code, 0)
        self.assertIn("owner@example.com (12 messages, 7 threads)", output)
        self.assertIn("Jane Doe <jane@example.com>", output)
        self.assertIn("Quarterly report", output)
        self.assertIn("2026-11-20 09:30", output)
        self.assertNotIn("hidden", output)
        instance.fetch_recent.assert_called_once_with(limit=1)

    def test_reports_api_failure(self) -> None:
        instance = MagicMock()
        instance.get_profile.side_effect = GmailApiError("Gmail API failed", status=500)
        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["gmail", "test", "personal"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL]", output)
        self.assertIn("Gmail API failed", output)

    def test_negative_limit_is_rejected_by_the_parser(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["gmail", "test", "personal", "--limit", "-1"])

        self.assertEqual(ctx.exception.code, 2)


class TokenStatusCommandTest(GmailCliTestCase):
    def test_accounts_command_marks_unreadable_scopes(self) -> None:
        status = TokenStatus(
            present=True,
            email="owner@example.com",
            scopes=(),
            expired=False,
            has_refresh_token=True,
            error="token is missing the gmail.readonly scope; re-authenticate",
        )
        instance = MagicMock()
        instance.token_status.return_value = status

        with patch("email_guardian.cli.GmailClient", return_value=instance):
            code, output = self.run_cli(["gmail", "accounts"])

        self.assertEqual(code, 0)
        self.assertIn("token is missing the gmail.readonly scope", output)


if __name__ == "__main__":
    unittest.main()
