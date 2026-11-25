"""Tests for the one-shot `ai test` command."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from email_guardian.cli import main
from email_guardian.gmail_client import (
    EmailBody,
    EmailMessage,
    GmailApiError,
    GmailAuthError,
    GmailProfile,
)
from email_guardian.logging_setup import shutdown_logging


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def classification_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "category": "URGENT",
        "importance_score": 9,
        "summary": "Production is down.",
        "reason": "The sender reports an active outage.",
        "required_action": "Reply to Alice.",
        "deadline": "2026-11-24",
        "confidence": 0.85,
    }
    payload.update(overrides)
    return json.dumps(payload)


def chat_response(content: str) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        payload={"message": {"role": "assistant", "content": content}},
    )


def tags_response(*names: str) -> FakeResponse:
    return FakeResponse(
        status_code=200, payload={"models": [{"name": name} for name in names]}
    )


def make_email(account: str = "personal", message_id: str = "m1") -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        thread_id="t1",
        account=account,
        sender_name="Alice",
        sender_email="alice@example.com",
        subject=f"Subject {message_id}",
        snippet="preview",
        received_at=datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
    )


def make_client(account: str, messages: list[EmailMessage]) -> MagicMock:
    client = MagicMock()
    client.get_profile.return_value = GmailProfile(
        account=account,
        email=f"{account}@example.com",
        messages_total=10,
        threads_total=5,
    )
    client.fetch_recent.return_value = messages
    client.fetch_body.side_effect = lambda message_id: EmailBody(
        message_id=message_id,
        text="BODY-TOP-SECRET",
        content_type="text/plain",
        truncated=False,
    )
    return client


class AiCliTestCase(unittest.TestCase):
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
            "OLLAMA_MODEL": "llama3.2",
            "OLLAMA_TIMEOUT_SECONDS": "5",
            "IMPORTANCE_THRESHOLD": "7",
        }
        self.log_file = self.tmp / "logs" / "email_guardian.log"

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

    def run_ai(
        self,
        account: str,
        messages: list[EmailMessage],
        *,
        limit: int = 3,
        post: object | None = None,
    ) -> tuple[int, str, MagicMock]:
        client = make_client(account, messages)
        with patch(
            "email_guardian.cli.GmailClient", return_value=client
        ), patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("llama3.2:latest"),
        ), patch(
            "email_guardian.summarizer.requests.post",
            return_value=post or chat_response(classification_json()),
        ) as post_mock:
            code, output = self.run_cli(
                ["ai", "test", account, "--limit", str(limit)]
            )
        return code, output, client

    def test_analyzes_a_message_and_prints_the_structure(self) -> None:
        code, output, client = self.run_ai("personal", [make_email()])

        self.assertEqual(code, 0)
        self.assertIn("[OK  ] personal -> personal@example.com", output)
        self.assertIn("[URGENT] 9/10 (confidence 85%)", output)
        self.assertIn("[meets threshold]", output)
        self.assertIn("Summary : Production is down.", output)
        self.assertIn("Reason  : The sender reports an active outage.", output)
        self.assertIn("Action  : Reply to Alice.", output)
        self.assertIn("Deadline: 2026-11-24", output)
        self.assertIn("Analyzed 1 message(s), 0 failed.", output)
        client.fetch_recent.assert_called_once_with(limit=3)
        client.fetch_body.assert_called_once_with("m1")

    def test_accounts_are_independent(self) -> None:
        work_messages = [make_email(account="work", message_id="w1")]
        client = make_client("work", work_messages)
        with patch(
            "email_guardian.cli.GmailClient", return_value=client
        ), patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("llama3.2:latest"),
        ), patch(
            "email_guardian.summarizer.requests.post",
            return_value=chat_response(classification_json(category="NORMAL")),
        ) as post:
            code, output = self.run_cli(["ai", "test", "work", "--limit", "3"])

        self.assertEqual(code, 0)
        self.assertIn("work@example.com", output)
        self.assertIn("[NORMAL]", output)
        user_prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Account: work", user_prompt)
        client.fetch_recent.assert_called_once_with(limit=3)

    def test_limit_is_forwarded_to_gmail(self) -> None:
        messages = [make_email(message_id="m1"), make_email(message_id="m2")]
        _, _, client = self.run_ai("personal", messages, limit=2)

        client.fetch_recent.assert_called_once_with(limit=2)

    def test_malformed_response_is_reported_and_remaining_mail_is_analyzed(self) -> None:
        messages = [make_email(message_id="m1"), make_email(message_id="m2")]
        responses = [
            chat_response("not json"),
            chat_response("still not json"),
            chat_response(classification_json(category="NORMAL")),
        ]
        client = make_client("personal", messages)
        with patch(
            "email_guardian.cli.GmailClient", return_value=client
        ), patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("llama3.2:latest"),
        ), patch(
            "email_guardian.summarizer.requests.post", side_effect=responses
        ):
            code, output = self.run_cli(["ai", "test", "personal", "--limit", "2"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL] m1", output)
        self.assertIn("invalid JSON", output)
        self.assertIn("[NORMAL]", output)
        self.assertIn("Analyzed 1 message(s), 1 failed.", output)

    def test_body_failure_is_counted_and_other_mail_is_analyzed(self) -> None:
        messages = [make_email(message_id="m1"), make_email(message_id="m2")]
        client = make_client("personal", messages)
        client.fetch_body.side_effect = [
            GmailApiError("body unavailable", status=500),
            EmailBody(
                message_id="m2",
                text="second body",
                content_type="text/plain",
                truncated=False,
            ),
        ]
        with patch(
            "email_guardian.cli.GmailClient", return_value=client
        ), patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("llama3.2:latest"),
        ), patch(
            "email_guardian.summarizer.requests.post",
            return_value=chat_response(classification_json(category="NORMAL")),
        ):
            code, output = self.run_cli(["ai", "test", "personal", "--limit", "2"])

        self.assertEqual(code, 1)
        self.assertIn("[FAIL] m1", output)
        self.assertIn("[NORMAL]", output)
        self.assertIn("Analyzed 1 message(s), 1 failed.", output)

    def test_ollama_unavailable_fails_before_touching_gmail(self) -> None:
        with patch(
            "email_guardian.cli.GmailClient"
        ) as gmail_factory, patch(
            "email_guardian.summarizer.requests.get",
            side_effect=requests.exceptions.ConnectionError(),
        ):
            code, output = self.run_cli(["ai", "test", "personal"])

        self.assertEqual(code, 1)
        self.assertIn("Ollama is not reachable", output)
        self.assertIn("ollama serve", output)
        gmail_factory.assert_not_called()

    def test_missing_model_fails_with_pull_hint(self) -> None:
        with patch(
            "email_guardian.cli.GmailClient"
        ) as gmail_factory, patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("mistral:latest"),
        ):
            code, output = self.run_cli(["ai", "test", "personal"])

        self.assertEqual(code, 1)
        self.assertIn("ollama pull llama3.2", output)
        gmail_factory.assert_not_called()

    def test_gmail_auth_failure_aborts_with_reauth_hint(self) -> None:
        client = make_client("personal", [make_email()])
        client.get_profile.side_effect = GmailAuthError(
            "Google rejected the stored token.", account="personal"
        )
        with patch(
            "email_guardian.cli.GmailClient", return_value=client
        ), patch(
            "email_guardian.summarizer.requests.get",
            return_value=tags_response("llama3.2:latest"),
        ):
            code, output = self.run_cli(["ai", "test", "personal"])

        self.assertEqual(code, 1)
        self.assertIn("gmail auth personal", output)

    def test_unconfigured_account_is_rejected(self) -> None:
        with patch(
            "email_guardian.summarizer.requests.get"
        ) as get_models:
            code, output = self.run_cli(["ai", "test", "other"])

        self.assertEqual(code, 2)
        self.assertIn("not configured", output)
        get_models.assert_not_called()

    def test_zero_limit_does_nothing(self) -> None:
        with patch(
            "email_guardian.summarizer.requests.get"
        ) as get_models:
            code, output = self.run_cli(["ai", "test", "personal", "--limit", "0"])

        self.assertEqual(code, 0)
        self.assertIn("Nothing to analyze", output)
        get_models.assert_not_called()

    def test_no_matching_mail_exits_cleanly(self) -> None:
        code, output, client = self.run_ai("personal", [])

        self.assertEqual(code, 0)
        self.assertIn("no messages matched", output)
        client.fetch_body.assert_not_called()

    def test_missing_action_and_deadline_print_as_none(self) -> None:
        payload = json.loads(classification_json(category="NORMAL"))
        del payload["required_action"]
        del payload["deadline"]
        code, output, _ = self.run_ai(
            "personal", [make_email()], post=chat_response(json.dumps(payload))
        )

        self.assertEqual(code, 0)
        self.assertIn("Action  : None", output)
        self.assertNotIn("Deadline:", output)

    def test_body_and_subject_are_not_written_to_logs(self) -> None:
        code, output, _ = self.run_ai("personal", [make_email()])

        self.assertEqual(code, 0)
        self.assertNotIn("BODY-TOP-SECRET", output)
        log_text = self.log_file.read_text(encoding="utf-8")
        self.assertNotIn("BODY-TOP-SECRET", log_text)
        self.assertNotIn("Subject m1", log_text)
        self.assertIn("Classified message m1", log_text)


if __name__ == "__main__":
    unittest.main()
