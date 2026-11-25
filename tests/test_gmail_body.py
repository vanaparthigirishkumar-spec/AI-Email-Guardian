"""Tests for MIME body extraction and attachment skipping."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import (
    MAX_BODY_CHARS,
    GmailClient,
    extract_body,
)


def make_settings(tmp: Path) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
    }
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def encode(text: str, charset: str = "utf-8") -> str:
    return base64.urlsafe_b64encode(text.encode(charset)).decode("ascii").rstrip("=")


def part(
    mime_type: str,
    text: str = "",
    *,
    charset: str = "utf-8",
    filename: str = "",
    attachment_id: str | None = None,
) -> dict:
    data = {"mimeType": mime_type, "filename": filename}
    if text:
        data["body"] = {"data": encode(text, charset)}
    else:
        data["body"] = {}
    if attachment_id:
        data["body"]["attachmentId"] = attachment_id
    data["headers"] = [
        {"name": "Content-Type", "value": f"{mime_type}; charset={charset}"}
    ]
    return data


class ExtractBodyTest(unittest.TestCase):
    def test_plain_text_body_is_returned(self) -> None:
        payload = {"mimeType": "text/plain", **part("text/plain", "Hello\r\nworld")}

        text, content_type, truncated = extract_body(payload)

        self.assertEqual(text, "Hello\nworld")
        self.assertEqual(content_type, "text/plain")
        self.assertFalse(truncated)

    def test_html_only_body_is_converted_to_text(self) -> None:
        html = (
            "<html><head><style>.x{color:red}</style><title>Ignored</title></head>"
            "<body><p>Hello&nbsp;<b>world</b></p><p>Second &amp; last</p>"
            "<script>alert('x')</script></body></html>"
        )
        payload = part("text/html", html)

        text, content_type, _ = extract_body(payload)

        self.assertEqual(content_type, "text/html")
        self.assertIn("Hello world", text)
        self.assertIn("Second & last", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("color:red", text)
        self.assertNotIn("Ignored", text)

    def test_multipart_prefers_plain_over_html_regardless_of_order(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                part("text/html", "<p>HTML BODY</p>"),
                part("text/plain", "PLAIN BODY"),
            ],
        }

        text, content_type, _ = extract_body(payload)

        self.assertEqual(text, "PLAIN BODY")
        self.assertEqual(content_type, "text/plain")

    def test_nested_multipart_skips_attachments(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        part("text/plain", "REAL BODY"),
                        part("text/html", "<p>REAL HTML</p>"),
                    ],
                },
                part("text/plain", "ATTACHMENT SECRET", filename="notes.txt"),
                part("application/pdf", "PDFSECRET", filename="invoice.pdf"),
                part(
                    "text/plain",
                    "ATTACHMENTID SECRET",
                    attachment_id="att-123",
                ),
            ],
        }

        text, content_type, _ = extract_body(payload)

        self.assertEqual(text, "REAL BODY")
        self.assertEqual(content_type, "text/plain")
        self.assertNotIn("SECRET", text)

    def test_charset_is_respected(self) -> None:
        payload = part("text/plain", "caf\u00e9 r\u00e9sum\u00e9", charset="iso-8859-1")

        text, _, _ = extract_body(payload)

        self.assertEqual(text, "caf\u00e9 r\u00e9sum\u00e9")

    def test_empty_or_missing_body_returns_none(self) -> None:
        self.assertEqual(extract_body({}), ("", "none", False))
        self.assertEqual(
            extract_body({"mimeType": "multipart/mixed", "parts": []}),
            ("", "none", False),
        )
        self.assertEqual(
            extract_body(part("text/plain", "   \n\t  ")), ("", "none", False)
        )

    def test_undecodable_part_is_skipped(self) -> None:
        payload = {
            "mimeType": "text/plain",
            "body": {"data": "@@@not-base64@@@"},
            "headers": [{"name": "Content-Type", "value": "text/plain; charset=utf-8"}],
        }

        self.assertEqual(extract_body(payload), ("", "none", False))

    def test_long_body_is_truncated(self) -> None:
        payload = part("text/plain", "A" * (MAX_BODY_CHARS + 250))

        text, _, truncated = extract_body(payload)

        self.assertTrue(truncated)
        self.assertEqual(len(text), MAX_BODY_CHARS)


class FetchBodyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = make_settings(Path(self._tmp.name))

    def test_fetch_body_uses_full_format_and_never_fetches_attachments(self) -> None:
        payload = {
            "id": "m1",
            "payload": {
                "mimeType": "multipart/mixed",
                "parts": [
                    part("text/plain", "Body text"),
                    part("application/pdf", "PDFSECRET", filename="a.pdf"),
                ],
            },
        }
        service = MagicMock()
        service.users.return_value.messages.return_value.get.return_value.execute.return_value = (
            payload
        )
        client = GmailClient(self.settings, "personal", service=service)

        body = client.fetch_body("m1")

        self.assertEqual(body.message_id, "m1")
        self.assertEqual(body.text, "Body text")
        self.assertEqual(body.content_type, "text/plain")
        service.users.return_value.messages.return_value.get.assert_called_once_with(
            userId="me", id="m1", format="full"
        )
        service.users.return_value.attachments.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
