"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from email_guardian.config import (
    PROJECT_ROOT,
    ConfigError,
    load_settings,
)


class SettingsDefaultsTest(unittest.TestCase):
    def test_defaults_applied_when_environment_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_env = Path(tmp) / "absent.env"
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(env_file=missing_env)

        self.assertEqual(settings.poll_interval_seconds, 60)
        self.assertEqual(settings.ollama_model, "llama3.2")
        self.assertEqual(settings.importance_threshold, 7)
        self.assertFalse(settings.is_configured("telegram_bot_token"))

    def test_relative_paths_resolve_against_project_root(self) -> None:
        env = {"DATABASE_PATH": "data/test.db", "LOG_DIR": "logs"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, env, clear=True):
                settings = load_settings(env_file=Path(tmp) / "absent.env")

        self.assertEqual(settings.database_path, PROJECT_ROOT / "data" / "test.db")
        self.assertEqual(settings.data_dir, PROJECT_ROOT / "data")

    def test_secrets_are_masked(self) -> None:
        env = {"TELEGRAM_BOT_TOKEN": "123456789:abcdefghijklmnop"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, env, clear=True):
                settings = load_settings(env_file=Path(tmp) / "absent.env")

        masked = settings.masked("telegram_bot_token")
        self.assertNotIn("abcdefghijklmnop", masked)
        self.assertEqual(settings.masked("telegram_chat_id"), "<not set>")


class SettingsValidationTest(unittest.TestCase):
    def test_invalid_integer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"POLL_INTERVAL_SECONDS": "soon"}, clear=True):
                with self.assertRaises(ConfigError):
                    load_settings(env_file=Path(tmp) / "absent.env")

    def test_interval_below_minimum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"POLL_INTERVAL_SECONDS": "3"}, clear=True):
                with self.assertRaises(ConfigError):
                    load_settings(env_file=Path(tmp) / "absent.env")

    def test_threshold_out_of_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"IMPORTANCE_THRESHOLD": "42"}, clear=True):
                with self.assertRaises(ConfigError):
                    load_settings(env_file=Path(tmp) / "absent.env")


if __name__ == "__main__":
    unittest.main()


class GmailAccountSettingsTest(unittest.TestCase):
    def _load(self, **env: str):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, env, clear=True):
                return load_settings(env_file=Path(tmp) / "absent.env")

    def test_no_accounts_by_default(self) -> None:
        self.assertEqual(self._load().gmail_accounts, ())

    def test_accounts_are_parsed_and_normalized(self) -> None:
        settings = self._load(GMAIL_ACCOUNTS=" Personal , work ")
        self.assertEqual(settings.gmail_accounts, ("personal", "work"))

    def test_invalid_alias_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(GMAIL_ACCOUNTS="bad alias")

    def test_duplicate_alias_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(GMAIL_ACCOUNTS="work,work")

    def test_token_paths_live_under_token_dir(self) -> None:
        settings = self._load(GMAIL_ACCOUNTS="personal")
        self.assertEqual(settings.gmail_token_dir, PROJECT_ROOT / "data" / "tokens")
        self.assertEqual(
            settings.gmail_token_path("personal"),
            PROJECT_ROOT / "data" / "tokens" / "personal.json",
        )
        self.assertEqual(
            settings.gmail_metadata_path("personal"),
            PROJECT_ROOT / "data" / "tokens" / "personal.meta.json",
        )

    def test_per_account_query_and_email_overrides(self) -> None:
        settings = self._load(
            GMAIL_ACCOUNTS="personal,work",
            GMAIL_QUERY="in:inbox newer_than:1d",
            GMAIL_QUERY_WORK="in:inbox newer_than:3d",
            GMAIL_ACCOUNT_WORK_EMAIL="Work@Example.com",
        )
        self.assertEqual(settings.gmail_query_for("personal"), "in:inbox newer_than:1d")
        self.assertEqual(settings.gmail_query_for("work"), "in:inbox newer_than:3d")
        self.assertEqual(settings.expected_email_for("work"), "Work@Example.com")
        self.assertIsNone(settings.expected_email_for("personal"))

    def test_default_query_does_not_depend_on_read_state(self) -> None:
        settings = self._load()
        self.assertNotIn("is:unread", settings.gmail_query)
        self.assertIn("in:inbox", settings.gmail_query)

    def test_dashed_alias_maps_to_environment_key(self) -> None:
        settings = self._load(
            GMAIL_ACCOUNTS="work-main",
            GMAIL_QUERY_WORK_MAIN="label:work",
            GMAIL_ACCOUNT_WORK_MAIN_EMAIL="work@example.com",
        )
        self.assertEqual(settings.gmail_query_for("work-main"), "label:work")
        self.assertEqual(settings.expected_email_for("work-main"), "work@example.com")

    def test_fetch_limit_must_be_positive(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(GMAIL_FETCH_LIMIT="0")

    def test_fetch_limit_upper_bound_is_enforced(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(GMAIL_FETCH_LIMIT="501")
