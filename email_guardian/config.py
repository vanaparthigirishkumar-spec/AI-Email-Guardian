"""Configuration loading and validation.

Settings come from environment variables, optionally loaded from a `.env`
file at the project root. Relative paths are resolved against the project
root so the application behaves the same on Windows and Linux.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_POLL_INTERVAL_SECONDS = 10
MIN_IMPORTANCE_THRESHOLD = 0
MAX_IMPORTANCE_THRESHOLD = 10
MIN_GMAIL_FETCH_LIMIT = 1
MAX_GMAIL_FETCH_LIMIT = 500
MAX_GMAIL_ACCOUNTS = 10
ACCOUNT_ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class ConfigError(RuntimeError):
    """Raised when an environment value cannot be parsed or is out of range."""


def _raw(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _as_int(name: str, default: int) -> int:
    value = _raw(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc


def _as_float(name: str, default: float) -> float:
    value = _raw(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def _as_path(name: str, default: str) -> Path:
    value = _raw(name) or default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_suffix(account: str) -> str:
    return account.upper().replace("-", "_")


def _parse_gmail_accounts() -> tuple[str, ...]:
    raw = _raw("GMAIL_ACCOUNTS")
    if not raw:
        return ()

    accounts: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        account = part.strip().lower()
        if not account:
            continue
        if not ACCOUNT_ALIAS_PATTERN.match(account):
            raise ConfigError(
                f"Invalid Gmail account alias {account!r} in GMAIL_ACCOUNTS; use lowercase "
                "letters, digits, '-' or '_' (max 32 characters)."
            )
        if account in seen:
            raise ConfigError(f"Duplicate Gmail account alias {account!r} in GMAIL_ACCOUNTS.")
        seen.add(account)
        accounts.append(account)

    if len(accounts) > MAX_GMAIL_ACCOUNTS:
        raise ConfigError(f"GMAIL_ACCOUNTS supports at most {MAX_GMAIL_ACCOUNTS} accounts.")
    return tuple(accounts)


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings for the application."""

    gmail_credentials_file: Path
    gmail_token_dir: Path
    gmail_accounts: tuple[str, ...]
    gmail_account_queries: dict[str, str]
    gmail_account_emails: dict[str, str]
    gmail_query: str
    gmail_fetch_limit: int
    poll_interval_seconds: int
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    telegram_bot_token: str
    telegram_chat_id: str
    database_path: Path
    log_dir: Path
    log_level: str
    importance_threshold: int

    @property
    def data_dir(self) -> Path:
        return self.database_path.parent

    def is_configured(self, field_name: str) -> bool:
        return bool(getattr(self, field_name))

    def masked(self, field_name: str) -> str:
        value = getattr(self, field_name)
        if not value:
            return "<not set>"
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-2:]}"

    def gmail_token_path(self, account: str) -> Path:
        return self.gmail_token_dir / f"{account}.json"

    def gmail_metadata_path(self, account: str) -> Path:
        return self.gmail_token_dir / f"{account}.meta.json"

    def gmail_query_for(self, account: str) -> str:
        return self.gmail_account_queries.get(account, self.gmail_query)

    def expected_email_for(self, account: str) -> str | None:
        return self.gmail_account_emails.get(account)


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings, optionally reading a specific .env file.

    Existing environment variables take precedence over the .env file.
    """

    path = env_file if env_file is not None else PROJECT_ROOT / ".env"
    if path.is_file():
        load_dotenv(path, override=False)

    poll_interval = _as_int("POLL_INTERVAL_SECONDS", 60)
    if poll_interval < MIN_POLL_INTERVAL_SECONDS:
        raise ConfigError(
            f"POLL_INTERVAL_SECONDS must be >= {MIN_POLL_INTERVAL_SECONDS}, got {poll_interval}"
        )

    threshold = _as_int("IMPORTANCE_THRESHOLD", 7)
    if not MIN_IMPORTANCE_THRESHOLD <= threshold <= MAX_IMPORTANCE_THRESHOLD:
        raise ConfigError(
            f"IMPORTANCE_THRESHOLD must be between {MIN_IMPORTANCE_THRESHOLD} and "
            f"{MAX_IMPORTANCE_THRESHOLD}, got {threshold}"
        )

    fetch_limit = _as_int("GMAIL_FETCH_LIMIT", 20)
    if not MIN_GMAIL_FETCH_LIMIT <= fetch_limit <= MAX_GMAIL_FETCH_LIMIT:
        raise ConfigError(
            f"GMAIL_FETCH_LIMIT must be between {MIN_GMAIL_FETCH_LIMIT} and "
            f"{MAX_GMAIL_FETCH_LIMIT}, got {fetch_limit}"
        )

    accounts = _parse_gmail_accounts()
    account_queries: dict[str, str] = {}
    account_emails: dict[str, str] = {}
    for account in accounts:
        suffix = _env_suffix(account)
        query = _raw(f"GMAIL_QUERY_{suffix}")
        if query:
            account_queries[account] = query
        expected_email = _raw(f"GMAIL_ACCOUNT_{suffix}_EMAIL")
        if expected_email:
            account_emails[account] = expected_email

    return Settings(
        gmail_credentials_file=_as_path("GMAIL_CREDENTIALS_FILE", "credentials.json"),
        gmail_token_dir=_as_path("GMAIL_TOKEN_DIR", "data/tokens"),
        gmail_accounts=accounts,
        gmail_account_queries=account_queries,
        gmail_account_emails=account_emails,
        gmail_query=_raw("GMAIL_QUERY", "in:inbox newer_than:1d"),
        gmail_fetch_limit=fetch_limit,
        poll_interval_seconds=poll_interval,
        ollama_base_url=_raw("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_raw("OLLAMA_MODEL", "llama3.2"),
        ollama_timeout_seconds=_as_int("OLLAMA_TIMEOUT_SECONDS", 300),
        telegram_bot_token=_raw("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_raw("TELEGRAM_CHAT_ID"),
        database_path=_as_path("DATABASE_PATH", "data/emails.db"),
        log_dir=_as_path("LOG_DIR", "logs"),
        log_level=_raw("LOG_LEVEL", "INFO").upper(),
        importance_threshold=threshold,
    )


def ensure_runtime_dirs(settings: Settings) -> None:
    """Create the data, token and log directories if they do not exist yet."""

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.gmail_token_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
