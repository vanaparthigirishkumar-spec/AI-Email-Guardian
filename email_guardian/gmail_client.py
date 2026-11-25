"""Gmail API integration: OAuth 2.0, profiles and inbox metadata.

Only the read-only scope is requested. This module never sends, deletes,
modifies or marks messages as read.

Every configured account has its own token file under ``data/tokens`` while
sharing a single OAuth client (``credentials.json``).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from email_guardian.config import Settings

LOGGER = logging.getLogger(__name__)

READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES: tuple[str, ...] = (READONLY_SCOPE,)
DISALLOWED_SCOPES = frozenset(
    {
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.insert",
        "https://www.googleapis.com/auth/gmail.labels",
        "https://www.googleapis.com/auth/gmail.settings.basic",
        "https://www.googleapis.com/auth/gmail.settings.sharing",
    }
)
METADATA_HEADERS = ("From", "To", "Subject", "Date", "Message-ID")
PAGE_SIZE = 100
MAX_BODY_CHARS = 20000
MAX_PAGES = 10
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_REASONS = frozenset(
    {"rateLimitExceeded", "userRateLimitExceeded", "backendError", "internalError"}
)
AUTH_REASONS = frozenset({"insufficientPermissions", "authError", "invalidCredentials"})
ACCOUNT_REAUTH_HINT = "python main.py gmail auth {account}"


class GmailError(RuntimeError):
    """Base class for Gmail integration failures."""


class GmailAuthError(GmailError):
    """Raised when credentials are missing, invalid or too broad."""

    def __init__(self, message: str, *, account: str, action: str | None = None) -> None:
        self.account = account
        self.action = action or ACCOUNT_REAUTH_HINT.format(account=account)
        super().__init__(f"{message} Run: {self.action}")


class GmailApiError(GmailError):
    """Raised when the Gmail API returns an unusable response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class EmailMessage:
    """Metadata for one inbox message, without its body."""

    message_id: str
    thread_id: str
    account: str
    sender_name: str
    sender_email: str
    subject: str
    snippet: str
    received_at: datetime | None
    labels: tuple[str, ...] = ()
    rfc822_message_id: str = ""

    @property
    def sender(self) -> str:
        if self.sender_name and self.sender_email:
            return f"{self.sender_name} <{self.sender_email}>"
        return self.sender_name or self.sender_email or "unknown sender"


@dataclass(frozen=True)
class EmailBody:
    """Plain-text body extracted from one message.

    Attachments are never downloaded or decoded as content.
    """

    message_id: str
    text: str
    content_type: str
    truncated: bool


@dataclass(frozen=True)
class GmailProfile:
    """Result of the Gmail ``getProfile`` endpoint."""

    account: str
    email: str
    messages_total: int
    threads_total: int


@dataclass(frozen=True)
class TokenStatus:
    """Offline summary of a stored token, used by ``doctor``."""

    present: bool
    email: str | None
    scopes: tuple[str, ...]
    expired: bool
    has_refresh_token: bool
    error: str | None


@dataclass(frozen=True)
class AccountMetadata:
    """Local record of which address a token belongs to."""

    account: str
    email: str
    scopes: tuple[str, ...]
    authenticated_at: str


def _decode_mime_header(value: str) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except (LookupError, UnicodeDecodeError, ValueError):
        return value.strip()


def _parse_sender(value: str) -> tuple[str, str]:
    name, address = parseaddr(value or "")
    return name.strip(), address.strip()


def _parse_received_at(internal_date: Any, date_header: str) -> datetime | None:
    if internal_date is not None:
        try:
            milliseconds = int(str(internal_date))
            return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            LOGGER.debug("Unparseable internalDate; falling back to the Date header")

    if not date_header:
        return None
    try:
        parsed = parsedate_to_datetime(date_header)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class _HtmlTextExtractor(HTMLParser):
    BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "div",
            "dl",
            "dt",
            "dd",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )
    SKIP_TAGS = frozenset({"head", "script", "style", "title", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(self._parts)


def html_to_text(html: str) -> str:
    """Extract readable text from an HTML body without external packages."""

    parser = _HtmlTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, AssertionError):
        LOGGER.debug("Ignored a malformed HTML fragment")
    return parser.text()


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v\u00a0]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_parts(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield payload
    parts = payload.get("parts")
    if not isinstance(parts, list):
        return
    for part in parts:
        if isinstance(part, Mapping):
            yield from _iter_parts(part)


def _part_charset(part: Mapping[str, Any]) -> str | None:
    for header in part.get("headers") or ():
        if not isinstance(header, Mapping):
            continue
        if str(header.get("name") or "").lower() != "content-type":
            continue
        value = str(header.get("value") or "")
        match = re.search(r"charset\s*=\s*\"?([\w.:-]+)", value, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _decode_part(data: Any, part: Mapping[str, Any]) -> str:
    if not isinstance(data, str) or not data:
        return ""
    try:
        padding = "=" * (-len(data) % 4)
        raw = base64.urlsafe_b64decode(data + padding)
    except (binascii.Error, ValueError):
        LOGGER.debug("Skipped an undecodable MIME part")
        return ""
    charset = _part_charset(part) or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def extract_body(payload: Mapping[str, Any]) -> tuple[str, str, bool]:
    """Return ``(text, content_type, truncated)`` for a Gmail message payload.

    ``text/plain`` parts are preferred; HTML is converted only as a fallback.
    Parts with a filename or attachment id are skipped and never downloaded.
    """

    plain: list[str] = []
    html: list[str] = []

    for part in _iter_parts(payload):
        body = part.get("body")
        if not isinstance(body, Mapping):
            body = {}
        if part.get("filename") or body.get("attachmentId"):
            continue
        mime_type = str(part.get("mimeType") or "").lower()
        if mime_type not in ("text/plain", "text/html"):
            continue
        data = body.get("data")
        decoded = _decode_part(data, part)
        if not decoded.strip():
            continue
        if mime_type == "text/plain":
            plain.append(decoded)
        else:
            html.append(html_to_text(decoded))

    if plain:
        text, content_type = "\n\n".join(plain), "text/plain"
    elif html:
        text, content_type = "\n\n".join(html), "text/html"
    else:
        return "", "none", False

    text = _normalise_text(text)
    if not text:
        return "", "none", False
    truncated = len(text) > MAX_BODY_CHARS
    return text[:MAX_BODY_CHARS], content_type, truncated


def _as_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _http_status(error: HttpError) -> int | None:
    status = getattr(getattr(error, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _http_reason(error: HttpError) -> str:
    content = getattr(error, "content", b"")
    if isinstance(content, str):
        raw = content
    else:
        try:
            raw = content.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            return ""
    try:
        payload = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(payload, Mapping):
        return ""
    details = payload.get("error")
    if not isinstance(details, Mapping):
        return ""
    reasons = details.get("errors")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
        reason = reasons[0].get("reason")
        if reason:
            return str(reason)
    return str(details.get("status") or "")


class GmailClient:
    """Read-only Gmail access for one configured account."""

    def __init__(self, settings: Settings, account: str, *, service: Any | None = None) -> None:
        if account not in settings.gmail_accounts:
            configured = ", ".join(settings.gmail_accounts) or "none"
            raise GmailError(
                f"Gmail account {account!r} is not configured "
                f"(configured accounts: {configured})."
            )
        self._settings = settings
        self.account = account
        self._service = service

    @property
    def token_path(self) -> Path:
        return self._settings.gmail_token_path(self.account)

    @property
    def metadata_path(self) -> Path:
        return self._settings.gmail_metadata_path(self.account)

    def authenticate(self, *, force: bool = False) -> str:
        """Run the OAuth desktop flow (or reuse a valid token) and return the address.

        Tokens are written to disk immediately after the browser flow returns,
        before any Gmail API verification, so a profile/network failure cannot
        throw away credentials the user just granted.
        """

        self._prepare_token_directory()

        credentials_file = self._settings.gmail_credentials_file
        if not credentials_file.is_file():
            raise GmailError(
                f"OAuth client file not found: {credentials_file}. "
                "Download it from Google Cloud Console (Desktop app) and place it there."
            )

        if not force and self.token_path.is_file():
            try:
                credentials = self.load_credentials()
                email = self._require_email(self._profile_payload(credentials))
            except GmailAuthError as exc:
                LOGGER.warning(
                    "Stored token for account %s is unusable (%s); starting a new OAuth flow",
                    self.account,
                    exc,
                )
            else:
                self._verify_identity(email, check_metadata=True)
                self._write_metadata(email, tuple(credentials.scopes or ()))
                LOGGER.info("Reused existing credentials for account %s", self.account)
                return email

        credentials = self._run_oauth_flow(credentials_file)
        self._validate_scopes(credentials.scopes or ())
        self._save_credentials(credentials)
        LOGGER.info("Saved OAuth token for account %s", self.account)

        try:
            email = self._require_email(self._profile_payload(credentials))
        except GmailError as exc:
            raise GmailApiError(
                f"{exc} The OAuth token was saved, so the sign-in does not need to be "
                f"repeated; retry with 'python main.py gmail test {self.account}'.",
                status=getattr(exc, "status", None),
            ) from exc

        self._write_metadata(email, tuple(credentials.scopes or ()))
        self._verify_identity(email, check_metadata=False)
        LOGGER.info("Authenticated Gmail account %s", self.account)
        return email

    def _prepare_token_directory(self) -> None:
        """Create and probe the token directory before opening the browser."""

        directory = self._settings.gmail_token_dir
        probe = directory / ".write_probe"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise GmailError(
                f"Token directory is not writable: {directory}. "
                "Check permissions or set GMAIL_TOKEN_DIR in .env."
            ) from exc

    def _run_oauth_flow(self, credentials_file: Path) -> Credentials:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), list(SCOPES)
            )
        except (OSError, ValueError) as exc:
            raise GmailError(f"Could not read OAuth client file: {credentials_file}") from exc

        try:
            credentials = flow.run_local_server(
                port=0, access_type="offline", prompt="select_account consent"
            )
        except Exception as exc:
            raise GmailAuthError(
                f"The OAuth flow did not complete "
                f"({type(exc).__name__}: {exc}).",
                account=self.account,
            ) from exc

        if not isinstance(credentials, Credentials):
            raise GmailAuthError(
                "The OAuth flow returned no credentials.",
                account=self.account,
            )
        if not getattr(credentials, "token", None):
            raise GmailAuthError(
                "The OAuth flow returned no access token.",
                account=self.account,
            )
        if not credentials.scopes:
            LOGGER.warning(
                "OAuth flow did not report granted scopes for account %s; "
                "recording the requested read-only scope",
                self.account,
            )
            credentials = Credentials(
                token=credentials.token,
                refresh_token=credentials.refresh_token,
                id_token=getattr(credentials, "id_token", None),
                token_uri=credentials.token_uri,
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                scopes=list(SCOPES),
                expiry=credentials.expiry,
            )
        if not credentials.refresh_token:
            LOGGER.warning(
                "Google returned no refresh token for account %s; the sign-in "
                "will expire in about an hour and need to be repeated",
                self.account,
            )
        return credentials

    def load_credentials(self, *, refresh: bool = True) -> Credentials:
        """Load the stored token and refresh it when it has expired."""

        path = self.token_path
        if not path.is_file():
            raise GmailAuthError(
                f"No OAuth token stored for account {self.account!r}.",
                account=self.account,
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GmailAuthError(
                f"Token file for account {self.account!r} is unreadable.",
                account=self.account,
            ) from exc
        if not isinstance(data, Mapping):
            raise GmailAuthError(
                f"Token file for account {self.account!r} is malformed.",
                account=self.account,
            )

        self._validate_scopes(tuple(str(scope) for scope in (data.get("scopes") or ())))

        try:
            credentials = Credentials.from_authorized_user_info(dict(data))
        except (ValueError, KeyError, TypeError) as exc:
            raise GmailAuthError(
                f"Token file for account {self.account!r} is malformed.",
                account=self.account,
            ) from exc

        if credentials.expired:
            if not credentials.refresh_token:
                raise GmailAuthError(
                    f"Token for account {self.account!r} expired and has no refresh token.",
                    account=self.account,
                )
            if refresh:
                self._refresh(credentials)
                self._save_credentials(credentials)

        return credentials

    def authenticated_email(self) -> str:
        """Return the address bound to this account's stored OAuth token."""

        metadata = self._read_metadata()
        return metadata.email if metadata else ""

    def get_profile(self) -> GmailProfile:
        """Return the authenticated address and mailbox counters."""

        payload = self._execute(self._get_service().users().getProfile(userId="me"))
        email = self._require_email(payload)
        self._verify_identity(email, check_metadata=True)
        return GmailProfile(
            account=self.account,
            email=email,
            messages_total=_as_count(payload.get("messagesTotal")),
            threads_total=_as_count(payload.get("threadsTotal")),
        )

    def fetch_body(self, message_id: str) -> EmailBody:
        """Download and extract one message's text body (never attachments)."""

        service = self._get_service()
        request = service.users().messages().get(
            userId="me", id=message_id, format="full"
        )
        payload = self._execute(request)
        text, content_type, truncated = extract_body(payload.get("payload") or {})
        LOGGER.debug(
            "Loaded %s body for message %s (account %s, %d characters%s)",
            content_type,
            message_id,
            self.account,
            len(text),
            ", truncated" if truncated else "",
        )
        return EmailBody(
            message_id=str(payload.get("id") or message_id),
            text=text,
            content_type=content_type,
            truncated=truncated,
        )

    def fetch_recent(self, limit: int | None = None) -> list[EmailMessage]:
        """Fetch recent inbox messages using the configurable query.

        Read state is deliberately not part of the default query, so a
        message opened manually before the next poll is still observed.
        """

        effective_limit = self._settings.gmail_fetch_limit if limit is None else int(limit)
        if effective_limit <= 0:
            return []

        query = self._settings.gmail_query_for(self.account)
        service = self._get_service()
        messages: list[EmailMessage] = []
        seen: set[str] = set()
        page_token: str | None = None
        pages = 0

        while len(messages) < effective_limit and pages < MAX_PAGES:
            pages += 1
            request = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=min(PAGE_SIZE, effective_limit - len(messages)),
                pageToken=page_token,
            )
            response = self._execute(request)

            for item in response.get("messages") or ():
                message_id = item.get("id")
                if not message_id or message_id in seen:
                    continue
                seen.add(message_id)
                if len(messages) >= effective_limit:
                    break
                messages.append(self._fetch_metadata(service, str(message_id)))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        LOGGER.info("Fetched %d message(s) for account %s", len(messages), self.account)
        return messages

    def token_status(self) -> TokenStatus:
        """Describe the stored token without any network access."""

        path = self.token_path
        if not path.is_file():
            return TokenStatus(False, None, (), False, False, None)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("token payload is not an object")
            credentials = Credentials.from_authorized_user_info(dict(data))
        except (OSError, ValueError, KeyError, TypeError):
            return TokenStatus(True, None, (), False, False, "token file is unreadable")

        scopes = tuple(credentials.scopes or ())
        error: str | None = None
        if any(scope not in scopes for scope in SCOPES):
            error = "token is missing the gmail.readonly scope; re-authenticate"
        elif set(scopes) & DISALLOWED_SCOPES:
            error = "token grants broader access than gmail.readonly; re-authenticate"

        metadata = self._read_metadata()
        return TokenStatus(
            present=True,
            email=metadata.email if metadata else None,
            scopes=scopes,
            expired=bool(credentials.expired),
            has_refresh_token=bool(credentials.refresh_token),
            error=error,
        )

    def _refresh(self, credentials: Credentials) -> None:
        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            raise GmailAuthError(
                f"Google rejected the stored token for account {self.account!r} "
                "(it expired or was revoked).",
                account=self.account,
            ) from exc
        except (TransportError, OSError) as exc:
            raise GmailAuthError(
                f"Could not refresh the token for account {self.account!r}; check connectivity.",
                account=self.account,
            ) from exc
        LOGGER.info("Refreshed access token for account %s", self.account)

    def _validate_scopes(self, scopes: Iterable[str]) -> None:
        granted = set(scopes or ())
        if any(scope not in granted for scope in SCOPES):
            raise GmailAuthError(
                f"Stored credentials for account {self.account!r} are missing the "
                "gmail.readonly scope.",
                account=self.account,
            )
        broader = sorted(granted & DISALLOWED_SCOPES)
        if broader:
            raise GmailAuthError(
                f"Stored credentials for account {self.account!r} grant broader access "
                "than gmail.readonly.",
                account=self.account,
                action=(
                    f"delete {self.token_path} and run "
                    f"python main.py gmail auth {self.account}"
                ),
            )

    def _verify_identity(self, email: str, *, check_metadata: bool) -> None:
        expected = self._settings.expected_email_for(self.account)
        if expected and expected.lower() != email.lower():
            raise GmailAuthError(
                f"Signed in as {email}, but GMAIL_ACCOUNT_{self.account.upper().replace('-', '_')}"
                f"_EMAIL is {expected} for account {self.account!r}.",
                account=self.account,
            )

        if not check_metadata:
            return
        metadata = self._read_metadata()
        if metadata and metadata.email and metadata.email.lower() != email.lower():
            raise GmailAuthError(
                f"The stored token for account {self.account!r} belongs to {metadata.email}, "
                f"but Gmail reports {email}.",
                account=self.account,
            )

    @staticmethod
    def _require_email(payload: Mapping[str, Any]) -> str:
        email = str(payload.get("emailAddress") or "").strip()
        if not email:
            raise GmailApiError("Gmail profile response did not contain an email address.")
        return email

    def _profile_payload(self, credentials: Credentials) -> Mapping[str, Any]:
        service = self._build_service(credentials)
        return self._execute(service.users().getProfile(userId="me"))

    @staticmethod
    def _build_service(credentials: Credentials) -> Any:
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def _get_service(self) -> Any:
        if self._service is None:
            self._service = self._build_service(self.load_credentials())
        return self._service

    def _fetch_metadata(self, service: Any, message_id: str) -> EmailMessage:
        request = service.users().messages().get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=list(METADATA_HEADERS),
        )
        payload = self._execute(request)
        message = self._parse_message(message_id, payload)
        LOGGER.debug("Loaded metadata for message %s (account %s)", message_id, self.account)
        return message

    def _parse_message(self, message_id: str, payload: Mapping[str, Any]) -> EmailMessage:
        body = payload.get("payload") or {}
        headers: dict[str, str] = {}
        for header in body.get("headers") or ():
            name = str(header.get("name") or "").lower()
            if name:
                headers[name] = str(header.get("value") or "")

        sender_name, sender_email = _parse_sender(headers.get("from", ""))
        return EmailMessage(
            message_id=str(payload.get("id") or message_id),
            thread_id=str(payload.get("threadId") or ""),
            account=self.account,
            sender_name=sender_name,
            sender_email=sender_email,
            subject=_decode_mime_header(headers.get("subject", "")) or "(no subject)",
            snippet=str(payload.get("snippet") or ""),
            received_at=_parse_received_at(
                payload.get("internalDate"), headers.get("date", "")
            ),
            labels=tuple(str(label) for label in (payload.get("labelIds") or ())),
            rfc822_message_id=headers.get("message-id", "").strip(),
        )

    def _execute(self, request: Any) -> dict[str, Any]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return request.execute()
            except HttpError as exc:
                status = _http_status(exc)
                reason = _http_reason(exc)

                if status == 401 or (status == 403 and reason in AUTH_REASONS):
                    raise GmailAuthError(
                        f"Gmail rejected the access token for account {self.account!r}.",
                        account=self.account,
                    ) from exc

                retryable = status in RETRY_STATUSES or reason in RETRY_REASONS
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise GmailApiError(
                        f"Gmail API request failed for account {self.account!r} "
                        f"(HTTP {status or 'unknown'}, reason {reason or 'unknown'}).",
                        status=status,
                    ) from exc

                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                LOGGER.warning(
                    "Gmail API HTTP %s for account %s; retrying in %.0fs",
                    status,
                    self.account,
                    delay,
                )
                time.sleep(delay)

        raise AssertionError("retry loop exhausted without returning or raising")

    def _save_credentials(self, credentials: Credentials) -> None:
        serialized = credentials.to_json()
        if not serialized or not serialized.strip():
            raise GmailError("OAuth credentials could not be serialized.")
        self._write_private_file(self.token_path, serialized)
        LOGGER.debug("Stored OAuth token for account %s at %s", self.account, self.token_path)

    def _read_metadata(self) -> AccountMetadata | None:
        path = self.metadata_path
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("metadata payload is not an object")
            return AccountMetadata(
                account=self.account,
                email=str(data["email"]),
                scopes=tuple(str(scope) for scope in (data.get("scopes") or ())),
                authenticated_at=str(data.get("authenticated_at") or ""),
            )
        except (OSError, ValueError, KeyError, TypeError):
            LOGGER.warning("Ignoring unreadable metadata for account %s", self.account)
            return None

    def _write_metadata(self, email: str, scopes: Iterable[str]) -> None:
        payload = {
            "account": self.account,
            "email": email,
            "scopes": list(scopes),
            "authenticated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_private_file(self.metadata_path, json.dumps(payload, indent=2))

    @staticmethod
    def _write_private_file(path: Path, text: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(path.parent, 0o700)
            temporary.write_text(text, encoding="utf-8")
            if os.name != "nt":
                os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except OSError as exc:
            cleanup_error: OSError | None = None
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
            message = (
                f"Could not write {path.name} in {path.parent}: "
                f"{exc.strerror or exc}"
            )
            if cleanup_error is not None:
                message += f" (and could not remove the partial file {temporary.name})"
            raise GmailError(message) from exc
