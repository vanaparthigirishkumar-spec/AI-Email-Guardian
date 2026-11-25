"""Token-safe Telegram Bot API integration.

The bot token is read from Settings and used only in outgoing request URLs.
It is never logged, printed, stored in SQLite or included in exception text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

from email_guardian.config import Settings
from email_guardian.gmail_client import EmailMessage
from email_guardian.summarizer import ImportanceResult

LOGGER = logging.getLogger(__name__)

API_BASE_URL = "https://api.telegram.org"
GMAIL_WEB_URL = "https://mail.google.com/mail"
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 15
MAX_CHAT_LIMIT = 100
_REDACTED = "<redacted>"

TEST_MESSAGE_TEXT = (
    "AI Email Guardian\n"
    "Telegram notification test successful!\n"
    "Your bot is connected and can deliver notifications to your Telegram account."
)


class TelegramError(RuntimeError):
    """Raised when Telegram cannot be reached or rejects a request."""


@dataclass(frozen=True)
class TelegramChat:
    """Public chat identity returned by getUpdates."""

    chat_id: str
    title: str
    chat_type: str


@dataclass(frozen=True)
class TelegramBotIdentity:
    """Bot identity confirmed by getMe (never contains the token)."""

    username: str


@dataclass(frozen=True)
class TelegramDelivery:
    """Message information returned by a confirmed sendMessage call."""

    message_id: int
    chat_id: str


def _account_label(account: str) -> str:
    label = str(account or "").replace("_", " ").replace("-", " ").strip()
    return label.title() or "Unknown"


def _clean_rfc822_message_id(value: str) -> str:
    cleaned = str(value or "").strip()
    if len(cleaned) >= 2 and cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def gmail_message_url(account_email: str, rfc822_message_id: str) -> str:
    """Build a Gmail search link for one Message-ID in a specific account.

    Gmail selects signed-in accounts by numeric profile index plus ``authuser``;
    an email address in the ``/u/`` path is not a valid selector. The RFC 822
    Message-ID is used verbatim (without its angle brackets) so the link cannot
    point at a Gmail API id or thread id by mistake.
    """

    message_id = _clean_rfc822_message_id(rfc822_message_id)
    if not account_email or not message_id:
        return ""
    authuser = quote(str(account_email).strip(), safe="")
    query = quote(f"rfc822msgid:{message_id}", safe="")
    return f"{GMAIL_WEB_URL}/u/0/?authuser={authuser}#search/{query}"


class TelegramNotifier:
    """Sends alerts and discovers chats using the local Bot API configuration."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._timeout = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

    def has_token(self) -> bool:
        return bool(self._token)

    def has_chat(self) -> bool:
        return bool(self._chat_id)

    def check_availability(self) -> bool:
        if not self.has_token():
            LOGGER.warning("Telegram bot token is not configured")
            return False
        try:
            result = self._request("getMe")
        except TelegramError as exc:
            LOGGER.warning("Telegram availability check failed: %s", exc)
            return False
        return isinstance(result, Mapping) and bool(result.get("is_bot"))

    def bot_identity(self) -> TelegramBotIdentity:
        """Confirm the configured bot with getMe and return its public identity."""

        if not self.has_token():
            raise TelegramError("TELEGRAM_BOT_TOKEN is not configured.")
        result = self._request("getMe")
        if not isinstance(result, Mapping) or not bool(result.get("is_bot")):
            raise TelegramError("Telegram getMe did not return a bot identity.")
        username = str(result.get("username") or "unknown")
        return TelegramBotIdentity(username=username)

    def send_test_message(self) -> TelegramDelivery:
        """Deliver a plain-text test message and confirm Telegram's response."""

        return self._send_text(TEST_MESSAGE_TEXT)

    def send_alert(
        self,
        email: EmailMessage,
        analysis: ImportanceResult,
        gmail_url: str | None = None,
    ) -> TelegramDelivery | None:
        """Send one alert; return confirmation only when Telegram confirms it."""

        if not self.has_token():
            LOGGER.warning("Telegram bot token is not configured; alert not sent")
            return None
        if not self.has_chat():
            LOGGER.warning("Telegram chat id is not configured; alert not sent")
            return None
        markup = None
        if gmail_url:
            markup = {
                "inline_keyboard": [
                    [{"text": "Open in Gmail", "url": gmail_url}]
                ]
            }
        try:
            return self._send_text(
                self._format_alert(email, analysis), reply_markup=markup
            )
        except TelegramError as exc:
            LOGGER.warning("Telegram alert was not delivered: %s", exc)
            return None

    def _send_text(
        self, text: str, *, reply_markup: dict[str, Any] | None = None
    ) -> TelegramDelivery:
        if not self.has_token():
            raise TelegramError("TELEGRAM_BOT_TOKEN is not configured.")
        if not self.has_chat():
            raise TelegramError("TELEGRAM_CHAT_ID is not configured.")

        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = self._request("sendMessage", payload=payload)
        if not isinstance(result, Mapping):
            raise TelegramError("Telegram sendMessage did not confirm delivery.")

        message_id = result.get("message_id")
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise TelegramError("Telegram sendMessage did not return a message id.")

        chat = result.get("chat")
        chat_id = str(self._chat_id)
        if isinstance(chat, Mapping) and chat.get("id") is not None:
            returned_id = str(chat["id"])
            if returned_id != chat_id:
                raise TelegramError(
                    "Telegram sendMessage confirmed a different chat than configured."
                )
            chat_id = returned_id
        return TelegramDelivery(message_id=message_id, chat_id=chat_id)

    def recent_chats(self, limit: int = 20) -> list[TelegramChat]:
        if not self.has_token():
            raise TelegramError("TELEGRAM_BOT_TOKEN is not configured.")
        count = max(1, min(int(limit), MAX_CHAT_LIMIT))
        result = self._request("getUpdates", params={"offset": -count, "limit": count})
        if not isinstance(result, list):
            return []

        chats: list[TelegramChat] = []
        seen: set[str] = set()
        for update in result:
            chat = self._chat_from_update(update)
            if chat is None or chat.chat_id in seen:
                continue
            seen.add(chat.chat_id)
            chats.append(chat)
        return chats

    def _format_alert(
        self, email: EmailMessage, analysis: ImportanceResult
    ) -> str:
        received = (
            email.received_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            if email.received_at
            else "unknown date"
        )
        lines = [
            f"[{analysis.category}] {analysis.score}/10 "
            f"(confidence {analysis.confidence_percent}%)",
            f"Account: {_account_label(email.account)}",
            f"From: {email.sender}",
            f"Subject: {email.subject}",
            f"Received: {received}",
            f"Summary: {analysis.summary}",
            f"Reason: {analysis.reason}",
        ]
        if analysis.required_action:
            lines.append(f"Action: {analysis.required_action}")
        if analysis.deadline:
            lines.append(f"Deadline: {analysis.deadline}")
        return "\n".join(lines)

    def _chat_from_update(self, update: Any) -> TelegramChat | None:
        if not isinstance(update, Mapping):
            return None
        for key in (
            "message",
            "channel_post",
            "edited_message",
            "edited_channel_post",
            "callback_query",
            "my_chat_member",
            "chat_member",
            "chat_join_request",
        ):
            item = update.get(key)
            if isinstance(item, Mapping) and isinstance(item.get("chat"), Mapping):
                chat = item["chat"]
                chat_id = chat.get("id")
                if chat_id is None:
                    return None
                title = chat.get("title")
                if not title:
                    name = " ".join(
                        part
                        for part in (
                            str(chat.get("first_name") or ""),
                            str(chat.get("last_name") or ""),
                        )
                        if part
                    )
                    title = name or chat.get("username") or "unnamed chat"
                return TelegramChat(
                    chat_id=str(chat_id),
                    title=str(title),
                    chat_type=str(chat.get("type") or "unknown"),
                )
        return None

    def _request(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{API_BASE_URL}/bot{self._token}/{method}"
        try:
            if payload is None:
                response = requests.get(url, params=params, timeout=self._timeout)
            else:
                response = requests.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            message = f"Telegram {method} request failed: {self._safe(exc)}"
            LOGGER.warning("%s", message)
            raise TelegramError(message) from exc

        if response.status_code != 200:
            detail = self._response_detail(response)
            message = (
                f"Telegram {method} returned HTTP {response.status_code}: {detail}"
            )
            LOGGER.warning("%s", message)
            raise TelegramError(message)

        try:
            data = response.json()
        except ValueError as exc:
            message = f"Telegram {method} returned a non-JSON response."
            LOGGER.warning("%s", message)
            raise TelegramError(message) from exc

        if not isinstance(data, Mapping) or data.get("ok") is not True:
            detail = ""
            if isinstance(data, Mapping):
                detail = self._safe(data.get("description") or "request rejected")
            message = f"Telegram {method} was rejected: {detail}"
            LOGGER.warning("%s", message)
            raise TelegramError(message)
        return data.get("result")

    def _response_detail(self, response: Any) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, Mapping):
            description = payload.get("description")
            if isinstance(description, str) and description:
                return self._safe(description)
        return "no error description"

    def _safe(self, value: Any) -> str:
        text = str(value)
        if self._token and self._token in text:
            text = text.replace(self._token, _REDACTED)
        return text[:500]
