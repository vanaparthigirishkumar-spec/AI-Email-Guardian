"""Tests for the Gmail integration using mocked API responses."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

from email_guardian.config import Settings, load_settings
from email_guardian.gmail_client import (
    METADATA_HEADERS,
    READONLY_SCOPE,
    SCOPES,
    EmailMessage,
    GmailApiError,
    GmailAuthError,
    GmailClient,
    GmailError,
)

GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


def make_settings(tmp: Path, **overrides: str) -> Settings:
    env = {
        "GMAIL_ACCOUNTS": "personal,work",
        "GMAIL_CREDENTIALS_FILE": str(tmp / "credentials.json"),
        "GMAIL_TOKEN_DIR": str(tmp / "tokens"),
        "DATABASE_PATH": str(tmp / "data" / "emails.db"),
        "LOG_DIR": str(tmp / "logs"),
    }
    env.update(overrides)
    with patch.dict("os.environ", env, clear=True):
        return load_settings(env_file=tmp / "absent.env")


def write_token(
    settings: Settings,
    account: str = "personal",
    *,
    email: str = "owner@example.com",
    scopes: tuple[str, ...] = SCOPES,
    expired: bool = False,
    token: str = "access-token",
) -> Path:
    path = settings.gmail_token_path(account)
    path.parent.mkdir(parents=True, exist_ok=True)
    expiry = datetime.now(timezone.utc) + (
        timedelta(hours=-1) if expired else timedelta(hours=1)
    )
    path.write_text(
        json.dumps(
            {
                "token": token,
                "refresh_token": "test-refresh-token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "scopes": list(scopes),
                "expiry": expiry.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    settings.gmail_metadata_path(account).write_text(
        json.dumps(
            {
                "account": account,
                "email": email,
                "scopes": list(scopes),
                "authenticated_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def message_payload(message_id: str, **overrides: object) -> dict:
    payload: dict = {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Secret snippet content",
        "internalDate": "1700000000000",
        "payload": {
            "headers": [
                {"name": "From", "value": '"Jane Doe" <Jane@Example.com>'},
                {"name": "To", "value": "owner@example.com"},
                {"name": "Subject", "value": "Quarterly report"},
                {"name": "Date", "value": "Tue, 14 Nov 2023 22:13:20 +0000"},
            ]
        },
    }
    payload.update(overrides)
    return payload


def fake_service(pages: list[dict], payloads: list[dict]):
    service = MagicMock()
    messages_api = service.users.return_value.messages.return_value

    list_request = MagicMock()
    list_request.execute.side_effect = list(pages)
    messages_api.list.return_value = list_request

    get_requests = []
    for payload in payloads:
        request = MagicMock()
        request.execute.return_value = payload
        get_requests.append(request)
    messages_api.get.side_effect = get_requests
    return service, messages_api


class FakeResponse:
    def __init__(self, status: int, reason: str = "Error") -> None:
        self.status = status
        self.reason = reason


def http_error(status: int, reason: str = "backendError", message: str = "boom") -> HttpError:
    content = json.dumps(
        {"error": {"code": status, "message": message, "errors": [{"reason": reason}]}}
    ).encode("utf-8")
    return HttpError(FakeResponse(status), content)


class GmailClientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def client(self, settings: Settings, service=None, account: str = "personal") -> GmailClient:
        return GmailClient(settings, account, service=service)


class FetchRecentTest(GmailClientTestCase):
    def test_fetch_uses_configurable_query_and_metadata_format(self) -> None:
        settings = make_settings(self.tmp, GMAIL_QUERY="in:inbox newer_than:2d")
        service, messages_api = fake_service(
            [{"messages": [{"id": "m1"}]}], [message_payload("m1")]
        )

        messages = self.client(settings, service).fetch_recent(limit=1)

        self.assertEqual(len(messages), 1)
        messages_api.list.assert_called_once_with(
            userId="me", q="in:inbox newer_than:2d", maxResults=1, pageToken=None
        )
        get_kwargs = messages_api.get.call_args.kwargs
        self.assertEqual(get_kwargs["format"], "metadata")
        self.assertEqual(get_kwargs["metadataHeaders"], list(METADATA_HEADERS))
        self.assertEqual(get_kwargs["id"], "m1")

    def test_default_query_observes_opened_messages(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service(
            [{"messages": [{"id": "m1"}]}], [message_payload("m1")]
        )

        self.client(settings, service).fetch_recent(limit=1)

        query = messages_api.list.call_args.kwargs["q"]
        self.assertIn("in:inbox", query)
        self.assertNotIn("is:unread", query)

    def test_fetch_paginates_until_limit(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service(
            [
                {"messages": [{"id": "m1"}], "nextPageToken": "next"},
                {"messages": [{"id": "m2"}, {"id": "m3"}]},
            ],
            [message_payload("m1"), message_payload("m2"), message_payload("m3")],
        )

        messages = self.client(settings, service).fetch_recent(limit=3)

        self.assertEqual([m.message_id for m in messages], ["m1", "m2", "m3"])
        self.assertEqual(messages_api.list.call_count, 2)
        second_call = messages_api.list.call_args_list[1].kwargs
        self.assertEqual(second_call["pageToken"], "next")
        self.assertEqual(second_call["maxResults"], 2)

    def test_fetch_stops_at_limit(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service(
            [{"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}],
            [message_payload("m1"), message_payload("m2"), message_payload("m3")],
        )

        messages = self.client(settings, service).fetch_recent(limit=2)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages_api.get.call_count, 2)

    def test_fetch_deduplicates_message_ids_across_pages(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service(
            [
                {"messages": [{"id": "m1"}], "nextPageToken": "next"},
                {"messages": [{"id": "m1"}, {"id": "m2"}]},
            ],
            [message_payload("m1"), message_payload("m2")],
        )

        messages = self.client(settings, service).fetch_recent(limit=2)

        self.assertEqual([m.message_id for m in messages], ["m1", "m2"])
        self.assertEqual(messages_api.get.call_count, 2)

    def test_zero_limit_fetches_nothing(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service([], [])

        self.assertEqual(self.client(settings, service).fetch_recent(limit=0), [])
        messages_api.list.assert_not_called()

    def test_parses_metadata_headers_and_timestamps(self) -> None:
        settings = make_settings(self.tmp)
        payload = message_payload("m1", payload={
            "headers": [
                {"name": "From", "value": '"Jane Doe" <Jane@Example.com>'},
                {"name": "Subject", "value": "=?utf-8?B?VGVzdCDwn5iA?="},
            ]
        })
        service, _ = fake_service([{"messages": [{"id": "m1"}]}], [payload])

        message = self.client(settings, service).fetch_recent(limit=1)[0]

        self.assertIsInstance(message, EmailMessage)
        self.assertEqual(message.message_id, "m1")
        self.assertEqual(message.thread_id, "thread-m1")
        self.assertEqual(message.account, "personal")
        self.assertEqual(message.sender_name, "Jane Doe")
        self.assertEqual(message.sender_email, "Jane@Example.com")
        self.assertEqual(message.sender, "Jane Doe <Jane@Example.com>")
        self.assertEqual(message.subject, "Test 😀")
        self.assertEqual(
            message.received_at,
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )
        self.assertEqual(message.labels, ("INBOX", "UNREAD"))
        self.assertEqual(message.snippet, "Secret snippet content")

    def test_missing_headers_use_safe_fallbacks(self) -> None:
        settings = make_settings(self.tmp)
        payload = {"id": "m1", "threadId": "t1", "payload": {"headers": []}}
        service, _ = fake_service([{"messages": [{"id": "m1"}]}], [payload])

        message = self.client(settings, service).fetch_recent(limit=1)[0]

        self.assertEqual(message.subject, "(no subject)")
        self.assertEqual(message.sender, "unknown sender")
        self.assertIsNone(message.received_at)
        self.assertEqual(message.labels, ())


class ProfileTest(GmailClientTestCase):
    def test_profile_returns_address_and_counters(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings)
        service = MagicMock()
        service.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "owner@example.com",
            "messagesTotal": 42,
            "threadsTotal": 17,
        }

        profile = self.client(settings, service).get_profile()

        self.assertEqual(profile.account, "personal")
        self.assertEqual(profile.email, "owner@example.com")
        self.assertEqual(profile.messages_total, 42)
        self.assertEqual(profile.threads_total, 17)

    def test_profile_rejects_configured_email_mismatch(self) -> None:
        settings = make_settings(
            self.tmp, GMAIL_ACCOUNT_PERSONAL_EMAIL="expected@example.com"
        )
        write_token(settings)
        service = MagicMock()
        service.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "someone-else@example.com"
        }

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings, service).get_profile()

        self.assertIn("expected@example.com", str(ctx.exception))
        self.assertIn("gmail auth personal", str(ctx.exception))

    def test_profile_rejects_token_metadata_mismatch(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings, email="previous@example.com")
        service = MagicMock()
        service.users.return_value.getProfile.return_value.execute.return_value = {
            "emailAddress": "owner@example.com"
        }

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings, service).get_profile()

        self.assertIn("previous@example.com", str(ctx.exception))
        self.assertIn("gmail auth personal", str(ctx.exception))

    def test_missing_token_reports_reauth_command(self) -> None:
        settings = make_settings(self.tmp)

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings).get_profile()

        self.assertIn("python main.py gmail auth personal", str(ctx.exception))

    def test_constructor_rejects_unconfigured_account(self) -> None:
        settings = make_settings(self.tmp)

        with self.assertRaises(GmailError):
            GmailClient(settings, "unknown")


class CredentialScopeTest(GmailClientTestCase):
    def test_broader_scope_token_is_rejected(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings, scopes=(READONLY_SCOPE, GMAIL_MODIFY_SCOPE))

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings).load_credentials()

        self.assertIn("broader access", str(ctx.exception))

    def test_missing_readonly_scope_is_rejected(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings, scopes=("openid",))

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings).load_credentials()

        self.assertIn("gmail.readonly", str(ctx.exception))

    def test_unreadable_token_file_is_reported(self) -> None:
        settings = make_settings(self.tmp)
        path = settings.gmail_token_path("personal")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        with self.assertRaises(GmailAuthError) as ctx:
            self.client(settings).load_credentials()

        self.assertIn("unreadable", str(ctx.exception))


class RefreshTest(GmailClientTestCase):
    def test_expired_token_is_refreshed_and_saved(self) -> None:
        settings = make_settings(self.tmp)
        token_path = write_token(settings, expired=True)

        def fake_refresh(credentials: Credentials, request) -> None:
            credentials.token = "refreshed-token"
            credentials.expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        client = self.client(settings)
        with patch.object(Credentials, "refresh", autospec=True, side_effect=fake_refresh):
            credentials = client.load_credentials()

        self.assertEqual(credentials.token, "refreshed-token")
        stored = json.loads(token_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["token"], "refreshed-token")

    def test_revoked_token_reports_reauthentication(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings, expired=True)

        client = self.client(settings)
        with patch.object(
            Credentials, "refresh", autospec=True, side_effect=RefreshError("invalid_grant")
        ):
            with self.assertRaises(GmailAuthError) as ctx:
                client.load_credentials()

        self.assertIn("revoked", str(ctx.exception))
        self.assertIn("python main.py gmail auth personal", str(ctx.exception))

    def test_token_status_does_not_refresh(self) -> None:
        settings = make_settings(self.tmp)
        write_token(settings, expired=True)

        with patch.object(Credentials, "refresh") as refresh:
            status = self.client(settings).token_status()

        refresh.assert_not_called()
        self.assertTrue(status.present)
        self.assertTrue(status.expired)
        self.assertTrue(status.has_refresh_token)
        self.assertEqual(status.email, "owner@example.com")
        self.assertIsNone(status.error)

    def test_token_status_missing_token(self) -> None:
        settings = make_settings(self.tmp)

        status = self.client(settings).token_status()

        self.assertFalse(status.present)
        self.assertIsNone(status.email)


class ApiErrorHandlingTest(GmailClientTestCase):
    def test_transient_error_is_retried(self) -> None:
        settings = make_settings(self.tmp)
        service, messages_api = fake_service(
            [http_error(503), {"messages": [{"id": "m1"}]}],
            [message_payload("m1")],
        )

        with patch("email_guardian.gmail_client.time.sleep") as sleep:
            messages = self.client(settings, service).fetch_recent(limit=1)

        self.assertEqual([m.message_id for m in messages], ["m1"])
        self.assertEqual(messages_api.list.return_value.execute.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_401_is_not_retried(self) -> None:
        settings = make_settings(self.tmp)
        service, _ = fake_service([http_error(401, "authError")], [])

        with patch("email_guardian.gmail_client.time.sleep") as sleep:
            with self.assertRaises(GmailAuthError):
                self.client(settings, service).fetch_recent(limit=1)

        sleep.assert_not_called()

    def test_client_error_is_not_retried_and_hides_body(self) -> None:
        settings = make_settings(self.tmp)
        service, _ = fake_service([http_error(400, "badRequest", message="secret details")], [])

        with patch("email_guardian.gmail_client.time.sleep") as sleep:
            with self.assertRaises(GmailApiError) as ctx:
                self.client(settings, service).fetch_recent(limit=1)

        sleep.assert_not_called()
        self.assertEqual(ctx.exception.status, 400)
        self.assertNotIn("secret details", str(ctx.exception))


class LoggingPrivacyTest(GmailClientTestCase):
    def test_logs_do_not_contain_message_content(self) -> None:
        settings = make_settings(self.tmp)
        service, _ = fake_service(
            [{"messages": [{"id": "m1"}]}], [message_payload("m1")]
        )

        with self.assertLogs("email_guardian.gmail_client", level="DEBUG") as captured:
            self.client(settings, service).fetch_recent(limit=1)

        output = "\n".join(captured.output)
        self.assertIn("m1", output)
        self.assertNotIn("Secret snippet content", output)
        self.assertNotIn("Quarterly report", output)
        self.assertNotIn("Jane@Example.com", output)
        self.assertNotIn("access-token", output)


class AuthenticateTest(GmailClientTestCase):
    def test_missing_client_file_is_reported(self) -> None:
        settings = make_settings(self.tmp)

        with self.assertRaises(GmailError) as ctx:
            self.client(settings).authenticate()

        self.assertIn("OAuth client file not found", str(ctx.exception))

    def test_desktop_flow_stores_token_and_metadata(self) -> None:
        settings = make_settings(self.tmp)
        settings.gmail_credentials_file.write_text("{}", encoding="utf-8")
        credentials = Credentials(
            token="new-access-token",
            refresh_token="new-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=[READONLY_SCOPE],
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        flow = MagicMock()
        flow.run_local_server.return_value = credentials

        client = self.client(settings)
        with patch.object(
            InstalledAppFlow,
            "from_client_secrets_file",
            return_value=flow,
        ) as factory, patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com", "messagesTotal": 3},
        ):
            email = client.authenticate()

        self.assertEqual(email, "owner@example.com")
        factory.assert_called_once_with(
            str(settings.gmail_credentials_file), [READONLY_SCOPE]
        )
        flow.run_local_server.assert_called_once_with(
            port=0, access_type="offline", prompt="select_account consent"
        )

        token_data = json.loads(
            settings.gmail_token_path("personal").read_text(encoding="utf-8")
        )
        self.assertEqual(token_data["scopes"], [READONLY_SCOPE])
        self.assertEqual(token_data["token"], "new-access-token")

        metadata = json.loads(
            settings.gmail_metadata_path("personal").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["email"], "owner@example.com")
        self.assertEqual(metadata["scopes"], [READONLY_SCOPE])

    def test_valid_token_is_reused_without_opening_a_browser(self) -> None:
        settings = make_settings(self.tmp)
        settings.gmail_credentials_file.write_text("{}", encoding="utf-8")
        write_token(settings)

        client = self.client(settings)
        with patch.object(
            InstalledAppFlow,
            "from_client_secrets_file",
        ) as factory, patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            email = client.authenticate()

        self.assertEqual(email, "owner@example.com")
        factory.assert_not_called()

    def test_flow_with_broader_scope_is_rejected(self) -> None:
        settings = make_settings(self.tmp)
        settings.gmail_credentials_file.write_text("{}", encoding="utf-8")
        credentials = Credentials(
            token="new-access-token",
            refresh_token="new-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=[READONLY_SCOPE, GMAIL_MODIFY_SCOPE],
            expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        flow = MagicMock()
        flow.run_local_server.return_value = credentials

        client = self.client(settings)
        with patch.object(
            InstalledAppFlow,
            "from_client_secrets_file",
            return_value=flow,
        ):
            with self.assertRaises(GmailAuthError):
                client.authenticate()

        self.assertFalse(settings.gmail_token_path("personal").exists())


def oauth_credentials(*, scopes=None, refresh_token="new-refresh-token") -> Credentials:
    return Credentials(
        token="new-access-token",
        refresh_token=refresh_token,
        client_id="test-client-id",
        client_secret="test-client-secret",
        token_uri="https://oauth2.googleapis.com/token",
        scopes=list(SCOPES) if scopes is None else list(scopes),
        expiry=datetime.now(timezone.utc) + timedelta(hours=1),
    )


class TokenPersistenceTest(GmailClientTestCase):
    def settings_with_client_file(self, **env: str) -> Settings:
        settings = make_settings(self.tmp, **env)
        settings.gmail_credentials_file.write_text("{}", encoding="utf-8")
        return settings

    @staticmethod
    def flow_returning(credentials: Credentials) -> MagicMock:
        flow = MagicMock()
        flow.run_local_server.return_value = credentials
        return flow

    def test_token_is_written_to_the_account_specific_path(self) -> None:
        settings = self.settings_with_client_file()
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            email = client.authenticate()

        self.assertEqual(email, "owner@example.com")
        self.assertEqual(client.token_path, settings.gmail_token_path("personal"))
        self.assertEqual(client.token_path.name, "personal.json")
        self.assertTrue(client.token_path.is_file())
        self.assertFalse(settings.gmail_token_path("work").exists())

        data = json.loads(client.token_path.read_text(encoding="utf-8"))
        self.assertEqual(data["token"], "new-access-token")
        self.assertEqual(data["refresh_token"], "new-refresh-token")
        self.assertEqual(data["scopes"], [READONLY_SCOPE])

        metadata = json.loads(
            settings.gmail_metadata_path("personal").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["email"], "owner@example.com")

    def test_token_is_saved_even_when_profile_verification_fails(self) -> None:
        settings = self.settings_with_client_file()
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            side_effect=GmailApiError("Gmail API request failed", status=403),
        ):
            with self.assertRaises(GmailApiError) as ctx:
                client.authenticate()

        self.assertIn("OAuth token was saved", str(ctx.exception))
        self.assertTrue(client.token_path.is_file())
        self.assertFalse(settings.gmail_metadata_path("personal").exists())

    def test_identity_mismatch_keeps_the_token_for_a_retry(self) -> None:
        settings = self.settings_with_client_file(
            GMAIL_ACCOUNT_PERSONAL_EMAIL="expected@example.com"
        )
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "wrong@example.com"},
        ):
            with self.assertRaises(GmailAuthError) as ctx:
                client.authenticate()

        self.assertIn("GMAIL_ACCOUNT_PERSONAL_EMAIL", str(ctx.exception))
        self.assertTrue(client.token_path.is_file())
        metadata = json.loads(
            settings.gmail_metadata_path("personal").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["email"], "wrong@example.com")

    def test_identity_mismatch_on_reuse_does_not_restart_the_browser(self) -> None:
        settings = self.settings_with_client_file(
            GMAIL_ACCOUNT_PERSONAL_EMAIL="expected@example.com"
        )
        client = GmailClient(settings, "personal")
        settings.gmail_token_dir.mkdir(parents=True)
        client.token_path.write_text(
            oauth_credentials().to_json(), encoding="utf-8"
        )

        with patch.object(InstalledAppFlow, "from_client_secrets_file") as factory, \
                patch.object(
                    GmailClient,
                    "_profile_payload",
                    return_value={"emailAddress": "owner@example.com"},
                ):
            with self.assertRaises(GmailAuthError):
                client.authenticate()

        factory.assert_not_called()
        self.assertTrue(client.token_path.is_file())

    def test_stale_metadata_does_not_block_a_fresh_sign_in(self) -> None:
        settings = self.settings_with_client_file()
        settings.gmail_metadata_path("personal").parent.mkdir(parents=True)
        settings.gmail_metadata_path("personal").write_text(
            json.dumps(
                {
                    "account": "personal",
                    "email": "previous@example.com",
                    "scopes": [READONLY_SCOPE],
                }
            ),
            encoding="utf-8",
        )
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            email = client.authenticate()

        self.assertEqual(email, "owner@example.com")
        metadata = json.loads(
            settings.gmail_metadata_path("personal").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["email"], "owner@example.com")

    def test_token_directory_is_created(self) -> None:
        settings = self.settings_with_client_file()
        self.assertFalse(settings.gmail_token_dir.exists())
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            client.authenticate()

        self.assertTrue(settings.gmail_token_dir.is_dir())
        self.assertTrue(client.token_path.is_file())

    def test_unwritable_token_directory_fails_before_the_browser_opens(self) -> None:
        settings = make_settings(self.tmp)
        settings.gmail_token_dir.write_text("not a directory", encoding="utf-8")
        client = GmailClient(settings, "personal")

        with patch.object(InstalledAppFlow, "from_client_secrets_file") as factory:
            with self.assertRaises(GmailError) as ctx:
                client.authenticate()

        factory.assert_not_called()
        self.assertIn("not writable", str(ctx.exception))

    def test_oauth_flow_failure_is_reported(self) -> None:
        settings = self.settings_with_client_file()
        client = GmailClient(settings, "personal")
        flow = MagicMock()
        flow.run_local_server.side_effect = RuntimeError("local server closed")

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ):
            with self.assertRaises(GmailAuthError) as ctx:
                client.authenticate()

        message = str(ctx.exception)
        self.assertIn("OAuth flow did not complete", message)
        self.assertIn("RuntimeError", message)
        self.assertFalse(client.token_path.exists())

    def test_missing_reported_scopes_fall_back_to_readonly(self) -> None:
        settings = self.settings_with_client_file()
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials(scopes=[]))

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ):
            client.authenticate()

        data = json.loads(client.token_path.read_text(encoding="utf-8"))
        self.assertEqual(data["scopes"], [READONLY_SCOPE])

    def test_auth_logs_never_contain_tokens_or_secrets(self) -> None:
        settings = self.settings_with_client_file()
        client = GmailClient(settings, "personal")
        flow = self.flow_returning(oauth_credentials())

        with patch.object(
            InstalledAppFlow, "from_client_secrets_file", return_value=flow
        ), patch.object(
            GmailClient,
            "_profile_payload",
            return_value={"emailAddress": "owner@example.com"},
        ), self.assertLogs("email_guardian.gmail_client", level="DEBUG") as captured:
            client.authenticate()

        output = "\n".join(captured.output)
        self.assertNotIn("new-access-token", output)
        self.assertNotIn("new-refresh-token", output)
        self.assertNotIn("test-client-secret", output)


if __name__ == "__main__":
    unittest.main()
