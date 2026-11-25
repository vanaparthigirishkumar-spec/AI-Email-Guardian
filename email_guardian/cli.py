"""Command-line interface for AI Email Guardian."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from email_guardian import __version__
from email_guardian.agent import CycleReport, EmailGuardianAgent, should_notify
from email_guardian.config import (
    PROJECT_ROOT,
    ConfigError,
    Settings,
    ensure_runtime_dirs,
    load_settings,
)
from email_guardian.gmail_client import (
    EmailMessage,
    GmailAuthError,
    GmailClient,
    GmailError,
    TokenStatus,
)
from email_guardian.logging_setup import setup_logging, shutdown_logging
from email_guardian.notifier import TelegramError, TelegramNotifier
from email_guardian.storage import EmailStore, NotificationStatus, StorageError
from email_guardian.summarizer import (
    Classification,
    OllamaAnalyzer,
    OllamaError,
    OllamaUnavailableError,
)

MIN_PYTHON = (3, 10)


def _write_dir_probe(directory: Path) -> bool:
    probe = directory / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _report(ok: bool, label: str, detail: str = "") -> None:
    marker = "OK  " if ok else "FAIL"
    print(f"[{marker}] {label}" + (f" - {detail}" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    print(f"[WARN] {label}" + (f" - {detail}" if detail else ""))


def _check_secret(settings: Settings, field: str, label: str) -> bool:
    configured = settings.is_configured(field)
    _report(configured, label, settings.masked(field) if configured else "add it to .env")
    return configured


def _check_optional_secret(settings: Settings, field: str, label: str) -> bool:
    configured = settings.is_configured(field)
    if configured:
        _report(True, label, settings.masked(field))
    else:
        _warn(label, "not configured - expected before Phase 5")
    return configured


def _doctor_gmail(settings: Settings, *, online: bool) -> bool:
    print("\nGmail:")

    credentials_ok = settings.gmail_credentials_file.is_file()
    _report(
        credentials_ok,
        "OAuth client file",
        str(settings.gmail_credentials_file)
        if credentials_ok
        else f"missing - expected at {settings.gmail_credentials_file}",
    )

    if not settings.gmail_accounts:
        _report(False, "Configured accounts", "set GMAIL_ACCOUNTS in .env (e.g. personal,work)")
        return False

    _report(True, "Configured accounts", ", ".join(settings.gmail_accounts))

    all_ok = credentials_ok
    for account in settings.gmail_accounts:
        all_ok = _doctor_gmail_account(settings, account, online=online) and all_ok
    return all_ok


def _doctor_gmail_account(settings: Settings, account: str, *, online: bool) -> bool:
    client = GmailClient(settings, account)
    status = client.token_status()
    ok = True

    print(f"  {account}:")
    if not status.present:
        _report(False, "Token", f"missing - run: python main.py gmail auth {account}")
        return False

    _report(True, "Token file", str(client.token_path))

    if status.error:
        _report(False, "Token permissions", status.error)
        ok = False
    else:
        _report(True, "Token permissions", "gmail.readonly")

    if status.email:
        _report(True, "Account email", status.email)
    else:
        _report(False, "Account email", "unknown - re-authenticate to record it")
        ok = False

    if not status.has_refresh_token:
        _report(False, "Refresh token", "missing - re-authenticate")
        ok = False
    elif status.expired:
        _warn("Access token", "expired - will refresh automatically on next use")

    expected = settings.expected_email_for(account)
    if expected and status.email and expected.lower() != status.email.lower():
        _report(False, "Expected email", f"{expected} (token belongs to {status.email})")
        ok = False

    if online:
        try:
            profile = client.get_profile()
        except GmailError as exc:
            _report(False, "Live profile", str(exc))
            ok = False
        else:
            _report(
                True,
                "Live profile",
                f"{profile.email} ({profile.messages_total} messages, "
                f"{profile.threads_total} threads)",
            )

    return ok


def run_doctor(settings: Settings, *, online: bool = False) -> int:
    """Print a setup report, optionally contacting Gmail for each account."""

    print(f"AI Email Guardian {__version__} - setup check\n")

    version_ok = sys.version_info >= MIN_PYTHON
    current = ".".join(str(part) for part in sys.version_info[:3])
    required = ".".join(str(part) for part in MIN_PYTHON)
    _report(version_ok, "Python version", f"{current} (requires >= {required})")

    env_ok = (PROJECT_ROOT / ".env").is_file()
    _report(
        env_ok,
        ".env file",
        str(PROJECT_ROOT / ".env") if env_ok else "missing - copy .env.example to .env",
    )

    ensure_runtime_dirs(settings)
    data_ok = _write_dir_probe(settings.data_dir)
    _report(data_ok, "Data directory writable", str(settings.data_dir))
    tokens_ok = _write_dir_probe(settings.gmail_token_dir)
    _report(tokens_ok, "Token directory writable", str(settings.gmail_token_dir))
    logs_ok = _write_dir_probe(settings.log_dir)
    _report(logs_ok, "Log directory writable", str(settings.log_dir))

    db_ok = True
    store = EmailStore(settings)
    try:
        store.initialize()
        db_version = store.schema_version()
    except StorageError as exc:
        db_ok = False
        db_detail = str(exc)
    else:
        db_detail = f"schema version {db_version}, WAL enabled"
    finally:
        store.close()
    _report(db_ok, "Database schema", db_detail)

    gmail_ok = _doctor_gmail(settings, online=online)

    print("\nIntegrations:")
    _report(True, "Ollama endpoint", settings.ollama_base_url)
    _report(True, "Ollama model", settings.ollama_model)
    _check_optional_secret(settings, "telegram_bot_token", "Telegram bot token")
    _check_optional_secret(settings, "telegram_chat_id", "Telegram chat id")

    print("\nRuntime:")
    print(f"  Poll interval : {settings.poll_interval_seconds}s")
    print(f"  Gmail query   : {settings.gmail_query}")
    print(f"  Database      : {settings.database_path}")
    print(f"  Log level     : {settings.log_level}")

    core_ok = version_ok and data_ok and tokens_ok and logs_ok and db_ok
    if not core_ok:
        print("\nFix the failing checks above before continuing.")
        return 1
    if online and not gmail_ok:
        print("\nGmail checks failed while running with --online.")
        return 1

    print(
        "\nCore setup looks good. Account status: "
        f"{_account_status_summary(settings)}"
    )
    return 0


def _list_gmail_accounts(settings: Settings) -> int:
    if not settings.gmail_accounts:
        print(
            "No Gmail accounts configured. Set GMAIL_ACCOUNTS in .env, "
            "for example: personal,work"
        )
        return 1

    for account in settings.gmail_accounts:
        status = GmailClient(settings, account).token_status()
        state = _describe_token(status)
        expected = settings.expected_email_for(account)
        if expected and status.email and expected.lower() != status.email.lower():
            state += f" (configured email is {expected})"
        print(f"  {account}: {state}")
    return 0


def _describe_token(status: TokenStatus) -> str:
    if not status.present:
        return "not authenticated"
    if status.error:
        return status.error
    details = []
    if status.email:
        details.append(f"authenticated as {status.email}")
    else:
        details.append("token present")
    if status.expired:
        details.append("access token expired (will refresh)")
    return ", ".join(details)


def _account_status_summary(settings: Settings) -> str:
    if not settings.gmail_accounts:
        return "no Gmail accounts configured"
    statuses = []
    for account in settings.gmail_accounts:
        status = GmailClient(settings, account).token_status()
        statuses.append(f"{account}: {_describe_token(status)}")
    return "; ".join(statuses)


def _print_recent_messages(messages: list) -> None:
    for message in messages:
        received = (
            message.received_at.strftime("%Y-%m-%d %H:%M")
            if message.received_at
            else "unknown date"
        )
        print(f"  - {received} | {message.sender} | {message.subject}")


def run_gmail(settings: Settings, args: argparse.Namespace) -> int:
    setup_logging(settings.log_dir, settings.log_level)

    if args.gmail_command == "accounts":
        return _list_gmail_accounts(settings)

    account = args.account
    if account not in settings.gmail_accounts:
        configured = ", ".join(settings.gmail_accounts) or "none"
        print(
            f"Gmail account {account!r} is not configured (configured: {configured}).",
            file=sys.stderr,
        )
        return 2

    client = GmailClient(settings, account)

    if args.gmail_command == "auth":
        try:
            email = client.authenticate(force=args.force)
        except GmailError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print(f"[OK  ] Authenticated {account} as {email}")
        print(f"       Token saved to {client.token_path}")
        return 0

    try:
        profile = client.get_profile()
        print(
            f"[OK  ] {account} -> {profile.email} "
            f"({profile.messages_total} messages, {profile.threads_total} threads)"
        )
        messages = client.fetch_recent(limit=args.limit)
    except GmailError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print(f"[OK  ] Fetched {len(messages)} recent message(s)")
    _print_recent_messages(messages)
    return 0


def _print_classification(
    message: EmailMessage, result: Classification, threshold: int
) -> None:
    received = (
        message.received_at.strftime("%Y-%m-%d %H:%M")
        if message.received_at
        else "unknown date"
    )
    flag = "  [meets threshold]" if result.meets_threshold(threshold) else ""
    print(
        f"  [{result.category}] {result.score}/10 "
        f"(confidence {result.confidence_percent}%){flag}"
    )
    print(f"      {received} | {message.sender} | {message.subject}")
    print(f"      Summary : {result.summary}")
    print(f"      Reason  : {result.reason}")
    print(f"      Action  : {result.required_action}")
    if result.deadline:
        print(f"      Deadline: {result.deadline}")


def run_ai(settings: Settings, args: argparse.Namespace) -> int:
    """Analyze a few recent emails once, without starting the polling loop."""

    setup_logging(settings.log_dir, settings.log_level)

    account = args.account
    if account not in settings.gmail_accounts:
        configured = ", ".join(settings.gmail_accounts) or "none"
        print(
            f"Gmail account {account!r} is not configured (configured: {configured}).",
            file=sys.stderr,
        )
        return 2

    if args.limit <= 0:
        print("Nothing to analyze (limit is 0).")
        return 0

    analyzer = OllamaAnalyzer(settings)
    try:
        analyzer.ensure_available()
    except OllamaError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if settings.ollama_timeout_seconds < 180:
        _warn(
            "Ollama timeout",
            f"{settings.ollama_timeout_seconds}s - the first request may need longer "
            "while the model loads; consider OLLAMA_TIMEOUT_SECONDS=180",
        )

    client = GmailClient(settings, account)
    try:
        profile = client.get_profile()
        messages = client.fetch_recent(limit=args.limit)
    except GmailError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if not messages:
        print(f"[OK  ] {account} -> {profile.email}; no messages matched the query.")
        return 0

    print(
        f"[OK  ] {account} -> {profile.email}; analyzing {len(messages)} message(s) "
        f"with {analyzer.model} (threshold {settings.importance_threshold})"
    )

    analyzed = 0
    failed = 0
    for message in messages:
        try:
            body = client.fetch_body(message.message_id)
            result = analyzer.analyze(message, body.text)
        except OllamaUnavailableError as exc:
            print(f"[FAIL] Ollama became unavailable: {exc}", file=sys.stderr)
            return 1
        except GmailAuthError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        except (GmailError, OllamaError) as exc:
            failed += 1
            print(f"[FAIL] {message.message_id}: {exc}", file=sys.stderr)
            continue
        analyzed += 1
        _print_classification(message, result, settings.importance_threshold)

    print(f"[OK  ] Analyzed {analyzed} message(s), {failed} failed.")
    return 0 if failed == 0 else 1


def _telegram_test(notifier: TelegramNotifier) -> int:
    """Verify the bot, deliver a real test message and confirm Telegram's reply."""

    try:
        identity = notifier.bot_identity()
    except TelegramError as exc:
        print(f"[FAIL] Telegram bot verification failed: {exc}", file=sys.stderr)
        return 1

    if not notifier.has_chat():
        print(
            "[FAIL] TELEGRAM_CHAT_ID is not configured. Add it to .env and rerun.",
            file=sys.stderr,
        )
        return 1

    try:
        delivery = notifier.send_test_message()
    except TelegramError as exc:
        print(f"[FAIL] Telegram test message was not delivered: {exc}", file=sys.stderr)
        return 1

    print("[OK  ] Telegram bot is reachable (bot token not shown).")
    print(f"[OK  ] Verified bot: @{identity.username}")
    print("[OK  ] Test message delivered and confirmed by Telegram.")
    print(f"       Chat id: {delivery.chat_id} | Message id: {delivery.message_id}")
    print("       Check your Telegram app for the test notification.")
    return 0


def run_telegram(settings: Settings, args: argparse.Namespace) -> int:
    """Send a real test notification or list recent chat ids safely."""

    setup_logging(settings.log_dir, settings.log_level)
    notifier = TelegramNotifier(settings)

    if not notifier.has_token():
        print("[FAIL] TELEGRAM_BOT_TOKEN is not configured.", file=sys.stderr)
        return 1

    if args.telegram_command == "chats":
        try:
            chats = notifier.recent_chats(limit=args.limit)
        except TelegramError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        if not chats:
            print("[OK  ] Bot is reachable; no recent chats were found.")
            return 0
        print(f"[OK  ] Found {len(chats)} recent chat(s):")
        for chat in chats:
            print(f"  - {chat.chat_id} | {chat.chat_type} | {chat.title}")
        print("Use one of these ids for TELEGRAM_CHAT_ID. The bot token is never shown.")
        return 0

    return _telegram_test(notifier)


def _print_cycle_report(report: CycleReport, *, dry_run: bool) -> None:
    if dry_run:
        print("[OK  ] Dry run complete - no Telegram notifications were delivered.")
    else:
        print("[OK  ] Processing cycle complete.")
    print(
        f"       Accounts: {report.accounts_processed} | "
        f"Fetched: {report.fetched} | Analyzed: {report.analyzed} | "
        f"Stored: {report.stored} | Duplicates: {report.duplicates}"
    )
    print(
        f"       Sent: {report.notifications_sent} | "
        f"Recovered: {report.recovered} | "
        f"Pending (dry run): {report.dry_run_pending} | "
        f"Failed: {report.failed + report.account_failures + report.notifications_failed}"
    )
    print(f"       Checkpoints updated: {report.checkpoints_updated}")


def run_process(settings: Settings, args: argparse.Namespace) -> int:
    """Run exactly one processing cycle for every configured account."""

    setup_logging(settings.log_dir, settings.log_level)

    if not settings.gmail_accounts:
        print(
            "No Gmail accounts configured. Set GMAIL_ACCOUNTS in .env, "
            "for example: personal,work",
            file=sys.stderr,
        )
        return 1

    agent = EmailGuardianAgent(settings)
    if args.process_command == "watch":
        try:
            agent.run_forever()
        except KeyboardInterrupt:
            print("\n[OK  ] Monitoring stopped (Ctrl+C).")
        return 0

    dry_run = bool(getattr(args, "dry_run", False))
    try:
        report = agent.run_once(dry_run=dry_run)
    except (OllamaError, StorageError) as exc:
        print(f"[FAIL] Processing cycle could not start: {exc}", file=sys.stderr)
        return 1

    _print_cycle_report(report, dry_run=dry_run)
    return 1 if report.has_failures else 0


def _change_suppression(settings: Settings, args: argparse.Namespace) -> int:
    """Suppress or restore one alert by changing only its notification status."""

    store = EmailStore(settings)
    account = args.account
    message_id = args.message_id
    suppressing = args.notification_command == "suppress"
    try:
        status = store.get_notification_status(account, message_id)
        if status is None:
            print(
                f"[FAIL] No notification found for account {account!r}, "
                f"message id {message_id!r}.",
                file=sys.stderr,
            )
            return 1
        if status is NotificationStatus.SENT:
            print(
                "[FAIL] This notification was already delivered; nothing was changed.",
                file=sys.stderr,
            )
            return 1

        if suppressing:
            new_status = store.suppress_notification(account, message_id)
            print(
                f"[OK  ] Suppressed notification for account {account}, "
                f"message {message_id}."
            )
            print(
                f"       Status: {new_status.value}; it will not be delivered automatically."
            )
            print(
                "       Restore with: python main.py notifications unsuppress "
                "<account> <message-id>"
            )
            return 0

        if status is NotificationStatus.PENDING:
            print("[OK  ] Notification is already pending; nothing was changed.")
            return 0
        classification = store.get_classification(account, message_id)
        if classification is None or not should_notify(
            str(classification["category"]),
            int(classification["importance_score"]),
            settings.importance_threshold,
        ):
            print(
                "[FAIL] This email is not eligible for notification; it was not restored.",
                file=sys.stderr,
            )
            return 1
        restored = store.mark_notification_pending(account, message_id)
        print(
            f"[OK  ] Restored notification for account {account}, "
            f"message {message_id}. Status: {restored.value}"
        )
        return 0
    except StorageError as exc:
        print(f"[FAIL] Could not change notification status: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


def run_notifications(settings: Settings, args: argparse.Namespace) -> int:
    """Inspect the queue or suppress/restore one notification safely."""

    setup_logging(settings.log_dir, settings.log_level)
    if getattr(args, "notification_command", None) in ("suppress", "unsuppress"):
        return _change_suppression(settings, args)

    store = EmailStore(settings)
    try:
        total = store.count_pending_notifications()
        pending = store.pending_notifications_detailed()
    except StorageError as exc:
        print(f"[FAIL] Could not read the notification queue: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(f"Pending notifications: {total}")
    if not pending:
        print("[OK  ] No pending notifications.")
        return 0

    for index, item in enumerate(pending, 1):
        print(f"  {index}. Account: {item.account_alias} | Message id: {item.message_id}")
        print(f"     Sender : {item.sender or '(unknown)'}")
        print(f"     Subject: {item.subject or '(no subject)'}")
        print(
            f"     AI classification: {item.category or 'unknown'} "
            f"| Score: {item.score}/10"
        )
        print(f"     Date   : {item.email_date or 'unknown'}")
        print(f"     Action : {item.required_action or '(none recorded)'}")
    print("Read-only view: no notifications were sent and no records were changed.")
    return 0


def run_agent(settings: Settings) -> int:
    """Start the polling agent (implemented in later phases)."""

    logger = setup_logging(settings.log_dir, settings.log_level)
    logger.info("Starting AI Email Guardian %s", __version__)
    print("The agent is not implemented yet. Phases 4-5 add storage and alerts.")
    return 2


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-email-guardian",
        description="Local Gmail monitor with Ollama summaries and Telegram alerts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="verify the local setup")
    doctor.add_argument(
        "--online",
        action="store_true",
        help="also contact Gmail to verify each configured account",
    )

    subparsers.add_parser("run", help="start the polling agent (later phases)")

    gmail = subparsers.add_parser("gmail", help="manage Gmail accounts")
    gmail_commands = gmail.add_subparsers(dest="gmail_command", required=True)

    auth = gmail_commands.add_parser("auth", help="authenticate one Gmail account")
    auth.add_argument("account", help="account alias from GMAIL_ACCOUNTS")
    auth.add_argument(
        "--force",
        action="store_true",
        help="ignore an existing token and run the OAuth flow again",
    )

    test = gmail_commands.add_parser("test", help="verify one account and list recent mail")
    test.add_argument("account", help="account alias from GMAIL_ACCOUNTS")
    test.add_argument(
        "--limit",
        type=_non_negative_int,
        default=5,
        help="number of recent messages to fetch (default: 5)",
    )

    gmail_commands.add_parser("accounts", help="list configured accounts and token status")

    ai = subparsers.add_parser("ai", help="local AI email analysis")
    ai_commands = ai.add_subparsers(dest="ai_command", required=True)

    ai_test = ai_commands.add_parser(
        "test", help="analyze a few recent emails once (no polling)"
    )
    ai_test.add_argument("account", help="account alias from GMAIL_ACCOUNTS")
    ai_test.add_argument(
        "--limit",
        type=_non_negative_int,
        default=3,
        help="number of recent messages to analyze (default: 3)",
    )

    process = subparsers.add_parser(
        "process", help="run one end-to-end processing cycle"
    )
    process_commands = process.add_subparsers(
        dest="process_command", required=True
    )
    once = process_commands.add_parser(
        "once", help="process both accounts once and exit"
    )
    once.add_argument(
        "--dry-run",
        action="store_true",
        help="analyze and store results but deliver no Telegram notifications",
    )
    process_commands.add_parser(
        "watch", help="monitor both accounts continuously until Ctrl+C"
    )

    notifications = subparsers.add_parser(
        "notifications", help="inspect the notification queue (read-only)"
    )
    notification_commands = notifications.add_subparsers(
        dest="notification_command", required=True
    )
    notification_commands.add_parser(
        "pending", help="list queued alerts without sending or changing them"
    )
    suppress = notification_commands.add_parser(
        "suppress", help="suppress one pending alert without sending anything"
    )
    suppress.add_argument("account", help="Gmail account alias")
    suppress.add_argument(
        "message_id", help="Gmail message ID shown by `notifications pending`"
    )
    unsuppress = notification_commands.add_parser(
        "unsuppress", help="restore a suppressed alert to the pending queue"
    )
    unsuppress.add_argument("account", help="Gmail account alias")
    unsuppress.add_argument("message_id", help="Gmail message ID to restore")

    telegram = subparsers.add_parser("telegram", help="verify Telegram delivery")
    telegram_commands = telegram.add_subparsers(
        dest="telegram_command", required=True
    )
    telegram_commands.add_parser(
        "test",
        help="verify the bot with getMe and send a confirmed test notification",
    )
    chats = telegram_commands.add_parser(
        "chats", help="list recent chat ids without showing the bot token"
    )
    chats.add_argument(
        "--limit",
        type=_non_negative_int,
        default=20,
        help="number of recent updates to inspect (1-100, default: 20)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command in (None, "doctor"):
            return run_doctor(settings, online=getattr(args, "online", False))
        if args.command == "gmail":
            return run_gmail(settings, args)
        if args.command == "ai":
            return run_ai(settings, args)
        if args.command == "telegram":
            return run_telegram(settings, args)
        if args.command == "process":
            return run_process(settings, args)
        if args.command == "notifications":
            return run_notifications(settings, args)
        return run_agent(settings)
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
