# AI Email Guardian

A free, local-first AI agent that watches your Gmail inbox, decides which emails
matter, summarizes them with a local LLM, and pings you on Telegram.

No paid APIs. No cloud AI. Only the read-only Gmail scope is requested: the app
never sends, deletes, modifies or marks messages as read.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Project skeleton, configuration, logging, setup checks | Done |
| 2 | Gmail OAuth 2.0 (multi-account), profiles, inbox metadata | Done |
| 3 | Ollama importance scoring and summaries | Done |
| 4 | SQLite history and duplicate protection | Planned |
| 5 | Telegram alerts and the 60-second polling loop | Planned |

## Stack

- **Python 3.10+** application code
- **Gmail API** with OAuth 2.0 Desktop Application flow (free)
- **Ollama** for local inference (free, offline)
- **SQLite** for local history
- **Telegram Bot API** for notifications (free)

## Requirements

- Windows 11, PowerShell
- Python 3.10 or newer (`py --version`)
- [Ollama](https://ollama.com/download) installed and running
- A Google Cloud project with the Gmail API enabled
- A Telegram bot created with [@BotFather](https://t.me/BotFather)

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env   # only if .env does not exist yet
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Google Cloud setup (Gmail API)

Do this once; the same OAuth client serves every account.

1. Open <https://console.cloud.google.com/> and create or select a project.
2. **APIs & Services → Library**, search for **Gmail API**, open it, click **Enable**.
3. **APIs & Services → OAuth consent screen**:
   - User type: **External** (or **Internal** for Google Workspace).
   - Fill in app name, user support email and developer contact email.
   - **Scopes → Add or remove scopes**: add
     `https://www.googleapis.com/auth/gmail.readonly`.
     Do not add any other Gmail scope.
   - **Test users**: add every Gmail address you will monitor while the app
     status is **Testing**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**.
   - Application type: **Desktop app**.
   - Name it (e.g. `AI Email Guardian`) and click **Create**.
   - Click **Download JSON**.
5. Save the downloaded file as `credentials.json` in the project root
   (next to `main.py`). Never commit it; it is already gitignored.
6. Authenticate each configured account separately:

   ```powershell
   python main.py gmail auth personal
   python main.py gmail auth work
   ```

   A browser window opens; sign in to the Gmail account for that alias. The
   account chooser is always shown so you can pick a different account for the
   second alias. Each token is written to `data/tokens/<alias>.json`.
7. Verify:

   ```powershell
   python main.py gmail accounts
   python main.py gmail test personal --limit 5
   python main.py doctor --online
   ```

Notes:

- While the OAuth consent screen is in **Testing**, Google expires refresh
  tokens after 7 days. Re-run `gmail auth <alias>` weekly, or publish the app
  (Google verification may be required for external users).
- Tokens are per account; deleting `data/tokens/<alias>.json` only revokes the
  local copy. Revoke access from your Google Account → Security → Third-party
  apps at any time.

## Usage

```powershell
python main.py doctor                  # offline setup report
python main.py doctor --online         # also query Gmail profile for each account
python main.py gmail accounts          # configured accounts and token state
python main.py gmail auth <alias>      # OAuth desktop flow for one account
python main.py gmail auth <alias> --force
python main.py gmail test <alias> --limit 5
python main.py ai test <alias> --limit 3   # fetch + analyze recent mail once
python main.py process once            # one full cycle for all accounts, then exit
python main.py process once --dry-run  # analyze/store, deliver no notifications
python main.py process watch            # monitor both accounts until Ctrl+C
python main.py notifications pending   # read-only view of the pending alert queue
python main.py notifications suppress <account> <message-id>
python main.py notifications unsuppress <account> <message-id>
python main.py run                     # legacy stub; monitoring uses `process watch`
python -W error::ResourceWarning -m unittest discover -s tests -v
```

`gmail test` prints sender, subject and date for the newest messages and never
downloads bodies (`format=metadata`). `ai test` fetches the body of each selected
message (`format=full`), then asks the local Ollama model for a structured
classification. It analyzes only a handful of messages and exits; continuous
polling arrives in Phase 5.

## AI analysis (Phase 3)

`python main.py ai test personal --limit 3` prints one block per message:

```
  [URGENT] 9/10 (confidence 85%)  [meets threshold]
      2026-11-24 09:12 | Alice <alice@example.com> | Server outage notice
      Summary : Production is down and needs attention today.
      Reason  : The sender reports an active outage and asks for a reply.
      Action  : Check the dashboard and reply to Alice.
      Deadline: 2026-11-24
```

Every analysis contains a category (`URGENT`, `IMPORTANT`, `NORMAL`,
`PROMOTIONAL`, `SPAM`), an importance score (0-10), a short summary, the
classification reason, the required action, an optional deadline and a
confidence value. `IMPORTANCE_THRESHOLD` decides which results are flagged as
meeting the bar; notifications use the same threshold in Phase 5.

Genuine job-related mail (vacancies, job alerts, recruiter messages, interview
invitations, application updates, assessments and offers) is always raised to
`IMPORTANT` with a score at least equal to `IMPORTANCE_THRESHOLD`, even when the
model would otherwise call it promotional. Ordinary product advertising is not
treated as job mail just because it comes from a recruitment site.

The prompt requires evidence before a `SPAM` verdict: a link alone is not proof
of phishing, and the analyzer never claims a link was opened or a website
verified. When the message lacks enough evidence, the result says authenticity
cannot be determined and suggests visiting the official site directly.

The model runs entirely on this machine through Ollama. Requests ask Ollama for
schema-constrained JSON (falling back to plain JSON mode on older versions).
The response must be a single JSON object; malformed or incomplete output gets
one controlled correction attempt, then a clear error instead of being
trusted. Missing `required_action` or `deadline` values become `None`; nothing
is invented. Read timeouts are not retried (that would duplicate inference),
and each request's duration is logged. Missing Ollama/model errors include the
exact command to run.

Bodies are extracted from MIME messages: `text/plain` is preferred, HTML is
converted to text as a fallback, and attachment parts are skipped and never
downloaded. Very large bodies are truncated for analysis only.

## SQLite persistence (Phase 4)

`data/emails.db` is created on first use and checked by `python main.py doctor`.
The doctor closes its database connection after checking it, and its final line
shows the current status of each configured Gmail account. Missing Telegram
credentials are reported as expected until Phase 5 and never as database errors.
The schema is versioned with `PRAGMA user_version` plus a `schema_migrations`
table, so existing data is upgraded in order on restart. Connections use WAL
mode, a 5-second busy timeout, foreign keys and explicit transactions; cleanup
checkpoints WAL and closes handles, which matters on Windows.

Tables cover accounts, processed emails, AI classifications, notification
delivery, account-level sync checkpoints and applied migrations. A unique
constraint on `(account_alias, gmail_message_id)` prevents the same Gmail
message being stored twice for one account; the same message may exist under
different aliases. Duplicate inserts are no-ops and never overwrite the first
classification.

Notification status starts as `NOT_REQUIRED` or `PENDING`, then moves through
`FAILED` and/or `SENT`. `SENT` is only written by an explicit confirmation call
after a provider accepts delivery; restarts and repeated calls cannot create a
second alert. Delivery attempts and timestamps are retained. Telegram delivery
is used by `python main.py process once` and is confirmed by the Bot API reply.

Only metadata and analysis output are stored. Bodies, attachments, snippets,
OAuth tokens and Google credentials are never written to SQLite. The database,
`-wal` and `-shm` files are gitignored. **SQLite files are not encrypted by
default**; protect the `data/` directory with operating-system permissions and
encrypt any backups that leave this machine.

## End-to-end processing

`python main.py process once` runs one cycle for every configured account and
exits: fetch message metadata, skip already processed ids, fetch each new body,
classify it with Ollama, store metadata and analysis in SQLite, then deliver
notifications. Only `URGENT` mail and `IMPORTANT` mail scoring at or above
`IMPORTANCE_THRESHOLD` is notified; every genuine job-related email is raised to
`IMPORTANT` by the analyzer. Notification status is written as `PENDING` before
delivery and becomes `SENT` only when Telegram returns a confirmed message id,
or `FAILED` when delivery is rejected. Pending alerts are retried at the start
of the next cycle. Per-account sync checkpoints are preserved, and bodies,
snippets, credentials and tokens are never written to SQLite or logs.

`--dry-run` performs the fetch, analysis and storage but delivers nothing; the
notifications remain `PENDING` and are sent by the next normal cycle. Continuous
polling is not started by this command.

`python main.py notifications pending` is a read-only inspection of that queue:
it prints the total plus each pending alert's account alias, sender, subject,
AI classification, score, email date and required action. It sends nothing and
does not change statuses, checkpoints or records. Bodies, snippets, tokens and
Telegram credentials are never displayed.

`python main.py notifications suppress <account> <message-id>` moves one queued
alert into the existing terminal `NOT_REQUIRED` status. It changes only the
notification status row: the email, its classification and the account
checkpoint are untouched, no Telegram message is sent, and the alert is never
retried. The change persists across restarts and is repeatable (no duplicate
rows). `python main.py notifications unsuppress <account> <message-id>` restores
an eligible alert (`URGENT`, or `IMPORTANT` at or above the threshold) to
`PENDING` so the next normal cycle can deliver it. Suppression is never applied
automatically; an already delivered alert cannot be suppressed.

## Telegram delivery

`python main.py telegram test` verifies the bot with `getMe`, reads
`TELEGRAM_CHAT_ID`, sends a plain-text test notification with `sendMessage` and
reports success only after Telegram returns the message id and confirms the chat.
A missing chat id stops the command with a clear error.

`python main.py telegram chats --limit 20` uses `getUpdates` to list recent chat
ids and titles, so `TELEGRAM_CHAT_ID` can be copied from the output. Only the
chat id and its display name are printed; the bot token is never printed,
logged or stored in SQLite. Request and API errors replace any token text with
`<redacted>` before they are shown or logged.

Alerts are sent as plain text with metadata and analysis output only; email
bodies and attachments are never included. Delivery is reported as successful
only when the Bot API confirms the send. Each alert shows the receiving account
(`Personal` or `Work`) and carries an inline "Open in Gmail" button. The link
uses `/u/0/?authuser=<account address>` to select the correct signed-in Gmail
account and searches `rfc822msgid:<Message-ID>` (angle brackets removed) so it
opens the exact email. If no Message-ID is available, the button is omitted.
The address in `authuser` always comes from the authenticated Gmail account that
owns the alias — never from the sender, subject, AI output, a configured guess
or a stored queue value — so a Personal alert cannot link into the Work mailbox
(or vice versa), including when a pending alert is retried after a restart.

## Continuous monitoring

`python main.py process watch` runs the same cycle every 60 seconds
(`POLL_INTERVAL_SECONDS`) for both configured accounts until Ctrl+C. It reuses
the one-shot pipeline: new mail is analyzed and notified, duplicates are
skipped, pending alerts from earlier runs are retried, and a failure in one
account (or a temporary Gmail/Ollama/Telegram error) is logged and retried on
the next cycle instead of stopping the monitor. Gmail stays read-only; no
messages are marked read, deleted or modified. Stop cleanly with Ctrl+C.

Continuous monitoring is local only — no cloud hosting is implemented.

## Configuration

All settings live in `.env`, which is ignored by git. `.env.example` documents
every key with safe defaults.

| Key | Default | Purpose |
|-----|---------|---------|
| `GMAIL_ACCOUNTS` | empty | Comma-separated aliases of the **only** accounts monitored |
| `GMAIL_CREDENTIALS_FILE` | `credentials.json` | OAuth client file shared by all accounts |
| `GMAIL_TOKEN_DIR` | `data/tokens` | Per-account token files (gitignored) |
| `GMAIL_QUERY` | `in:inbox newer_than:1d` | Default Gmail search; read state is ignored |
| `GMAIL_QUERY_<ALIAS>` | empty | Optional per-account query override |
| `GMAIL_ACCOUNT_<ALIAS>_EMAIL` | empty | Optional expected address; mismatches abort |
| `GMAIL_FETCH_LIMIT` | `20` | Max messages per poll (1-500) |
| `POLL_INTERVAL_SECONDS` | `60` | Seconds between polls (minimum 10) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model used for analysis |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Per-request timeout; the first call may load the model |
| `TELEGRAM_BOT_TOKEN` | empty | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | empty | Your chat id |
| `DATABASE_PATH` | `data/emails.db` | SQLite file location |
| `LOG_DIR` | `logs` | Log directory |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `IMPORTANCE_THRESHOLD` | `7` | Minimum score (0-10) that triggers an alert |

`<ALIAS>` is the uppercase alias with `-` replaced by `_`, for example
`GMAIL_QUERY_WORK_MAIN` for the alias `work-main`. Relative paths are resolved
against the project root.

The default query intentionally avoids `is:unread`: a message you open before
the next poll is still fetched, because selection is based on arrival time.

## Project layout

```
AI-Email-Guardian/
├── main.py                     Entry point used by every command
├── requirements.txt            Runtime dependencies
├── .env.example                Documented template for local settings
├── .gitignore                  Keeps secrets, data and logs out of git
├── README.md                   This document
├── email_guardian/             Application package
│   ├── __init__.py             Package metadata and version
│   ├── cli.py                  Commands: doctor, run, gmail auth/test/accounts
│   ├── config.py               .env loading, validation, per-account paths
│   ├── logging_setup.py        Console + rotating file logging with cleanup
│   ├── gmail_client.py         OAuth, profile, metadata and MIME body fetch
│   ├── summarizer.py           Ollama JSON classification, validation, prompts
│   ├── storage.py              SQLite schema, migrations and persistence
│   ├── notifier.py             Telegram delivery, getMe and chat discovery
│   └── agent.py                Polling orchestration (Phase 5)
├── tests/                      unittest suite with mocked Gmail and Ollama
├── data/                       SQLite DB (gitignored)
│   └── tokens/                 Per-account OAuth tokens (gitignored)
└── logs/                       Rotating application logs (gitignored)
```

## Security notes

- Only `gmail.readonly` is requested. Tokens carrying broader Gmail scopes are
  rejected and never used.
- Tokens are stored under `data/tokens/`, which is gitignored; on POSIX they are
  written with `0600` permissions.
- `credentials.json`, `.env`, `data/tokens/` and `data/*.db` (plus `-wal`/`-shm`
  sidecars) are all ignored.
- Logs record account aliases, message counts and message IDs only. Subjects,
  senders, snippets, bodies, prompts, summaries and credentials are never logged.
- Email text is treated as untrusted data: prompts explicitly tell the model to
  ignore instructions inside emails, and the text is wrapped in per-request
  random delimiters. No links are opened and no code is executed.
- Email content is sent only to the locally configured Ollama endpoint.
- API errors report HTTP status and reason code, not response bodies.
- No code path sends, deletes, modifies or marks mail as read.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` opens the Microsoft Store | Use `py` or install Python from python.org with "Add to PATH" |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `OAuth client file not found` | Download the Desktop app client JSON as `credentials.json` |
| `No OAuth token stored` | Run `python main.py gmail auth <alias>` |
| Token expired/revoked after 7 days | Testing-mode tokens expire; re-run `gmail auth <alias>` |
| `access blocked` / `app not verified` | Add your address under OAuth consent screen → Test users |
| Wrong Google account signed in | Re-run with `--force` and pick the other account in the chooser |
| `token grants broader access` | Delete `data/tokens/<alias>.json` and re-authenticate |
| `Ollama is not reachable` | Start it with `ollama serve` (or launch the Ollama app) |
| `Model 'llama3.2' is not installed` | `ollama pull llama3.2` |
| `Ollama did not finish within ...s` | The first call may load the model; set `OLLAMA_TIMEOUT_SECONDS=300` (or higher) in `.env` |
| `Ollama returned an invalid classification twice` | The model output stayed malformed; re-run and check `ollama --version` |

## Windows verification (Phase 4)

Open PowerShell in the project folder and run:

```powershell
py -3 -m unittest discover -s tests -v
py -3 main.py doctor
Get-ChildItem data\emails.db*
```

Expected: all tests pass, `doctor` reports `Database schema ... schema version 2,
WAL enabled`, and the database appears under `data\` alongside any `-wal`/`-shm`
sidecars. Those files are gitignored; never commit them.
