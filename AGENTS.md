# AGENTS.md — Climate Discourse Analyzer

Guidance for AI agents (Claude Code, Copilot, Cursor, etc.) working on this codebase.

---

## Project Overview

**Climate Discourse Analyzer** is an automated NLP pipeline for classifying social media posts on environmental and climate topics. It performs 10 types of analysis — from relevance detection and geographic attribution to emotion classification and topic labelling — using a three-tier approach: regex patterns → SQLite database lookups → OpenAI GPT API.

**Stack:** FastAPI (backend) · Streamlit (frontend) · OpenAI API · SQLite · pandas · openpyxl

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env           # fill in OPENAI_API_KEY
python init_db.py              # build the local sample DB (sentiment_v2.db) — not shipped

# Terminal 1 — backend
uvicorn app:app --port 8000

# Terminal 2 — frontend
streamlit run index.py
```

---

## Repository Layout

```
app.py              FastAPI server — analysis orchestration, REST endpoints
LLM.py              All LLM logic — GPT calls, regex patterns, DB lookups, taxonomies
index.py            Streamlit UI — file upload, parameter toggles, status polling
handle.py           Excel I/O, column normalisation, colour-cell highlighting
gen_func.py         File logger (log / clean_log)
sentiment_v2.db     SQLite — author/place lookup tables (NOT in repo; built locally by init_db.py)
init_db.py          (Re)builds sentiment_v2.db with the sample lookup data
.env.example        Environment variable template (copy → .env, never commit .env)
AGENTS.md           This file
```

---

## Architecture

```
Streamlit (index.py)
    │  POST /files  ·  GET /status  ·  POST /stop
    ▼
FastAPI (app.py)
    │  BackgroundTasks → analyze(df, codes)
    ├─ Pattern check  →  check_patterns_*()       [LLM.py]
    ├─ DB lookup      →  check_*_in_db_*()        [LLM.py + sentiment_v2.db]
    └─ GPT call       →  define_*() / gpt_chat()  [LLM.py → OpenAI API]
    │
    ▼  writes
Result.xlsx  (colour-coded output, available for download)
```

### Analysis pipeline per row

For each DataFrame row the pipeline tries in order:
1. **Regex patterns** (`check_patterns_*`) — instant, free
2. **SQLite lookup** (`check_*_in_db_*`) — fast, free
3. **GPT call** (`define_*`) — slow, costs tokens

Results are written back to the DataFrame with a `_mask` colour column:
`green` = pattern hit · `gray` = DB hit · `blue` = GPT hit

---

## 10 Analysis Codes

| Code | Key | What it produces |
|------|-----|-----------------|
| 1 | `relevance` | `1` / `0` — is the post about climate/environment? |
| 2 | `country_pattern` | Country name — regex on author/group name |
| 3 | `country_bd` | Country name — SQLite lookup |
| 4 | `country_AI` | Country name — GPT |
| 5 | `region_bd` | Region name — SQLite lookup |
| 6 | `region_patterns` | Region name — regex |
| 7 | `emotions` | One of 18 emotion labels (localized) |
| 8 | `sentiments` | `positive` / `negative` / `unknown` |
| 9 | `messages` | Topic label from the climate topic taxonomy |
| 10 | `officials` | `0` or name of institution/position |

Files for codes 1–6 are uploaded via the first Streamlit file uploader; codes 7–10 via the second uploader.

---

## DataFrame Column Contract

Column names come from the source social-media monitoring export and **must not be renamed** — they are read and written by every analysis block:

| Column | Type | Description |
|--------|------|-------------|
| `Текст` | str | Post body text |
| `Заголовок` | str | Post headline |
| `Автор` | str | Author handle / name |
| `Місце публікації` | str | Group / page / channel |
| `Мова` | str | Detected language code |
| `Країна` | str | Author country (filled by codes 2–4) |
| `Регіон` | str | Author region (filled by codes 5–6) |
| `Емоції` | str | Emotion (filled by code 7) |
| `Настрій` | str | Sentiment (filled by code 8) |
| `Тема` | str | Topic label (filled by code 9) |
| `Офіційність` | str | `0` / `1` (filled by code 10) |
| `джерело` | str | Institution/position (filled by code 10) |
| `uMessage` | str | Raw GPT response when no taxonomy match |

---

## Environment Variables

Defined in `.env` (never commit):

```
OPENAI_API_KEY=...            # required

# Fine-tuned model IDs — if empty, falls back to base gpt-4o-mini
MODEL_RELEVANCE=
MODEL_ECO_MESSAGE=
MODEL_OFFICIALS=
MODEL_EMOTIONS=
MODEL_REGION_CHECK=
```

Both `LLM.py` and `app.py` call `load_dotenv()` and read these via `os.getenv()`.

---

## Key Invariants

### Stop mechanism
Every row-level loop **must** check `state['if_stop']` at the top:

```python
if state['if_stop']:
    state['status'] = 'stopped'
    state['if_stop'] = None
    return
```

### Status lifecycle
`done` → `busy` (on POST /files) → `done` | `error` | `stopped`

### Shared state dict (`app.state.state`)
Written by the background task, read by `/status`. Keys:
`status`, `details`, `if_stop`, `relevance`, `country`, `emotions`, `sentiments`, `messages`, `officials`

### Dual system_messages
`system_messages` and `gpt_fine_tunning_models` dicts appear in **both** `LLM.py` and `app.py`.  
**Always keep them in sync** when editing prompts or model keys.

---

## Topic Taxonomy

### Climate topic taxonomy (`eco_titles_list`, ~54 labels)
Topics covering climate and environment — from temperature records and green energy to activism and biodiversity loss.

The list is defined in `LLM.py`. The GPT system prompt for code 9 embeds the full label list inline; update both the `eco_titles_list` variable **and** the `system_messages['message']` entry (in `LLM.py` and `app.py`) when adding/removing labels.

---

## How to Extend

### Add a new analysis type
1. Add a new code → key entry to `ANALYSIS_TYPES` in `app.py`
2. Add a handler `if analysis_key == "new_key":` block inside `analyze()` following the existing pattern (pattern → DB → GPT → write to df)
3. Add the new key to `template_state` in `app.py`
4. Add a toggle to `index.py` and wire it to the correct code in the `option_map`

### Add a regex pattern
Add a `(label, r"pattern")` tuple to the relevant `check_patterns_*()` function in `LLM.py`.

### Add a new topic label
1. Append the label string to `eco_titles_list` in `LLM.py`
2. Update the `system_messages['message']` string in **both** `LLM.py` and `app.py`

### Add a country alias
Add `'foreign_name': 'local_name'` to the `country_aliases` dict inside the `country_bd` block in `app.py` (≈ line 280).

---

## Common Pitfalls

| Pitfall | What happens | Fix |
|---------|-------------|-----|
| Editing `system_messages` in only one file | Prompts diverge between LLM.py and app.py | Always edit both |
| Adding a label with embedded `"` inside a `"…"` string | `SyntaxError: '{' was never closed` | Use `'single-quoted'` labels or escape `\"` |
| Forgetting `if_stop` check in a new loop | Stop button hangs | Copy the check from an existing block |
| Uploading while `status == busy` | Second job silently rejected | Frontend checks `/status` before POST `/files` |
| Missing `.env` | `OpenAI(api_key=None)` → `AuthenticationError` | Copy `.env.example` → `.env`, add key |

---

## Testing

There is no automated test suite. Manual verification steps:

1. Start both servers (uvicorn + streamlit)
2. Upload a small Excel file (10–20 rows) with codes `(1,)` — check `Result.xlsx` for `marker` column
3. Enable all 10 codes and upload a larger file; verify the stop button works mid-run
4. Check `log_fastapi.txt` for per-row trace after each run
5. Confirm `Result.xlsx` colour coding: green = pattern, gray = DB, blue = GPT

---

## Logging

`gen_func.log()` appends timestamped lines to a text file (default `log.txt`, configurable per call).  
`app.py` writes to `log_fastapi.txt`. Logs are plain text, UTF-8, append-only.  
Call `clean_log()` to truncate a log file before a new run if needed.
