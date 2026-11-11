# Phase 2 validation

Implementation is complete. No additional packages were installed.
The original results below used 4096 context. See the CSV handling update at
the end for the current application behavior and offline checks.

## Offline tests

32 tests passed on each installed interpreter:

- `/usr/bin/python3`: Python 3.10.12
- `python3`: Python 3.13.5

Coverage includes normal and multiline input; separate streamed thinking and
answer fields (including a chunk containing both); explicit `think: true`,
exact model and 4096 context in the API payload; system prompt inclusion and
reload; follow-up context; SQLite persistence across a reopened connection;
conversation creation/list/load/isolation; context budgeting and preservation
of omitted history; CSV path/schema/column propagation; row counts, types,
missing values, sampling, numeric ranges including large integers; exact
header whitespace; malformed, missing and unreadable CSVs; missing monitoring
tools/server metrics; failed/interrupted/truncated streams; exports without
overwriting files; and non-execution of generated code.

Run again:

```bash
/usr/bin/python3 -m unittest discover \
  -s /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/tests -v
```

## Real local inference smoke test

Used the existing Ollama 0.40.16 server and `qwen3.5:9b-q4_K_M`, `think: true`,
4096 context and a 2048-token output reserve. Two separate CLI processes used
the same dedicated smoke SQLite database. The normal user database is not
populated with these test chats.

1. Attached the explicitly synthetic two-row `tests/fixtures/sample.csv`
   (columns `area`, `bedrooms`, `price`). Entered a multiline request to remember
   coefficient 17 and target column `price`. The final response was `ACK`;
   174 characters of thinking were emitted and persisted. Total wall time:
   15.14 seconds; 48 generated tokens.
2. Exited and restarted the application. Listed and loaded conversation 1.
   A follow-up returned: `You specified a coefficient of 17 and a target column
   named "price".` The saved prior turn and CSV metadata were present in the
   request. 586 characters of thinking were emitted and persisted. Total wall
   time: 17.13 seconds; 138 generated tokens; 8.63 generated tokens/second
   (thinking plus answer, excluding prompt/load time).
3. `:stats` reported RAM 5.52 GiB used / 15.26 GiB total, 9.74 GiB available,
   swap 0.086/2 GiB, VRAM 2989/4096 MiB, GPU utilization 16%, and Ollama's
   estimated 67% CPU / 33% GPU placement. These are on-demand snapshots,
   **not peak-memory measurements**.

Artifacts, relative to this directory:

- `data/smoke-1794389327160684243.json`: integration summary and metrics
- `data/smoke-1794389327160684243.sqlite3`: full messages, thinking and actual requests
- `data/smoke-1794389327160684243-session-1.log`
- `data/smoke-1794389327160684243-session-2.log`

This verifies application integration, streaming fields, short context recall,
attachments and restart persistence. It does **not** establish mathematical
accuracy, correctness of generated optimization code, or long-assignment
performance. No generated model code was run. No 8192-context test was run.

## Phase 1 preservation

SHA-256 comparisons against pre-implementation snapshots passed for:

- `../RESULTS.md`
- `../install-system.sh`
- `/etc/systemd/system/ollama.service.d/90-qwen-local.conf`

All Phase 2 files are under `chat/`. No GPU drivers, Ollama installation,
existing model files or existing model/service settings were changed.

## Original validation limits

Context estimates are heuristic, not exact tokenizer counts. Only recent
complete turns that fit are resent; old content remains in SQLite. Long
thinking or code may reach the output limit; this is reported and saved.
CSV metadata is an attachment-time snapshot, with up to three sample rows;
the full dataset is never sent. CSV inspection is limited to UTF-8 files up
to 64 MiB and 100 columns. Mathematical answers still require checking.

## CSV handling update — 2026-11-11

The app now requires an explicit `--header` or `--no-header` choice. Headerless
records are retained and assigned stable column names. Exact Pandas loading
options accompany metadata, preserving string cells, NA literals and blank
records. Full-file numeric/non-numeric counts and bounded bad-cell locations
are supplied even when random samples are removed from context. Optional
`--numeric` blocks missing, non-finite and non-numeric data before inference.
Delimiter overrides support ambiguous inputs; malformed non-comma files no
longer silently fall back to a single comma-separated column.

Attachments are rescanned before each question. Legacy attachments require a
new header choice; saved chats are retained. Files changing during inspection
block the request. No original datasets or saved user conversations were edited.

44 offline tests passed on `/usr/bin/python3` (3.10), including a round-trip
check with the already installed Pandas. On `python3` (3.13), 43 passed and the
optional Pandas check was skipped because Pandas is not installed there. No
packages were installed, no model inference was requested, and no generated
model code was executed. Tests cover the parsing options reaching the model,
first-row preservation, anomalies outside random samples, blank/NA handling,
numeric validation, rescan behavior and legacy attachment rejection.

The current defaults are 8192 context and 4096 thinking/answer tokens. The
existing smoke script now explicitly selects 8192/4096 to accommodate longer
custom system prompts; it was updated for the new CSV syntax but was not rerun.
Long 8192-context inference and model-generated mathematical correctness
remain unverified. The full dataset remains local; the model receives bounded
metadata and instructions to generate code that loads the complete file.
