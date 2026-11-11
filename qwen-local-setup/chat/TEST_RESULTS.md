# Phase 2 validation

Implementation is complete. No additional packages were installed.

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

## Limits

Context estimates are heuristic, not exact tokenizer counts. Only recent
complete turns that fit are resent; old content remains in SQLite. Long
thinking or code may reach the output limit; this is reported and saved.
CSV metadata is an attachment-time snapshot, with up to three sample rows;
the full dataset is never sent. CSV inspection is limited to UTF-8 files up
to 64 MiB and 100 columns. Mathematical answers still require checking.
