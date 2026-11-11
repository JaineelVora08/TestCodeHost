# Phase 2: local terminal chat

Python 3.10 or newer, standard library only. No pip installs are needed.
Uses the existing local Ollama service and exact model `qwen3.5:9b-q4_K_M`.
The Phase 1 drivers, Ollama installation, service configuration, model files
and `../RESULTS.md` are unchanged.

## Launch

From any directory:

```bash
/usr/bin/python3 /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/chat.py
```

The latest conversation resumes automatically. The startup banner gives its ID.
Thinking is **always enabled** through `think: true` on every `/api/chat`
request. The terminal shows `Thinking...`, then streams final-answer text
without reformatting Python code or equations. Both the thinking and final
answer are saved locally. Generated code is never executed by this app.

If Ollama is unavailable, the question is retained and the error is displayed.
The app does not install, download, start, or reconfigure any system software.

## Commands and examples

```text
:new Gradient descent assignment
:multi
Minimize f(x) = (x - 3)^2 using manual gradient descent.
Start at x = 0, use learning rate 0.1 and at most 100 iterations.
Stop when the absolute gradient is below 1e-6.
Generate simple NumPy code and a convergence plot.
:end
```

Normal single-line input works too. Follow-up questions share the selected
chat's recent context. Within multiline input, everything except the exact
line `:end` is literal text, including blank lines and indentation. Ctrl+C
cancels input; during generation it stops and saves the partial response.
Ctrl+D or `:exit` exits; messages already received have been committed.

```text
:chats
:load 1
:history
:stats
:save /tmp/gradient-descent.txt
:save /tmp/gradient-descent.py
:exit
```

`:save` refuses to overwrite any existing file. `.py` exports only code inside
Python/py Markdown fences; multiple blocks are joined in order. Other suffixes
save the final response verbatim. Exporting does not check or run the code;
separate notebook cells may still need editing. `:help` lists all commands.

## Editable instructions

Edit `system_prompt.txt`. It contains the complete requested optimization
assistant instructions and is reread before **every** model request, including
requests in resumed conversations. No app restart is needed. Changes affect
future requests; each saved response also records the exact prompt/context
that was sent, for later analysis. Missing/empty prompt files cause a visible
error instead of sending a request without instructions.

## Persistent history and context budget

`data/chats.sqlite3` contains conversations, timestamped messages, thinking
traces, response status, request context, timing/token metrics, CSV paths and
dataset metadata. The last selected conversation ID is also saved. Use
`:new`, `:chats`, and `:load <id>` to manage independent conversations.
`:history` displays questions and final answers; thinking traces are stored
in SQLite, separate from final answers. Nothing in older history is deleted
when it leaves the model's context.

Default `num_ctx` is **8192**. The app reserves **4096 tokens for thinking plus
the answer**, and 256 for overhead/estimation margin. Input is budgeted with a
conservative UTF-8-byte heuristic (roughly two bytes per token). This is not
the exact Qwen tokenizer. The request always contains the editable system
prompt, compact attachment metadata, and the entire current question. Recent
complete question/answer pairs are included as a contiguous suffix while they
fit; at most the latest 40 pairs are considered. Relevance is based on recency,
not semantic search. Old thinking traces are not resent.

The app reports how many prior turns fit. CSV sample counts shrink from three
to one to zero if necessary; schemas retain exact names. If mandatory input
still cannot fit, it is saved but **not sent or silently cut**. Shorten the
question/prompt or use a new chat with fewer CSV attachments. Incomplete,
interrupted, errored, empty, and output-truncated replies remain saved but are
excluded from later model context.

The estimate cannot guarantee exact token fit for every language or unusual
code. Ollama may impose its own context handling. Long thinking can consume
the entire output reserve, leaving no final answer; this is reported explicitly.
No automatic retry disables thinking or increases context. The 8192 context is
not yet stress tested with long assignments; it can use more RAM/VRAM than
4096. If the system becomes slow or memory constrained, use the lower settings
shown below.

Options, without editing the service/model configuration:

```bash
/usr/bin/python3 /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/chat.py \
  --db /path/to/my-chats.sqlite3 \
  --prompt /path/to/instructions.txt \
  --num-ctx 8192 --response-tokens 4096
```

Context is configurable up to 8192 in this version. The app sends the chosen
context to Ollama with each request; the service's 4096 default does not limit
these explicit per-request settings. If needed, run with
`--num-ctx 4096 --response-tokens 2048`. Reducing the output reserve allows
more input, but leaves less room for thinking and the answer. The exact model
and thinking requirement cannot be switched off through application flags.
`--host` supports a loopback HTTP Ollama endpoint.

## CSV attachments

```text
:csv --header /home/jaineel/datasets/housing.csv
:files
:multi
Use area and bedrooms to predict price.
Implement the gradient descent updates manually; do not use sklearn fitting.
Generate code that loads the complete attached CSV.
:end
```

Choose the header explicitly so a data record is never silently used as column
names. `--header` means the first record contains names; `--no-header` keeps
the first record as data and assigns `column_1`, `column_2`, etc. Even numeric
headers are supported when explicitly selected. Header presence cannot be
reliably guessed for arbitrary categorical datasets. Bare `:csv <path>` gives
instructions instead of attaching with a guess.

```text
:csv --no-header /home/jaineel/datasets/two-column-data.csv
:csv --header --numeric /home/jaineel/datasets/numbers.csv
:csv --header --delimiter semicolon /home/jaineel/datasets/table.csv
```

`--numeric` is optional: it requires every cell in every column to contain a
finite number, rejecting blanks, text, booleans and NaN/Inf with the exact
column, data-row number and physical line. Omit it for categorical/mixed
datasets or intentional placeholders. Delimiter detection handles comma,
semicolon, tab and pipe; use `--delimiter comma|semicolon|tab|pipe` to override.
All options go before the path. Paths may contain spaces; do not add shell
quotes inside the app.

Attachments belong to the selected conversation and persist across restarts.
Reattaching the same absolute path updates its parsing choices. Switching to a
new conversation gives an empty attachment list. Attachments saved by the old
version must be reattached with a header choice before another model request;
old chats and saved responses are retained.

The CSV module scans UTF-8/UTF-8-BOM files, infers comma/semicolon/tab/pipe
delimiters, and reports row/column counts, exact column names, basic inferred
types, blank/whitespace missing counts, numeric min/max, and up to three
deterministically sampled rows. Each column also has numeric/non-numeric counts
and up to three non-numeric examples and missing locations, recorded with data
row numbers and physical line numbers. Mixed columns get a visible warning.
These diagnostics survive context-driven sample removal, so a rare bad value
does not disappear merely because it was absent from the random sample.
Missing markers such as `NA` remain literal strings, not automatically missing
values. Physically blank records count as rows of missing cells. Counts and
ranges cover the whole file. Types are heuristic; samples are illustrative.
Sample/diagnostic cells longer than 160 characters are marked as truncated.

Only this bounded metadata enters context. The model receives the absolute
path and exact `pd.read_csv` options: header mode, synthetic names when needed,
delimiter, encoding, string dtype, quoting rules, and preservation of blank
records/literal NA strings. It is instructed to load the **full CSV**, and to
handle missing/non-numeric cells only as explicitly requested. It must not
silently drop records, coerce invalid cells to NaN, invent replacements, or
claim execution. `--numeric` supplies an application-level numeric check before
inference; without it, the model must validate the columns selected for numeric
work. The application does not compute fitted solutions or execute generated
code. CSV handling cannot guarantee the correctness of model-generated code.
Duplicate/empty headers, ragged or malformed records, empty datasets, NUL bytes,
non-UTF-8 and unreadable files are rejected with an error.
Limits: 64 MiB, 100 columns, 256 characters per header, and Python csv's default
field-length limit (normally 128 KiB). PDF/Excel/notebook uploads are unsupported.
The file is rescanned with its saved header/delimiter choices before every
model request; updated metadata is saved, and strict numeric requirements are
checked again. A file changing during inspection blocks the request. Full-file
SHA-256 fingerprints are saved with the metadata. Generated code still refers
to the local path: it must exist, and later file changes may affect results
when that code is eventually run.

## Resource statistics

`:stats` reads `/proc/meminfo`, queries `nvidia-smi` if available, and calls
Ollama `/api/ps`. It reports RAM used/available, swap, GPU VRAM/utilization,
current model placement estimate and loaded context, and the selected chat's
last saved response time, generated token count and tokens/second.
Missing tools/server metrics display as unavailable. No psutil is needed.
Metrics are on demand, not a live monitor or peak-memory recorder. Placement
percentages describe memory allocation, not GPU utilization. Generated-token
metrics cover thinking plus answer; total wall time also includes prompt/load
time. Last response metrics persist with the conversation.

## Tests

Offline tests (no model calls):

```bash
/usr/bin/python3 -m unittest discover \
  -s /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/tests -v
```

Optional real-server smoke test (two short inference requests in separate
CLI processes, 8192 context and thinking enabled; not rerun since the CSV update):

```bash
/usr/bin/python3 /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/tests/smoke_test.py
```

It uses the explicitly synthetic `tests/fixtures/sample.csv` and creates a
separate timestamped database, terminal logs and JSON report under `data/`.
It verifies CSV propagation, multiline input, emitted thinking, final-answer
streaming, a remembered coefficient/column after restart, and resource stats.
It does **not** validate mathematical accuracy or generated code correctness.
See `TEST_RESULTS.md` for the completed validation.

Official API references:

- https://docs.ollama.com/api/chat
- https://docs.ollama.com/capabilities/thinking
- https://docs.ollama.com/api/ps
