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

Default `num_ctx` is **4096**. The app reserves **2048 tokens for thinking plus
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
No automatic retry disables thinking or increases context. A short smoke test
does not establish that long optimization assignments fit this budget.

Options, without editing the service/model configuration:

```bash
/usr/bin/python3 /home/jaineel/TestingLLMLocal/qwen-local-setup/chat/chat.py \
  --db /path/to/my-chats.sqlite3 \
  --prompt /path/to/instructions.txt \
  --num-ctx 4096 --response-tokens 2048
```

Context is configurable up to 4096 in this version. **8192 is rejected** until
a separate memory/performance test and deliberate code change are performed.
Reducing the output reserve allows more input, but less room for thinking and
the answer. The exact model and thinking requirement cannot be switched off
through application flags. `--host` supports a loopback HTTP Ollama endpoint.

## CSV attachments

```text
:csv /home/jaineel/datasets/housing.csv
:files
:multi
Use area and bedrooms to predict price.
Implement the gradient descent updates manually; do not use sklearn fitting.
Generate code that loads the complete attached CSV.
:end
```

Paths may contain spaces; do not add shell quotes inside the app. Attachments
belong to the selected conversation and persist across restarts. Reattaching
the same absolute path refreshes that attachment's metadata. Switching to a
new conversation gives an empty attachment list.

The CSV module scans UTF-8/UTF-8-BOM files, infers comma/semicolon/tab/pipe
delimiters, and reports row/column counts, exact column names, basic inferred
types, blank/whitespace missing counts, numeric min/max, and up to three
deterministically sampled rows. Missing markers such as `NA` remain literal
strings. Types are heuristic; the sample is illustrative, not a guarantee of
all categories or outliers. Sample cells longer than 160 characters are marked
as truncated. Counts and ranges come from scanning the file, not the sample.

Only this bounded metadata enters context. The model receives the absolute
path, encoding and delimiter, and instructions to generate code loading the
**full CSV** for actual computations. The application does not compute fitted
solutions or execute the generated code. A header row is required: the first
record is treated as column names; headerless files cannot be reliably detected.
Duplicate/empty headers, ragged or malformed records, non-UTF-8 and unreadable
files are rejected with an error.
Limits: 64 MiB, 100 columns, 256 characters per header, and Python csv's default
field-length limit (normally 128 KiB). PDF/Excel/notebook uploads are unsupported.
Metadata is a snapshot: reattach a file after changing it. Persisted paths must
still exist where the generated notebook code runs.

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
CLI processes, default 4096 context and thinking enabled):

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
