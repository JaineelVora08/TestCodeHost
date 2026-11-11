#!/usr/bin/env python3
"""Local terminal chat. Python 3.10+, standard library only; never executes replies."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request

from csv_support import dataset_context, inspect_csv
from monitoring import format_stats

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.5:9b-q4_K_M"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL, content TEXT NOT NULL, thinking TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'complete', reply_to INTEGER REFERENCES messages(id),
                metrics TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(conversation_id, id);
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                path TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(conversation_id, path));
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def new_chat(self, title="New conversation"):
        with self.db:
            cursor = self.db.execute("INSERT INTO conversations(title,created_at,updated_at) VALUES(?,?,?)", (title, now(), now()))
            chat_id = cursor.lastrowid
            self.select_chat(chat_id)
        return chat_id

    def select_chat(self, chat_id):
        row = self.db.execute("SELECT * FROM conversations WHERE id=?", (chat_id,)).fetchone()
        if row is None:
            raise ValueError(f"Conversation {chat_id} does not exist.")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('last_chat',?)", (str(chat_id),))
        return dict(row)

    def current_chat(self):
        row = self.db.execute("SELECT value FROM settings WHERE key='last_chat'").fetchone()
        return int(row[0]) if row else self.new_chat()

    def chats(self):
        return self.db.execute("SELECT c.*, (SELECT count(*) FROM messages m WHERE m.conversation_id=c.id AND role='user') AS questions FROM conversations c ORDER BY c.id DESC").fetchall()

    def add_message(self, chat_id, role, content, *, reply_to=None, status="complete"):
        with self.db:
            cursor = self.db.execute("INSERT INTO messages(conversation_id,role,content,reply_to,status,created_at) VALUES(?,?,?,?,?,?)", (chat_id, role, content, reply_to, status, now()))
            self.db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), chat_id))
            if role == "user":
                title = " ".join(content.split())[:60]
                self.db.execute("UPDATE conversations SET title=? WHERE id=? AND title='New conversation'", (title, chat_id))
        return cursor.lastrowid

    def update_answer(self, message_id, content, thinking, status, metrics):
        with self.db:
            self.db.execute("UPDATE messages SET content=?,thinking=?,status=?,metrics=? WHERE id=?", (content, thinking, status, json.dumps(metrics), message_id))

    def complete_turns(self, chat_id, limit=40):
        rows = self.db.execute("""SELECT u.id AS user_id, u.content AS question, a.content AS answer
            FROM messages a JOIN messages u ON a.reply_to=u.id
            WHERE a.conversation_id=? AND a.role='assistant' AND a.status='complete' AND a.content != ''
            ORDER BY a.id DESC LIMIT ?""", (chat_id, limit)).fetchall()
        return list(reversed(rows))

    def turn_count(self, chat_id):
        return self.db.execute("SELECT count(*) FROM messages WHERE conversation_id=? AND role='assistant' AND status='complete' AND content != ''", (chat_id,)).fetchone()[0]

    def history(self, chat_id):
        return self.db.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (chat_id,)).fetchall()

    def last_answer(self, chat_id):
        return self.db.execute("SELECT * FROM messages WHERE conversation_id=? AND role='assistant' ORDER BY id DESC LIMIT 1", (chat_id,)).fetchone()

    def attach(self, chat_id, metadata):
        with self.db:
            self.db.execute("""INSERT INTO attachments(conversation_id,path,metadata,created_at) VALUES(?,?,?,?)
                ON CONFLICT(conversation_id,path) DO UPDATE SET metadata=excluded.metadata,created_at=excluded.created_at""",
                (chat_id, metadata["path"], json.dumps(metadata, ensure_ascii=False), now()))

    def attachments(self, chat_id):
        return [json.loads(row[0]) for row in self.db.execute("SELECT metadata FROM attachments WHERE conversation_id=? ORDER BY id", (chat_id,))]


def estimate_tokens(text):
    # Conservative heuristic for English/code, NOT an exact Qwen tokenizer.
    return (len(text.encode("utf-8")) + 1) // 2 + 12


def build_context(system, turns, attachments, current, num_ctx, output_tokens):
    budget = num_ctx - output_tokens - 256  # Template overhead + estimation margin.
    if budget <= 0:
        raise ValueError("Output reserve leaves no input context. Reduce --response-tokens.")
    system_message = {"role": "system", "content": system}
    for sample_count in (3, 1, 0):
        description = dataset_context(attachments, sample_count)
        mandatory = [system_message]
        if description:
            mandatory.append({"role": "user", "content": description})
        current_message = {"role": "user", "content": current}
        used = sum(estimate_tokens(item["content"]) for item in mandatory + [current_message])
        if used <= budget:
            break
    else:
        raise ValueError("System prompt, CSV schema and current question exceed the input budget. Shorten the question/prompt, or use a new chat with fewer attachments. Nothing was sent; the full question is saved.")
    selected = []
    for turn in reversed(turns):
        pair = [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
        cost = sum(estimate_tokens(item["content"]) for item in pair)
        if used + cost > budget:
            break  # Keep a contiguous suffix rather than unrelated older turns.
        selected.insert(0, pair)
        used += cost
    history = [item for pair in selected for item in pair]
    return mandatory + history + [current_message], {"estimated_input_tokens": used, "input_budget": budget,
        "included_turns": len(selected), "csv_sample_rows_per_file": sample_count if attachments else 0}


class OllamaClient:
    model = MODEL

    def __init__(self, host="http://127.0.0.1:11434", timeout=300):
        parsed = urllib.parse.urlparse(host)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("--host must be a local HTTP Ollama address, such as http://127.0.0.1:11434")
        self.host, self.timeout = host.rstrip("/"), timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path, timeout=3):
        with self.opener.open(self.host + path, timeout=timeout) as response:
            return json.load(response)

    def stream(self, messages, num_ctx, output_tokens):
        payload = {"model": MODEL, "messages": messages, "stream": True, "think": True,
                   "options": {"num_ctx": num_ctx, "num_predict": output_tokens}}
        request = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict) or not isinstance(chunk.get("message", {}), dict):
                        raise RuntimeError("Ollama sent an invalid streaming message.")
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    yield chunk
                    if chunk.get("done"):
                        return
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read(2048).decode(errors='replace')}") from exc
        raise RuntimeError("Ollama stream ended without a completion marker.")


HELP = """Commands:
  :new [title]    New conversation
  :chats          List conversations
  :load <id>      Resume a conversation
  :multi          Multiline input; finish with a line containing :end
  :csv <path>     Inspect and attach a UTF-8 CSV (path may contain spaces)
  :files          Show attached dataset summaries
  :history        Display saved questions and final answers
  :stats          Current resources and last response metrics
  :save <path>    Save latest answer; .py extracts Python fences; never overwrite
  :help           Show commands
  :exit           Exit (messages are saved as they arrive)
Thinking is always enabled. Generated code is displayed, never executed."""


def read_question(first, input_fn=input):
    if first.strip() != ":multi":
        return first
    print("Multiline input: finish with :end on its own line. Ctrl+C cancels.")
    lines = []
    while True:
        line = input_fn("... ")
        if line == ":end":
            return "\n".join(lines)
        lines.append(line)


def export_answer(answer, filename):
    if not answer or not answer["content"]:
        raise ValueError("No final answer is available to save.")
    path = Path(filename).expanduser()
    content = answer["content"]
    if path.suffix.lower() == ".py":
        blocks = re.findall(r"^```(?:python|py)[ \t]*\n(.*?)^```[ \t]*$", content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if not blocks:
            raise ValueError("No fenced Python code found. Save as .txt or .md instead.")
        content = "\n\n".join(block.rstrip() for block in blocks) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(content)
    return path.resolve()


class ChatApp:
    def __init__(self, store, client, prompt_path, num_ctx=4096, output_tokens=2048):
        self.store, self.client = store, client
        self.prompt_path = Path(prompt_path).expanduser()
        self.num_ctx, self.output_tokens = num_ctx, output_tokens
        self.chat_id = store.current_chat()

    def ask(self, question):
        user_id = self.store.add_message(self.chat_id, "user", question)
        assistant_id = self.store.add_message(self.chat_id, "assistant", "", reply_to=user_id, status="incomplete")
        content, thinking, metrics = "", "", {}
        status, final = "incomplete", None
        started, saved_at = time.monotonic(), 0
        try:
            system = self.prompt_path.read_text(encoding="utf-8").strip()
            if not system:
                raise ValueError("System prompt is empty; edit the prompt file before sending.")
            attachments = self.store.attachments(self.chat_id)
            messages, context = build_context(system, self.store.complete_turns(self.chat_id), attachments,
                                                question, self.num_ctx, self.output_tokens)
            context["total_saved_turns"] = self.store.turn_count(self.chat_id)
            metrics.update({"model": MODEL, "think": True, "num_ctx": self.num_ctx,
                            "output_budget": self.output_tokens, "context": context,
                            "request_messages": messages})
            print(f"Context: {context['included_turns']}/{context['total_saved_turns']} prior turns; ~{context['estimated_input_tokens']}/{context['input_budget']} input tokens estimated. Older history stays in SQLite.")
            if attachments and context["csv_sample_rows_per_file"] < 3:
                print(f"CSV samples reduced to {context['csv_sample_rows_per_file']} rows/file to fit; exact schemas are retained.")
            print("Thinking...", flush=True)
            for chunk in self.client.stream(messages, self.num_ctx, self.output_tokens):
                message = chunk.get("message", {})
                thinking += message.get("thinking", "") or ""
                piece = message.get("content", "") or ""
                if piece:
                    if not content:
                        print("\nAssistant:")
                    print(piece, end="", flush=True)
                    content += piece
                if chunk.get("done"):
                    final = chunk
                if time.monotonic() - saved_at > 1:
                    self.store.update_answer(assistant_id, content, thinking, "incomplete", metrics)
                    saved_at = time.monotonic()
            if final is None:
                raise RuntimeError("Response stream was incomplete.")
            for key in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "load_duration", "total_duration", "done_reason"):
                if key in final:
                    metrics[key] = final[key]
            status = "truncated" if final.get("done_reason") == "length" else "complete"
            if not content:
                status = "truncated" if status == "truncated" else "empty"
                print("\nNo final answer was returned. Thinking is saved; the output budget may have been exhausted.")
            if status == "truncated":
                print("\n[Output limit reached: the answer/thinking may be incomplete. This turn stays saved but is excluded from future context. Thinking was not disabled.]")
        except KeyboardInterrupt:
            status = "interrupted"
            print("\nInterrupted. Partial thinking/answer saved; incomplete turn will not enter future context.")
        except (OSError, ValueError, RuntimeError) as exc:
            status = "error"
            metrics["error"] = str(exc)
            print(f"\nRequest failed: {exc}\nQuestion and partial response saved; incomplete turn excluded from future context.")
        finally:
            metrics["wall_seconds"] = time.monotonic() - started
            self.store.update_answer(assistant_id, content, thinking, status, metrics)
            print(f"\n[{status}; {metrics['wall_seconds']:.1f} s; thinking enabled]")
        return self.store.last_answer(self.chat_id)

    def command(self, line):
        command, _, argument = line.strip().partition(" ")
        argument = argument.strip()
        if command == ":exit":
            return False
        if command == ":help":
            print(HELP)
        elif command == ":new":
            self.chat_id = self.store.new_chat(argument or "New conversation")
            print(f"Created conversation {self.chat_id}.")
        elif command == ":chats":
            for chat in self.store.chats():
                print(f"{chat['id']}: {chat['title']} ({chat['questions']} questions; {chat['updated_at']})")
        elif command == ":load":
            chat = self.store.select_chat(int(argument))
            self.chat_id = chat["id"]
            print(f"Loaded conversation {self.chat_id}: {chat['title']}. Use :history to see saved messages.")
        elif command == ":csv":
            metadata = inspect_csv(argument)
            self.store.attach(self.chat_id, metadata)
            print(f"Attached {metadata['path']}: {metadata['rows']} rows × {metadata['column_count']} columns.")
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        elif command == ":files":
            print(json.dumps(self.store.attachments(self.chat_id), ensure_ascii=False, indent=2))
        elif command == ":history":
            for message in self.store.history(self.chat_id):
                print(f"\n{message['role']} [{message['status']}, {message['created_at']}]:\n{message['content']}")
        elif command == ":stats":
            last = self.store.last_answer(self.chat_id)
            print(format_stats(self.client, json.loads(last["metrics"]) if last else None))
        elif command == ":save":
            print(f"Saved {export_answer(self.store.last_answer(self.chat_id), argument)} (not executed).")
        else:
            print("Unknown command. Use :help.")
        return True

    def run(self, input_fn=input):
        print(f"{MODEL} | thinking always ON | context {self.num_ctx} | output reserve {self.output_tokens}")
        print(f"Conversation {self.chat_id}; database messages saved automatically. :help for commands.")
        while True:
            try:
                line = input_fn("\nYou> ")
                if line.strip().startswith(":") and line.strip() != ":multi":
                    if not self.command(line):
                        break
                else:
                    question = read_question(line, input_fn)
                    if question.strip():
                        self.ask(question)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInput cancelled. Use :exit to quit.")
            except (ValueError, OSError) as exc:
                print(f"Error: {exc}")
        print("Saved. Goodbye.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Personal local Qwen chat; thinking always enabled.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "chats.sqlite3")
    parser.add_argument("--prompt", type=Path, default=ROOT / "system_prompt.txt")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=4096, help="256–4096; larger contexts require a separate test")
    parser.add_argument("--response-tokens", type=int, default=2048, help="Reserve for thinking plus answer, within the context")
    args = parser.parse_args(argv)
    if not 256 <= args.num_ctx <= 4096:
        parser.error("Context must be between 256 and 4096. 8192 has not been tested; no automatic increase is allowed.")
    if not 1 <= args.response_tokens < args.num_ctx - 256:
        parser.error("--response-tokens must be positive and leave at least 256 tokens plus input space.")
    if not args.prompt.is_file():
        parser.error(f"System prompt not found: {args.prompt}")
    try:
        client = OllamaClient(args.host)
    except ValueError as exc:
        parser.error(str(exc))
    store = Store(args.db)
    try:
        ChatApp(store, client, args.prompt, args.num_ctx, args.response_tokens).run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
