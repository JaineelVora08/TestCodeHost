"""Explicit real-server integration test. Creates a separate smoke database/logs."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    DATA.mkdir(exist_ok=True)
    # Refuse to overwrite earlier smoke results.
    import time
    prefix = DATA / f"smoke-{time.time_ns()}"
    database = prefix.with_suffix(".sqlite3")
    sessions = [
        f":csv {ROOT / 'tests/fixtures/sample.csv'}\n:multi\nRemember my coefficient is 17 and my target column is price.\nReply with only ACK.\n:end\n:stats\n:exit\n",
        ":chats\n:load 1\nWhich coefficient and target column did I specify? Reply in one short sentence.\n:stats\n:exit\n",
    ]
    logs = []
    for index, session in enumerate(sessions, 1):
        log = Path(str(prefix) + f"-session-{index}.log")
        with log.open("x", encoding="utf-8") as output:
            subprocess.run([sys.executable, str(ROOT / "chat.py"), "--db", str(database)],
                           input=session, text=True, stdout=output, stderr=subprocess.STDOUT,
                           timeout=600, check=True)
        logs.append(str(log))
        print(f"Session {index} finished: {log}", flush=True)
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        answers = [dict(row) for row in db.execute("SELECT * FROM messages WHERE role='assistant' ORDER BY id")]
        assert len(answers) == 2 and all(answer["status"] == "complete" for answer in answers), answers
        assert all(answer["thinking"].strip() for answer in answers), "No thinking was emitted"
        assert "17" in answers[1]["content"] and "price" in answers[1]["content"].lower(), answers[1]["content"]
        assert db.execute("SELECT count(*) FROM attachments").fetchone()[0] == 1
        metrics = [json.loads(answer["metrics"]) for answer in answers]
        assert all(item["think"] is True and item["num_ctx"] == 4096 for item in metrics)
        assert metrics[1]["context"]["included_turns"] == 1
        assert all(item["request_messages"][0]["content"] == (ROOT / "system_prompt.txt").read_text().strip() for item in metrics)
        assert all("bedrooms" in json.dumps(item["request_messages"]) for item in metrics)
    summary = {"database": str(database), "logs": logs, "thinking_enabled": True, "context": 4096,
               "checks": "two separate CLI processes; multiline input; CSV; streamed thinking/final; persisted follow-up; stats",
               "answers": [{"content": a["content"], "thinking_characters": len(a["thinking"]),
                            "metrics": {k: v for k, v in m.items() if k != 'request_messages'}} for a, m in zip(answers, metrics)],
               "limitation": "This verifies integration and short context recall, not mathematical accuracy."}
    report = prefix.with_suffix(".json")
    report.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
