import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat import ChatApp, MODEL, OllamaClient, ROOT, Store, build_context, estimate_tokens, export_answer, main, read_question
from csv_support import dataset_context, inspect_csv, parse_csv_argument
import monitoring


def chunks(answer="Answer", thinking="Check first", reason="stop"):
    return [{"message": {"thinking": thinking}},
            {"message": {"content": answer[:2], "thinking": " then solve"}},
            {"message": {"content": answer[2:]}},
            {"done": True, "done_reason": reason, "eval_count": 10, "eval_duration": 2_000_000_000}]


class FakeClient:
    model = MODEL

    def __init__(self, events=None):
        self.events = chunks() if events is None else events
        self.requests = []

    def stream(self, messages, num_ctx, output_tokens):
        self.requests.append({"messages": messages, "num_ctx": num_ctx, "output_tokens": output_tokens})
        yield from self.events

    def get(self, path, timeout=3):
        return {"models": [{"name": MODEL, "size": 600, "size_vram": 200, "context_length": 4096}]}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "chats.sqlite3"
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("System instructions for manual mathematical algorithms.")
        self.store = Store(self.db)
        self.client = FakeClient()
        self.app = ChatApp(self.store, self.client, self.prompt)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def ask(self, text):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            answer = self.app.ask(text)
        return answer, output.getvalue()

    def test_basic_streaming_thinking_and_system(self):
        answer, output = self.ask("What is a gradient?")
        self.assertEqual(answer["content"], "Answer")
        self.assertEqual(answer["thinking"], "Check first then solve")
        self.assertEqual(answer["status"], "complete")
        self.assertIn("Thinking...", output)
        self.assertIn("Answer", output)
        self.assertNotIn("Check first", output)
        request = self.client.requests[0]
        self.assertEqual(request["num_ctx"], 8192)
        self.assertEqual(request["output_tokens"], 4096)
        self.assertEqual(request["messages"][0], {"role": "system", "content": self.prompt.read_text()})
        metrics = json.loads(answer["metrics"])
        self.assertIs(metrics["think"], True)
        self.assertEqual(metrics["eval_count"], 10)

    def test_followup_contains_prior_question_answer_not_thinking(self):
        self.ask("Let the initial value be 5.")
        self.ask("What initial value did I give?")
        messages = self.client.requests[-1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "Let the initial value be 5.")
        self.assertEqual(messages[2]["content"], "Answer")
        self.assertNotIn("thinking", json.dumps(messages))

    def test_prompt_reloaded_for_every_request(self):
        self.ask("First")
        self.prompt.write_text("Updated system instructions")
        self.ask("Second")
        self.assertEqual(self.client.requests[-1]["messages"][0]["content"], "Updated system instructions")

    def test_restart_resumes_last_chat_and_attachments(self):
        self.ask("Remember coefficient 17.")
        csv_file = self.root / "data.csv"
        csv_file.write_text("area,bedrooms,price\n100,2,300\n")
        self.store.attach(self.app.chat_id, inspect_csv(csv_file, has_header=True))
        old_id = self.app.chat_id
        self.store.close()
        self.store = Store(self.db)
        self.app = ChatApp(self.store, self.client, self.prompt)
        self.assertEqual(self.app.chat_id, old_id)
        self.ask("Use that coefficient.")
        request = json.dumps(self.client.requests[-1]["messages"])
        self.assertIn("coefficient 17", request)
        self.assertIn("bedrooms", request)
        self.assertEqual(len(self.store.history(old_id)), 4)

    def test_new_list_and_load_isolate_conversations(self):
        self.ask("Original secret coefficient")
        first = self.app.chat_id
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.app.command(":new Second assignment")
            second = self.app.chat_id
            self.app.command(":chats")
        self.assertIn("Second assignment", output.getvalue())
        self.assertNotEqual(first, second)
        self.ask("Independent question")
        self.assertNotIn("Original secret", json.dumps(self.client.requests[-1]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.command(f":load {first}")
        self.ask("Follow up")
        self.assertIn("Original secret", json.dumps(self.client.requests[-1]))
        with self.assertRaises(ValueError):
            self.app.command(":load 9999")
        self.assertEqual(self.app.chat_id, first)

    def test_csv_command_propagates_exact_names_and_metadata(self):
        path = self.root / "my dataset.csv"
        path.write_text("area,bedrooms,Sale Price\n100,2,300\n110,,350\n")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.app.command(f":csv --header {path}")
        self.assertIn("2 rows × 3 columns", output.getvalue())
        self.ask("Generate regression code.")
        request = json.dumps(self.client.requests[-1]["messages"])
        for text in [str(path), "Sale Price", "bedrooms", "missing", "full file", "read_csv_options"]:
            self.assertIn(text, request)

    def test_legacy_csv_blocks_request_until_header_choice(self):
        self.store.attach(self.app.chat_id, {"path": "/old.csv", "rows": 99})
        answer, output = self.ask("Use the CSV")
        self.assertEqual(answer["status"], "error")
        self.assertFalse(self.client.requests)
        self.assertIn("Reattach", output)

    def test_headerless_metadata_and_rare_values_reach_model(self):
        path = self.root / "dataset.csv"
        path.write_text("6.5,55.3\nr,29.4\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.command(f":csv --no-header {path}")
        self.ask("Generate code using both columns")
        context = self.client.requests[-1]["messages"][1]["content"]
        metadata = json.loads(context.split("\n", 1)[1])[0]
        self.assertEqual(metadata["rows"], 2)
        self.assertIsNone(metadata["read_csv_options"]["header"])
        self.assertEqual(metadata["samples"][0]["values"], ["6.5", "55.3"])
        self.assertEqual(metadata["columns"][0]["non_numeric_examples"][0]["value"], "r")

    def test_csv_is_rescanned_before_each_request(self):
        path = self.root / "data.csv"
        path.write_text("x,y\n1,2\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.command(f":csv --header {path}")
        path.write_text("x,y\n1,2\n3,bad\n")
        self.ask("Use the current CSV")
        metadata = self.store.attachments(self.app.chat_id)[0]
        self.assertEqual(metadata["rows"], 2)
        self.assertEqual(metadata["columns"][1]["non_numeric_examples"][0]["value"], "bad")
        self.assertIn('"value":"bad"', self.client.requests[-1]["messages"][1]["content"])

    def test_strict_numeric_csv_change_blocks_model_request(self):
        path = self.root / "data.csv"
        path.write_text("1,2\n3,4\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.command(f":csv --no-header --numeric {path}")
        path.write_text("1,2\n3,invalid\n")
        answer, output = self.ask("Fit the data")
        self.assertEqual(answer["status"], "error")
        self.assertFalse(self.client.requests)
        self.assertIn("invalid", output)

    def test_multiline_preserves_equations_and_blank_lines(self):
        lines = iter(["min f(x) = x^2", "", "    x >= 0", ":end"])
        with contextlib.redirect_stdout(io.StringIO()):
            question = read_question(":multi", lambda _: next(lines))
        self.ask(question)
        self.assertEqual(self.client.requests[-1]["messages"][-1]["content"], "min f(x) = x^2\n\n    x >= 0")

    def test_cli_exit_and_multiline_commands(self):
        lines = iter([":multi", "line one", "line two", ":end", ":new", ":chats", ":load 1", ":exit"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.run(lambda _: next(lines))
        self.assertEqual(self.store.history(1)[0]["content"], "line one\nline two")
        self.assertEqual(self.app.chat_id, 1)

    def test_oversized_input_is_saved_but_not_sent(self):
        answer, output = self.ask("x" * 12000)
        self.assertEqual(answer["status"], "error")
        self.assertFalse(self.client.requests)
        self.assertEqual(len(self.store.history(self.app.chat_id)[0]["content"]), 12000)
        self.assertIn("exceed", output)

    def test_old_messages_preserved_when_context_drops_them(self):
        self.client.events = chunks("A" * 2000)
        for i in range(5):
            self.ask(f"Question {i}")
        self.assertEqual(len(self.store.history(self.app.chat_id)), 10)
        self.assertLess(len(self.client.requests[-1]["messages"]), 11)

    def test_interrupted_partial_response_saved_and_excluded(self):
        def interrupted(*args):
            yield {"message": {"content": "partial", "thinking": "reason"}}
            raise KeyboardInterrupt
        with patch.object(self.client, "stream", interrupted):
            answer, _ = self.ask("Interrupted question")
        self.assertEqual(answer["content"], "partial")
        self.assertEqual(answer["thinking"], "reason")
        self.assertEqual(answer["status"], "interrupted")
        self.ask("Next independent question")
        self.assertNotIn("Interrupted question", json.dumps(self.client.requests[-1]))

    def test_truncated_thinking_is_not_a_final_answer(self):
        self.client.events = [{"message": {"thinking": "Still reasoning"}}, {"done": True, "done_reason": "length"}]
        answer, output = self.ask("Hard question")
        self.assertEqual(answer["status"], "truncated")
        self.assertEqual(answer["content"], "")
        self.assertIn("Output limit", output)

    def test_failed_api_saved_and_next_request_can_succeed(self):
        with patch.object(self.client, "stream", side_effect=OSError("server offline")):
            answer, _ = self.ask("First attempt")
        self.assertEqual(answer["status"], "error")
        answer, _ = self.ask("Retry")
        self.assertEqual(answer["status"], "complete")

    def test_generated_code_never_executed_and_export_cannot_overwrite(self):
        sentinel = self.root / "executed.txt"
        code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')"
        self.client.events = chunks("```python\n" + code + "\n```")
        answer, _ = self.ask("Return code")
        self.assertFalse(sentinel.exists())
        destination = self.root / "answer.py"
        export_answer(answer, destination)
        self.assertEqual(destination.read_text(), code + "\n")
        self.assertFalse(sentinel.exists())
        with self.assertRaises(FileExistsError):
            export_answer(answer, destination)
        with self.assertRaises(ValueError):
            export_answer({"content": "Only prose"}, self.root / "bad.py")

    def test_missing_csv_and_prompt_fail_gracefully(self):
        with self.assertRaises(ValueError):
            self.app.command(f":csv {self.root / 'missing.csv'}")
        self.prompt.unlink()
        answer, _ = self.ask("Question")
        self.assertEqual(answer["status"], "error")
        self.assertFalse(self.client.requests)


class ProtocolTests(unittest.TestCase):
    def test_real_client_payload_and_ndjson_stream_with_mocks(self):
        client = OllamaClient()
        wire = io.BytesIO(b"\n".join(json.dumps(event).encode() for event in chunks()))
        with patch.object(client.opener, "open", return_value=wire) as opened:
            events = list(client.stream([{"role": "system", "content": "instructions"}], 4096, 2048))
        request = opened.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertIs(payload["think"], True)
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["options"], {"num_ctx": 4096, "num_predict": 2048})
        self.assertIn("thinking", events[1]["message"])
        self.assertIn("content", events[1]["message"])

    def test_stream_missing_completion_and_error_chunk(self):
        client = OllamaClient()
        for body in [b'{"message":{"content":"partial"}}\n', b'{"error":"OOM"}\n', b'[]\n', b'{"message":"bad"}\n']:
            with self.subTest(body=body), patch.object(client.opener, "open", return_value=io.BytesIO(body)):
                with self.assertRaises(RuntimeError):
                    list(client.stream([], 4096, 2048))

    def test_remote_host_and_unsupported_context_rejected(self):
        for host in ["http://example.com", "https://localhost", "http://localhost/path", "http://user@localhost"]:
            with self.assertRaises(ValueError):
                OllamaClient(host)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--num-ctx", "16384"])
        self.assertEqual(error.exception.code, 2)

    def test_context_keeps_recent_complete_pairs(self):
        turns = [{"question": "Old" * 1000, "answer": "long" * 1000},
                 {"question": "Latest condition", "answer": "Latest answer"}]
        messages, info = build_context("system", turns, [], "Follow up", 4096, 2048)
        self.assertEqual(info["included_turns"], 1)
        self.assertEqual(messages[1]["content"], "Latest condition")
        self.assertLessEqual(info["estimated_input_tokens"], info["input_budget"])

    def test_real_prompt_fits_default_context_with_short_question(self):
        _, info = build_context((ROOT / "system_prompt.txt").read_text(), [], [], "What is gradient descent?", 8192, 4096)
        self.assertLess(info["estimated_input_tokens"], info["input_budget"])


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "data.csv"

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, text):
        self.path.write_text(text, encoding="utf-8")
        return inspect_csv(self.path, has_header=True)

    def test_schema_missing_types_and_numeric_ranges(self):
        result = self.inspect(' exact name ,price,flag,empty\n1,3.5,true,\n2,,false, \n3,8,true,\n')
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["columns"][0]["name"], " exact name ")
        self.assertEqual([c["type"] for c in result["columns"]], ["integer", "number", "boolean", "unknown (all missing)"])
        self.assertEqual(result["columns"][1]["missing"], 1)
        self.assertEqual(result["columns"][1]["max"], 8)

    def test_semicolon_and_quoted_fields(self):
        result = self.inspect('area;bedrooms;description\n10;2;"comma, and\nnewline"\n')
        self.assertEqual(result["delimiter"], ";")
        self.assertEqual(result["rows"], 1)
        self.assertIn("\n", result["samples"][0]["values"][-1])

    def test_spaces_in_headers_survive_dialect_detection(self):
        result = self.inspect("area, bedrooms, price\n1, 2, 3\n4, 5, 6\n")
        self.assertEqual([column["name"] for column in result["columns"]], ["area", " bedrooms", " price"])

    def test_large_integer_ranges_are_not_rounded_to_float(self):
        result = self.inspect("id\n9007199254740993\n9007199254740995\n")
        self.assertEqual(result["columns"][0]["min"], 9007199254740993)
        self.assertEqual(result["columns"][0]["max"], 9007199254740995)

    def test_context_can_drop_samples_without_dropping_schema(self):
        result = self.inspect("area,bedrooms,price\n100,2,300\n120,3,400\n")
        minimum = estimate_tokens("system") + estimate_tokens(dataset_context([result], 0)) + estimate_tokens("question")
        messages, info = build_context("system", [], [result], "question", minimum + 256 + 1000 + 5, 1000)
        self.assertEqual(info["csv_sample_rows_per_file"], 0)
        self.assertIn('"name":"bedrooms"', messages[1]["content"])
        self.assertEqual(messages[-1]["content"], "question")

    def test_full_scan_bounded_samples_and_context(self):
        result = self.inspect("x,y\n" + "".join(f"{i},{i*2}\n" for i in range(1000)))
        self.assertEqual(result["rows"], 1000)
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual(result["columns"][1]["max"], 1998)
        self.assertLess(len(dataset_context([result])), 2500)
        self.assertIn('"samples":[]', dataset_context([result], 0))

    def test_malformed_duplicate_empty_invalid_encoding(self):
        for text in ["", "x,x\n1,2\n", "x,y\n1,2,3\n", 'x,y\n"unterminated,2\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.inspect(text)
        self.path.write_bytes(b"x,y\n\xff,2\n")
        with self.assertRaises(ValueError):
            inspect_csv(self.path, has_header=True)

    def test_missing_permission_and_size_errors(self):
        with self.assertRaises(ValueError):
            inspect_csv(self.path, has_header=True)
        self.path.write_text("x\n1\n")
        with patch("csv_support.MAX_BYTES", 1), self.assertRaises(ValueError):
            inspect_csv(self.path, has_header=True)
        with patch.object(Path, "open", side_effect=PermissionError("denied")), self.assertRaises(ValueError):
            inspect_csv(self.path)

    def test_large_cell_bounded_and_mixed_type_not_numeric(self):
        result = self.inspect("x,y\n" + "a"*1000 + ",1\ntext,word\n")
        self.assertLess(len(result["samples"][0]["values"][0]), 200)
        self.assertNotIn("min", result["columns"][1])

    def test_explicit_header_choice_and_paths_with_spaces(self):
        for argument in [str(self.path), "--header --no-header x.csv", "--header", "--unknown x.csv"]:
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                parse_csv_argument(argument)
        path, options = parse_csv_argument("--no-header --numeric /tmp/my data.csv")
        self.assertEqual(path, "/tmp/my data.csv")
        self.assertEqual(options, {"has_header": False, "numeric": True})
        _, options = parse_csv_argument("--header --delimiter semicolon /tmp/my data.csv")
        self.assertEqual(options["delimiter"], ";")

    def test_ragged_noncomma_files_cannot_pass_as_single_column(self):
        for separator in [";", "\t", "|"]:
            with self.subTest(separator=separator), self.assertRaisesRegex(ValueError, "expected 2 fields"):
                self.inspect(f"x{separator}y\n1{separator}2{separator}3\n")

    def test_explicit_delimiter_and_ambiguous_fallback(self):
        self.path.write_text("x;y,z\n1;2,3;4\n5;6,7,8\n")
        with self.assertRaisesRegex(ValueError, "delimiter is ambiguous"):
            inspect_csv(self.path, has_header=True)
        self.path.write_text("x;y,z\n1;2,3;4\n")
        result = inspect_csv(self.path, has_header=True, delimiter=",")
        self.assertEqual([column["name"] for column in result["columns"]], ["x;y", "z"])

    def test_headerless_first_record_and_numeric_headers(self):
        self.path.write_text("6.5,55.3\n2.9,52.5\n")
        with self.assertRaises(ValueError):
            inspect_csv(self.path)
        result = inspect_csv(self.path, has_header=False)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["samples"][0]["values"], ["6.5", "55.3"])
        self.assertIsNone(result["read_csv_options"]["header"])
        self.assertEqual(result["read_csv_options"]["names"], ["column_1", "column_2"])
        # Numeric-looking names are legal when the user explicitly chooses headers.
        result = inspect_csv(self.path, has_header=True)
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["columns"][0]["name"], "6.5")

    def test_anomaly_outside_random_samples_stays_in_context(self):
        result = self.inspect("x,y\n" + "1,2\n" * 999 + "r,29.4\n")
        column = result["columns"][0]
        self.assertEqual((column["numeric_count"], column["non_numeric_count"]), (999, 1))
        self.assertEqual(column["non_numeric_examples"], [{"row": 1000, "line": 1001, "value": "r"}])
        self.assertTrue(all(sample["values"][0] != "r" for sample in result["samples"]))
        self.assertIn('"value":"r"', dataset_context([result], 0))
        self.assertTrue(result["warnings"])

    def test_numeric_validation_rejects_missing_nonfinite_and_text(self):
        for value in ["r", "NaN", "inf", "-inf", "NA", "", " "]:
            self.path.write_text(f"x,y\n1,2\n3,{value}\n")
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "data row 2, line 3"):
                inspect_csv(self.path, has_header=True, numeric=True)
        self.path.write_text("x,y\n1,2\n3,4\n")
        self.assertTrue(inspect_csv(self.path, has_header=True, numeric=True)["numeric_required"])

    def test_blank_records_and_literal_missing_markers_preserved(self):
        result = self.inspect("x,y\n1,NA\n\n3,4\n")
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["columns"][0]["missing"], 1)
        self.assertEqual(result["columns"][1]["non_numeric_examples"][0]["value"], "NA")
        self.assertIs(result["read_csv_options"]["keep_default_na"], False)
        self.assertIs(result["read_csv_options"]["skip_blank_lines"], False)
        self.assertEqual(len(result["sha256"]), 64)


class StatsTests(unittest.TestCase):
    def test_loaded_model_without_allocation_metrics_does_not_assume_cpu(self):
        with patch.object(FakeClient, "get", return_value={"models": [{"name": MODEL, "size": 600}]}), patch("monitoring.gpu_stats", return_value=[]):
            report = monitoring.format_stats(FakeClient())
        self.assertIn("allocation metrics unavailable", report)
        self.assertNotIn("100% CPU", report)

    def test_missing_monitoring_tools_and_server(self):
        with patch("monitoring.system_memory", return_value=None), patch("monitoring.subprocess.run", side_effect=FileNotFoundError), patch.object(FakeClient, "get", side_effect=OSError("offline")):
            report = monitoring.format_stats(FakeClient())
        self.assertIn("RAM/swap: unavailable", report)
        self.assertIn("GPU metrics: unavailable", report)
        self.assertIn("Ollama placement: unavailable", report)

    def test_realistic_metrics_and_placement(self):
        completed = subprocess.CompletedProcess([], 0, "RTX 2050, 2979, 4096, 22\n", "")
        with patch("monitoring.subprocess.run", return_value=completed):
            report = monitoring.format_stats(FakeClient(), {"wall_seconds": 5, "eval_count": 10, "eval_duration": 2_000_000_000})
        self.assertIn("2979/4096 MiB", report)
        self.assertIn("67% CPU / 33% GPU", report)
        self.assertIn("5.00 tokens/s", report)


if __name__ == "__main__":
    unittest.main()
