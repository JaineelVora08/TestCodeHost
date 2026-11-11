"""Check that supplied loading options reproduce the inspected CSV, without a model."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_support import inspect_csv

try:
    import pandas as pd
except ImportError:
    pd = None


@unittest.skipIf(pd is None, "Optional pandas loading check; no package installation needed")
class CsvLoadingTests(unittest.TestCase):
    def test_loading_options_match_rows_names_and_raw_cells(self):
        cases = [
            (True, "x,y\n1,2\n3,4\n"),
            (False, "6.5,55.3\n2.9,52.5\nr,29.4\n"),
            (True, " x , y \n001,NA\nnull,NaN\n\n3,\n"),
            (True, 'x;y\n"comma, and\nnewline";2\n"a; b";3\n'),
            (False, '"hello, world",1\n"a""b",2\n'),
            (True, "6.5,55.3\n1,2\n"),
            (True, 'x\n\n""\nNA\n'),
            (False, "label\tvalue\na\t2\n"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            for has_header, content in cases:
                with self.subTest(has_header=has_header, content=content):
                    path.write_text(content, encoding="utf-8-sig")
                    metadata = inspect_csv(path, has_header=has_header)
                    frame = pd.read_csv(path, **metadata["read_csv_options"])
                    self.assertEqual(frame.shape, (metadata["rows"], metadata["column_count"]))
                    self.assertEqual(list(frame.columns), [column["name"] for column in metadata["columns"]])
                    for sample in metadata["samples"]:
                        self.assertEqual(frame.iloc[sample["row"] - 1].tolist(), sample["values"])


if __name__ == "__main__":
    unittest.main()
