"""Black-box checks for safe trajectory-to-line export (stdlib only)."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dataset" / "scripts" / "split_lines.py"


def line_points(baseline, count=201):
    points = [
        [float(i), baseline + 10.0 * math.sin(i * math.pi / 25.0), 0]
        for i in range(count)
    ]
    points[-1][2] = 1
    return points


class SplitLinesCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="handwriting-lines-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.source = self.work / "source"
        (self.source / "jsons").mkdir(parents=True)
        (self.source / "texts").mkdir()
        self.output = self.work / "output"

    def add_source(self, sample_id, points, text):
        stem = f"trajectory_{sample_id}"
        (self.source / "jsons" / f"{stem}.json").write_text(
            json.dumps(points, ensure_ascii=False), encoding="utf-8"
        )
        (self.source / "texts" / f"{stem}.txt").write_text(text, encoding="utf-8")

    def source_snapshot(self):
        return {
            str(path.relative_to(self.source)): path.read_bytes()
            for path in self.source.rglob("*")
            if path.is_file()
        }

    def run_cli(self, *, write=False, overrides=None, output=None):
        before = self.source_snapshot()
        command = [
            sys.executable,
            str(SCRIPT),
            "--dataset-dir",
            str(self.source),
            "--output-dir",
            str(output if output is not None else self.output),
        ]
        if write:
            command.append("--write")
        if overrides is not None:
            override_path = self.work / "overrides.json"
            override_path.write_text(
                json.dumps(overrides, ensure_ascii=False), encoding="utf-8"
            )
            command.extend(["--overrides", str(override_path)])
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(before, self.source_snapshot(), "Source files were changed")
        return result

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def read_report(self):
        return json.loads((self.output / "report.json").read_text(encoding="utf-8"))

    def read_manifest(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def assert_geometry_preserved(self, original, rows):
        rows = sorted(rows, key=lambda row: row["line_index"])
        next_start = 0
        for expected_index, row in enumerate(rows, 1):
            start, stop = row["point_range"]
            self.assertEqual(row["line_index"], expected_index)
            self.assertEqual(start, next_start, "A boundary lost or duplicated points")
            self.assertGreater(stop, start)
            original_part = original[start:stop]
            stem = f"trajectory_{row['output_id']}"
            self.assertRegex(stem, r"^trajectory_\d+$")
            actual = json.loads(
                (self.output / "jsons" / f"{stem}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(actual), len(original_part))
            self.assertEqual([p[2] for p in actual], [p[2] for p in original_part])
            self.assertEqual(actual[-1][2], 1)
            for previous, current, old_previous, old_current in zip(
                actual, actual[1:], original_part, original_part[1:]
            ):
                for axis in (0, 1):
                    self.assertAlmostEqual(
                        current[axis] - previous[axis],
                        old_current[axis] - old_previous[axis],
                        places=9,
                    )
            exported_text = (self.output / "texts" / f"{stem}.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(exported_text.rstrip("\n"), row["text"])
            next_start = stop
        self.assertEqual(next_start, len(original))

    def test_default_is_report_only(self):
        self.add_source(1, line_points(10) + line_points(110), "Первая\nВторая")
        self.assert_success(self.run_cli())
        self.assertTrue((self.output / "report.json").is_file())
        self.assertTrue((self.output / "index.html").is_file())
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertFalse((self.output / "jsons").exists())
        self.assertFalse((self.output / "texts").exists())
        report = self.read_report()
        self.assertEqual(report["summary"]["total_sources"], 1)
        self.assertEqual(report["summary"]["ready_sources"], 1)
        self.assertEqual(report["summary"]["review_sources"], 0)
        self.assertEqual(report["summary"]["output_samples"], 2)

    def test_export_preserves_endpoints_short_strokes_and_hyphen(self):
        first = line_points(10) + [[190.0, 5.0, 0], [190.0, 5.0, 1]]
        points = first + line_points(110)
        self.add_source(1, points, "Первая стро-\nка.")
        self.assert_success(self.run_cli(write=True))
        sample = self.read_report()["samples"][0]
        self.assertEqual(sample["status"], "ready")
        self.assertEqual(sample["breaks"], [len(first)])
        rows = self.read_manifest()["samples"]
        self.assertEqual([row["text"] for row in rows], ["Первая стро-", "ка."])
        self.assertTrue(all(row["source_id"] == 1 for row in rows))
        self.assert_geometry_preserved(points, rows)

    def test_three_lines_have_complete_nonoverlapping_ranges(self):
        points = line_points(10) + line_points(110) + line_points(210)
        self.add_source(185, points, "Один\nДва\nТри")
        self.assert_success(self.run_cli(write=True))
        sample = self.read_report()["samples"][0]
        self.assertEqual(sample["breaks"], [201, 402])
        rows = self.read_manifest()["samples"]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["text"] for row in rows], ["Один", "Два", "Три"])
        self.assert_geometry_preserved(points, rows)

    def test_large_pen_down_jump_is_not_a_line_boundary(self):
        first = line_points(10)
        first[-1][2] = 0
        self.add_source(1, first + line_points(110), "Одна подпись")
        self.assert_success(self.run_cli())
        self.assertEqual(self.read_report()["samples"][0]["breaks"], [])

    def test_upward_return_for_diacritic_is_not_a_line_boundary(self):
        points = line_points(110) + [[20.0, 10.0, 0], [20.0, 10.0, 1]]
        self.add_source(115, points, "Ё")
        self.assert_success(self.run_cli())
        self.assertEqual(self.read_report()["samples"][0]["breaks"], [])

    def test_disagreement_between_labels_and_geometry_is_not_exported(self):
        self.add_source(171, line_points(10) + line_points(110), "Подпись без переноса")
        self.assert_success(self.run_cli(write=True))
        report = self.read_report()
        self.assertEqual(report["summary"]["review_sources"], 1)
        self.assertEqual(report["summary"]["output_samples"], 0)
        self.assertEqual(report["samples"][0]["status"], "needs_review")
        self.assertTrue(report["samples"][0]["issues"])
        self.assertEqual(self.read_manifest()["samples"], [])
        self.assertEqual(list((self.output / "jsons").glob("*.json")), [])
        self.assertEqual(list((self.output / "texts").glob("*.txt")), [])

    def test_explicit_reviewed_override_resolves_missing_text_break(self):
        points = line_points(10) + line_points(110)
        self.add_source(1, points, "Первая Вторая")
        override = {"trajectory_1": {"breaks": [201], "texts": ["Первая", "Вторая"]}}
        self.assert_success(self.run_cli(write=True, overrides=override))
        self.assertEqual(self.read_report()["samples"][0]["status"], "ready")
        rows = self.read_manifest()["samples"]
        self.assertEqual([row["text"] for row in rows], ["Первая", "Вторая"])
        self.assert_geometry_preserved(points, rows)

    def test_explicit_text_override_recovers_a_missing_text_file(self):
        points = line_points(10) + line_points(110)
        self.add_source(1, points, "Temporary fixture label")
        text_path = self.source / "texts" / "trajectory_1.txt"
        text_path.unlink()
        override = {"trajectory_1": {"breaks": [201], "texts": ["Первая", "Вторая"]}}
        self.assert_success(self.run_cli(write=True, overrides=override))
        sample = self.read_report()["samples"][0]
        self.assertEqual(sample["status"], "ready")
        self.assertEqual(sample["texts"], ["Первая", "Вторая"])
        rows = self.read_manifest()["samples"]
        self.assertEqual([row["text"] for row in rows], ["Первая", "Вторая"])
        self.assert_geometry_preserved(points, rows)
        self.assertFalse(text_path.exists(), "The original missing label must stay untouched")

    def test_bom_and_crlf_produce_two_clean_text_lines(self):
        points = line_points(10) + line_points(110)
        self.add_source(1, points, "Temporary fixture label")
        text_path = self.source / "texts" / "trajectory_1.txt"
        text_path.write_bytes("\ufeffПервая\r\nВторая\r\n".encode("utf-8"))
        self.assert_success(self.run_cli(write=True))
        sample = self.read_report()["samples"][0]
        self.assertEqual(sample["status"], "ready")
        self.assertEqual(sample["texts"], ["Первая", "Вторая"])
        rows = self.read_manifest()["samples"]
        self.assertEqual([row["text"] for row in rows], ["Первая", "Вторая"])
        self.assert_geometry_preserved(points, rows)

    def test_single_line_longer_than_training_limit_is_preserved_in_full(self):
        points = line_points(10, count=3201)
        self.add_source(466, points, "Длинная строка сохраняется целиком.")
        self.assert_success(self.run_cli(write=True))
        sample = self.read_report()["samples"][0]
        self.assertEqual(sample["status"], "ready")
        self.assertEqual(sample["breaks"], [])
        rows = self.read_manifest()["samples"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["point_range"], [0, 3201])
        self.assert_geometry_preserved(points, rows)

    def test_override_cannot_cut_through_a_stroke(self):
        self.add_source(1, line_points(10) + line_points(110), "Один\nДва")
        override = {"trajectory_1": {"breaks": [100], "texts": ["Один", "Два"]}}
        result = self.run_cli(write=True, overrides=override)
        if result.returncode == 0:
            self.assertEqual(self.read_report()["samples"][0]["status"], "needs_review")
            self.assertEqual(self.read_manifest()["samples"], [])
        self.assertEqual(list((self.output / "jsons").glob("*.json")), [])

    def test_nonempty_output_is_rejected_without_changing_it(self):
        self.add_source(1, line_points(10), "Строка")
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_bytes(b"Do not replace me")
        result = self.run_cli(write=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b"Do not replace me")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_source_directories_are_not_valid_output_destinations(self):
        self.add_source(1, line_points(10), "Строка")
        for destination in (self.source, self.source / "jsons", self.source / "texts"):
            with self.subTest(destination=destination.name):
                result = self.run_cli(write=True, output=destination)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
