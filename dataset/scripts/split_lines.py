"""Detect physical line returns; preview or export lossless JSON/TXT line pairs.

No PyTorch, NumPy or GUI dependencies. Coordinates use screen convention: y down.
An end-of-stroke flag belongs to the point BEFORE a possible line break.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from statistics import median


DATASET_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DATASET_DIR.parent / "runs" / "line_split" / "preview"
COLORS = ("#1565c0", "#c62828", "#16834b", "#8042a5")


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_points(points):
    if not isinstance(points, list) or not points:
        raise ValueError("Trajectory must be a nonempty list of [x, y, state].")
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"Point {index}: expected [x, y, state].")
        if any(isinstance(v, bool) or not isinstance(v, (float, int))
               or not math.isfinite(v) for v in point):
            raise ValueError(f"Point {index}: expected finite numeric values.")
        if point[2] not in (0, 1):
            raise ValueError(f"Point {index}: state must be 0 or 1.")
    if points[-1][2] != 1:
        raise ValueError("Last stroke is unfinished (last state is not 1).")


def stroke_ranges(points):
    start = 0
    for index, point in enumerate(points):
        if point[2] == 1:
            yield start, index + 1
            start = index + 1
    if start < len(points):
        yield start, len(points)


def split_points(points, breaks):
    if not isinstance(breaks, list) or any(type(k) is not int for k in breaks):
        raise ValueError("breaks must be a list of integer point indices.")
    if sorted(set(breaks)) != breaks:
        raise ValueError("breaks must be unique and increasing.")
    for cut in breaks:
        if not 0 < cut < len(points):
            raise ValueError(f"Break {cut} is outside the trajectory.")
        if points[cut - 1][2] != 1:
            raise ValueError(f"Break {cut} cuts through a stroke (previous state != 1).")
    edges = [0, *breaks, len(points)]
    return [points[a:b] for a, b in zip(edges, edges[1:])]


def detect_breaks(points, min_return_fraction=0.35, min_down_factor=1.0,
                  context_points=150):
    """Return candidates with measured evidence, not probability estimates.

    Consider weaker returns too: they require review instead of silently being
    exported as one line. Text newlines never force a geometric cut.
    """
    width = max(p[0] for p in points) - min(p[0] for p in points)
    strokes = list(stroke_ranges(points))
    heights = []
    for start, stop in strokes:
        ys = [p[1] for p in points[start:stop]]
        height = percentile(ys, 0.95) - percentile(ys, 0.05)
        if stop - start >= 5 and height > 0:
            heights.append(((start + stop - 1) / 2, height))
    candidates = []
    for cut in range(1, len(points)):
        before, after = points[cut - 1], points[cut]
        dx, dy = after[0] - before[0], after[1] - before[1]
        if before[2] != 1 or width <= 0 or dy <= 0:
            continue
        if -dx < 0.7 * min_return_fraction * width:
            continue
        nearby = sorted(heights, key=lambda item: abs(item[0] - cut))[:8]
        height = median(h for _, h in nearby) if nearby else None
        prev_y = [p[1] for p in points[max(0, cut - context_points):cut]]
        next_y = [p[1] for p in points[cut:cut + context_points]]
        shift = median(next_y) - median(prev_y)
        strong = (height is not None and -dx >= min_return_fraction * width
                  and dy >= min_down_factor * height
                  and shift >= min_down_factor * height)
        candidates.append({
            "point_index": cut, "dx": dx, "dy": dy,
            "return_fraction": -dx / width, "local_stroke_height": height,
            "median_y_shift": shift, "strong": strong,
        })
    return candidates


def geometry_issues(points, breaks):
    segments = split_points(points, breaks)
    centers = [median(p[1] for p in segment) for segment in segments]
    if any(a >= b for a, b in zip(centers, centers[1:])):
        return ["Line centers do not advance downwards; inspect the split."]
    # A late dot/cross can belong to an earlier physical line. A temporal cut
    # cannot repair this automatically; keep the whole source for review.
    for line_index, segment in enumerate(segments):
        for start, stop in stroke_ranges(segment):
            center = median(p[1] for p in segment[start:stop])
            nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - center))
            if nearest != line_index:
                return ["A stroke is closer to another line; possible delayed stroke."]
    return []


def text_lines(raw):
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")


def load_overrides(path, source_names):
    if path is None:
        return {}
    overrides = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be an object keyed by trajectory_<id>.")
    unknown = set(overrides) - source_names
    if unknown:
        raise ValueError(f"Unknown override sources: {', '.join(sorted(unknown))}")
    for name, override in overrides.items():
        if not isinstance(override, dict) or not set(override) <= {"breaks", "texts"}:
            raise ValueError(f"{name}: override may contain only breaks and texts.")
    return overrides


def analyze_source(path, text_dir, override, settings):
    source_id = int(path.stem.split("_")[-1])
    record = {
        "source_id": source_id, "source_name": path.stem,
        "status": "needs_review", "breaks": [], "texts": [], "issues": [],
        "warnings": [],
        "candidates": [], "outputs": [], "manual_override": override is not None,
        "source_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    try:
        points = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_points(points)
    except (ValueError, UnicodeError) as exc:
        record["issues"].append(str(exc))
        return record, []
    record["point_count"] = len(points)
    candidates = detect_breaks(points, **settings)
    breaks = [c["point_index"] for c in candidates]
    record["candidates"] = candidates
    label_path = text_dir / f"{path.stem}.txt"
    labels = []
    label_issue = None
    if label_path.exists():
        record["source_text_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        try:
            labels = text_lines(label_path.read_text(encoding="utf-8-sig"))
        except UnicodeError as exc:
            label_issue = f"Cannot decode text: {exc}"
    else:
        label_issue = "Missing TXT label."
    record["source_texts"] = labels[:]
    if override is not None:
        breaks = override.get("breaks", breaks)
        labels = override.get("texts", labels)
        # Invalid explicit overrides are configuration errors, not auto guesses.
        split_points(points, breaks)
        if (not isinstance(labels, list) or len(labels) != len(breaks) + 1
                or any(not isinstance(t, str) or not t.strip()
                       or "\n" in t or "\r" in t for t in labels)):
            raise ValueError(f"{path.stem}: override needs one nonempty text per line.")
    elif any(not c["strong"] for c in candidates):
        record["issues"].append("A return is below confidence thresholds; inspect it.")
    if label_issue:
        if override is not None and "texts" in override:
            record["warnings"].append(label_issue + " Using reviewed override texts.")
        else:
            record["issues"].append(label_issue)
    record["breaks"] = breaks
    record["texts"] = labels
    if len(labels) != len(breaks) + 1:
        record["issues"].append(
            f"Geometry has {len(breaks) + 1} lines, TXT has {len(labels)}. "
            "Provide reviewed texts/breaks in --overrides.")
    if any(not t.strip() for t in labels):
        record["issues"].append("Empty text line; it cannot be silently removed.")
    record["issues"].extend(geometry_issues(points, breaks))
    record["line_point_counts"] = [len(part) for part in split_points(points, breaks)]
    if not record["issues"]:
        record["status"] = "ready"
    return record, points


def svg_preview(points, breaks):
    if not points:
        return "<p>Некорректная траектория — см. описание ошибки.</p>"
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    left, top = min(xs) - 8, min(ys) - 8
    width, height = max(xs) - left + 8, max(ys) - top + 8
    pieces = [f'<svg viewBox="{left} {top} {width} {height}" '
              'role="img" aria-label="Строки траектории разными цветами">']
    for start, stop in stroke_ranges(points):
        color = COLORS[bisect_right(breaks, start) % len(COLORS)]
        stroke = points[start:stop]
        if all(p[:2] == stroke[0][:2] for p in stroke):
            pieces.append(f'<circle cx="{stroke[0][0]}" cy="{stroke[0][1]}" '
                          f'r="1.3" fill="{color}"/>')
        else:
            coords = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in stroke)
            pieces.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                          'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')
    pieces.append("</svg>")
    return "".join(pieces)


def write_html(path, report, point_sets):
    counts = report["summary"]
    blocks = []
    ordered = sorted(report["samples"], key=lambda r: (r["status"] == "ready", r["source_id"]))
    for record in ordered:
        needs_review = record["status"] != "ready"
        labels = "".join(f'<li style="color:{COLORS[i % len(COLORS)]}">{escape(t)}</li>'
                         for i, t in enumerate(record["texts"]))
        issues = "".join(f"<li>{escape(t)}</li>" for t in record["issues"] + record["warnings"])
        evidence = "; ".join(
            f"точка {c['point_index']}: dx={c['dx']:.1f}, dy={c['dy']:.1f}, "
            f"сдвиг строки={c['median_y_shift']:.1f}" for c in record["candidates"])
        flag = "НУЖНА ПРОВЕРКА" if needs_review else "готово"
        blocks.append(
            f'<details id="source-{record["source_id"]}" {"open" if needs_review else ""}>'
            f'<summary>{escape(record["source_name"])} — {flag}; '
            f'{len(record["breaks"]) + 1} строк</summary>'
            f'<ul class="issues">{issues}</ul>'
            f'{svg_preview(point_sets[record["source_id"]], record["breaks"])}'
            f'<ol>{labels}</ol><p class="evidence">{escape(evidence)}</p></details>')
    html = f'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>Разделение рукописи на строки</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#182536;background:#f6f8fb}}
h1{{font-size:26px}}details{{background:white;border:1px solid #d9e0e8;border-radius:8px;padding:14px;margin:12px 0}}
summary{{cursor:pointer;font-weight:600}}svg{{display:block;width:100%;max-height:420px;margin:15px 0}}
.issues{{color:#a12424}}.evidence{{font-size:13px;color:#536277}}li{{white-space:pre-wrap}}a{{color:#1565c0}}</style>
<h1>Разделение рукописи на строки</h1>
<p>Исходников: {counts['total_sources']}. Готово: {counts['ready_sources']}.
Требуют проверки: {counts['review_sources']}. Выходных строк: {counts['output_samples']}.</p>
<p>{'JSON/TXT сохранены.' if report['exported'] else 'Предварительный просмотр: JSON/TXT не экспортированы.'}
Цвет показывает предлагаемую строку. Разверните запись для просмотра.</p>
<p>Координаты и штрихи сохраняются полностью. Индекс границы — первая точка новой строки,
считая от нуля. Дефисы сохраняются. Геометрическая эвристика не заменяет проверку подписей.</p>
<p><a href="report.json">Подробный отчёт JSON</a></p>
{''.join(blocks)}</html>'''
    path.write_text(html, encoding="utf-8")


def check_output_directory(source_dir, output_dir):
    if (source_dir.is_relative_to(output_dir)
            or output_dir.is_relative_to(source_dir / "jsons")
            or output_dir.is_relative_to(source_dir / "texts")):
        raise ValueError("Output must not overlap the source dataset or its raw folders.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory is not empty; choose a new path: {output_dir}")


def run(args):
    source_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    check_output_directory(source_dir, output_dir)
    paths = sorted((source_dir / "jsons").glob("trajectory_*.json"),
                   key=lambda p: p.name)
    invalid_names = [p.name for p in paths if not re.fullmatch(r"trajectory_\d+", p.stem)]
    if invalid_names:
        raise ValueError(f"Expected numeric trajectory IDs: {invalid_names}")
    paths.sort(key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise ValueError(f"No trajectory JSON files found in {source_dir / 'jsons'}")
    source_ids = [int(p.stem.split("_")[-1]) for p in paths]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate numeric source IDs (for example 1 and 01).")
    overrides = load_overrides(Path(args.overrides) if args.overrides else None,
                               {p.stem for p in paths})
    settings = {"min_return_fraction": args.min_return_fraction,
                "min_down_factor": args.min_down_factor, "context_points": args.context_points}
    if (not math.isfinite(args.min_return_fraction) or not 0 < args.min_return_fraction <= 1
            or not math.isfinite(args.min_down_factor) or args.min_down_factor <= 0
            or args.context_points < 1):
        raise ValueError("Require 0 < return fraction <= 1, down factor > 0 and context >= 1.")
    records, point_sets, samples = [], {}, []
    for path in paths:
        record, points = analyze_source(path, source_dir / "texts", overrides.get(path.stem), settings)
        records.append(record)
        point_sets[record["source_id"]] = points
        if record["status"] != "ready":
            continue
        edges = [0, *record["breaks"], len(points)]
        for index, (start, stop, text) in enumerate(zip(edges, edges[1:], record["texts"]), 1):
            output_id = len(samples) + 1
            sample = {
                "output_id": output_id, "source_id": record["source_id"], "line_index": index,
                "split_group": record["source_name"], "point_range": [start, stop], "text": text,
                "json_path": f"jsons/trajectory_{output_id}.json",
                "text_path": f"texts/trajectory_{output_id}.txt",
                "ends_with_hyphen": text.rstrip().endswith("-"),
                "continuation_from_previous": index > 1 and record["texts"][index - 2].rstrip().endswith("-"),
                "source_json_sha256": record["source_json_sha256"],
                "source_text_sha256": record.get("source_text_sha256"),
                "coordinate_offset": [0, 0],
            }
            samples.append(sample)
            record["outputs"].append(output_id)
    statuses = Counter(r["status"] for r in records)
    summary = {
        "total_sources": len(records), "ready_sources": statuses["ready"],
        "review_sources": statuses["needs_review"], "output_samples": len(samples),
        "detected_breaks": sum(len(r["candidates"]) for r in records),
        "hyphenated_line_ends": sum(s["ends_with_hyphen"] for s in samples),
        "exported_points": sum(s["point_range"][1] - s["point_range"][0] for s in samples) if args.write else 0,
    }
    report = {"schema_version": 1, "source_dir": str(source_dir), "exported": args.write,
              "settings": settings, "summary": summary, "samples": records}
    # All validation precedes writing. Never reuse a populated export directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        (output_dir / "jsons").mkdir()
        (output_dir / "texts").mkdir()
        for sample in samples:
            start, stop = sample["point_range"]
            points = point_sets[sample["source_id"]][start:stop]
            (output_dir / sample["json_path"]).write_text(
                json.dumps(points, ensure_ascii=False), encoding="utf-8")
            (output_dir / sample["text_path"]).write_text(sample["text"] + "\n", encoding="utf-8")
        manifest = {"schema_version": 1, "source_dir": str(source_dir), "settings": settings,
                    "review_source_ids": [r["source_id"] for r in records if r["status"] != "ready"],
                    "samples": samples}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output_dir / "index.html", report, point_sets)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'index.html'}")
    if summary["review_sources"]:
        print("Sources needing review were excluded from export; see report.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="New or empty directory only.")
    parser.add_argument("--write", action="store_true", help="Export ready JSON/TXT pairs as well as reports.")
    parser.add_argument("--overrides", help="Reviewed point boundaries/texts keyed by trajectory_<source_id>.")
    parser.add_argument("--min-return-fraction", type=float, default=0.35)
    parser.add_argument("--min-down-factor", type=float, default=1.0)
    parser.add_argument("--context-points", type=int, default=150)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
