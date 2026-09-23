# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
import text_codec as config  # Dataset tools do not need to import PyTorch.


PUNCTUATION = set("„“.,!?-;:()")


def trajectory_id(path):
    match = re.fullmatch(r"trajectory_(\d+)\.[^.]+", path.name)
    return int(match.group(1)) if match else None


def sorted_trajectory_files(directory, suffix):
    paths = []
    for path in Path(directory).glob(f"trajectory_*.{suffix}"):
        item_id = trajectory_id(path)
        if item_id is not None:
            paths.append((item_id, path))
    return dict(sorted(paths, key=lambda item: item[0]))


def read_texts(text_dir):
    texts = {}
    for item_id, path in sorted_trajectory_files(text_dir, "txt").items():
        texts[item_id] = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return texts


def read_json_lengths(json_dir):
    lengths = {}
    issues = []
    for item_id, path in sorted_trajectory_files(json_dir, "json").items():
        try:
            with path.open("r", encoding="utf-8") as f:
                trajectory = json.load(f)
            points = np.asarray(trajectory, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 3:
                issues.append(f"{path.name}: expected shape (T, 3), got {points.shape}")
                continue
            lengths[item_id] = int(points.shape[0])
        except Exception as exc:
            issues.append(f"{path.name}: {exc}")
    return lengths, issues


def as_array_list(value, dtype):
    if value.dtype == np.object_:
        return [np.asarray(item, dtype=dtype) for item in value]
    return [np.asarray(value, dtype=dtype)]


def load_npz_summary(path):
    if not Path(path).exists():
        return None

    data = np.load(path, allow_pickle=True)
    points = as_array_list(data["points"], np.float32)
    text_indices = as_array_list(data["text_indices"], np.int64)
    eos_index = config.char_to_idx.get(getattr(config, "eos_token", None))
    eos_counts = []
    if eos_index is not None:
        eos_counts = [int((item.reshape(-1) == eos_index).sum()) for item in text_indices]
    return {
        "samples": len(points),
        "point_lengths": [int(item.shape[0]) for item in points],
        "text_lengths": [int(item.reshape(-1).shape[0]) for item in text_indices],
        "eos_tokens": sum(eos_counts),
        "samples_with_eos": sum(1 for count in eos_counts if count > 0),
    }


def strip_one_final_newline(text):
    return text[:-1] if text.endswith("\n") else text


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def basic_stats(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "max": 0,
            "mean": 0.0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": float(np.mean(values)),
    }


def format_stats(stats):
    return (
        f"count={stats['count']}, min={stats['min']}, p25={stats['p25']:.1f}, "
        f"median={stats['median']:.1f}, p75={stats['p75']:.1f}, p90={stats['p90']:.1f}, "
        f"p95={stats['p95']:.1f}, max={stats['max']}, mean={stats['mean']:.1f}"
    )


def bucket_counts(values, buckets):
    result = []
    for label, lo, hi in buckets:
        count = sum(1 for value in values if value >= lo and (hi is None or value <= hi))
        result.append((label, count))
    return result


def char_label(ch):
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "<space>"
    if ch == "\t":
        return "\\t"
    return ch


def md_escape(text):
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def text_preview(text, limit=70):
    preview = text.replace("\n", "\\n")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 1] + "…"


def count_categories(texts):
    total = len(texts)
    content_texts = [strip_one_final_newline(text) for text in texts]
    return {
        "samples": total,
        "with_space": sum(" " in text for text in content_texts),
        "with_punctuation": sum(any(ch in PUNCTUATION for ch in text) for text in content_texts),
        "with_digits": sum(any(ch.isdigit() for ch in text) for text in content_texts),
        "with_uppercase": sum(any(ch.isupper() for ch in text) for text in content_texts),
        "ending_newline": sum(text.endswith("\n") for text in texts),
        "without_ending_newline": sum(not text.endswith("\n") for text in texts),
        "internal_newline": sum("\n" in text[:-1] for text in texts),
        "leading_whitespace": sum(bool(text[:1] and text[0].isspace()) for text in texts),
        "trailing_whitespace_not_newline": sum(
            bool(text and text[-1].isspace() and text[-1] != "\n") for text in texts
        ),
    }


def make_recommendations(
    unsupported,
    low_frequency,
    unused_chars,
    categories,
    content_lengths,
    point_lengths,
    max_seq_len,
):
    recommendations = []
    sample_count = categories["samples"]
    with_space = categories["with_space"]
    with_punctuation = categories["with_punctuation"]
    with_digits = categories["with_digits"]
    with_uppercase = categories["with_uppercase"]

    if unsupported:
        recommendations.append(
            "Есть символы вне CHAR_SET. Сейчас converter.py заменяет их пробелами, поэтому такие подписи лучше очистить "
            "или расширить CHAR_SET."
        )

    if sample_count and with_space / sample_count < 0.1:
        recommendations.append(
            "Почти нет примеров с пробелами. Для предложений нужно собирать строки из нескольких слов."
        )

    if sample_count and with_punctuation / sample_count < 0.1:
        recommendations.append(
            "Пунктуация представлена слабо. Для фраз и абзацев нужны отдельные примеры с запятыми, точками и вопросами."
        )

    if sample_count and with_digits == 0:
        recommendations.append("Цифры есть в CHAR_SET, но в текущих подписях не встречаются.")

    if sample_count and with_uppercase / sample_count < 0.05:
        recommendations.append("Заглавные буквы представлены слабо; модель будет писать их нестабильно.")

    if content_lengths and max(content_lengths) < 20:
        recommendations.append(
            "Все подписи короткие. Модель можно проверять на словах, но для строк нужны примеры длиной 20-50 символов."
        )

    truncated_count = sum(length > max_seq_len for length in point_lengths)
    if truncated_count:
        recommendations.append(
            f"{truncated_count} примеров длиннее max_seq_len={max_seq_len}; при обучении они будут обрезаны."
        )

    if low_frequency:
        recommendations.append(
            "Есть символы с низкой частотой. Для надежной генерации лучше довести важные символы хотя бы до десятков "
            "вхождений, а лучше выше."
        )

    if len(unused_chars) > 10:
        recommendations.append(
            "Большая часть CHAR_SET не используется в текущем датасете. Это нормально для узкой задачи слов, но мало "
            "для универсального текста."
        )

    return recommendations


def render_report(args, texts_by_id, point_lengths_by_id, json_issues, npz_summary):
    ids = sorted(set(texts_by_id) | set(point_lengths_by_id))
    paired_ids = sorted(set(texts_by_id) & set(point_lengths_by_id))
    missing_text = sorted(set(point_lengths_by_id) - set(texts_by_id))
    missing_json = sorted(set(texts_by_id) - set(point_lengths_by_id))

    texts = [texts_by_id[item_id] for item_id in paired_ids]
    content_texts = [strip_one_final_newline(text) for text in texts]
    raw_text_lengths = [len(text) for text in texts]
    content_lengths = [len(text) for text in content_texts]
    point_lengths = [point_lengths_by_id[item_id] for item_id in paired_ids]
    points_per_char = [
        point_lengths_by_id[item_id] / max(1, len(strip_one_final_newline(texts_by_id[item_id])))
        for item_id in paired_ids
    ]

    char_counts = Counter()
    unsupported = Counter()
    charset = set(config.CHAR_SET)
    for text in texts:
        for ch in text:
            char_counts[ch] += 1
            if ch not in charset:
                unsupported[ch] += 1

    used_supported = {ch for ch in char_counts if ch in charset}
    unused_chars = [ch for ch in config.CHAR_SET if ch not in used_supported]
    low_frequency = [
        (ch, char_counts[ch])
        for ch in config.CHAR_SET
        if 0 < char_counts[ch] < args.low_count
    ]

    label_counts = Counter(content_texts)
    repeated_labels = [(label, count) for label, count in label_counts.most_common() if count > 1]
    categories = count_categories(texts)
    recommendations = make_recommendations(
        unsupported=unsupported,
        low_frequency=low_frequency,
        unused_chars=unused_chars,
        categories=categories,
        content_lengths=content_lengths,
        point_lengths=point_lengths,
        max_seq_len=args.max_seq_len,
    )

    lines = [
        "# Аудит датасета",
        "",
        f"- Датасет: `{Path(args.dataset).resolve()}`",
        f"- JSON: `{Path(args.json_dir).resolve()}`",
        f"- TXT: `{Path(args.text_dir).resolve()}`",
        f"- CHAR_SET size: `{len(config.CHAR_SET)}`",
        f"- VOCAB size: `{config.vocab_size}`",
        f"- EOS token: `{getattr(config, 'eos_token', '<none>')}`",
        f"- max_seq_len для проверки обрезания: `{args.max_seq_len}`",
        "",
        "## Общая сводка",
        "",
        f"- JSON-файлов: `{len(point_lengths_by_id)}`",
        f"- TXT-файлов: `{len(texts_by_id)}`",
        f"- Пар JSON+TXT: `{len(paired_ids)}`",
        f"- Всего точек пера в paired JSON: `{sum(point_lengths)}`",
        f"- Всего символов в paired TXT: `{sum(raw_text_lengths)}`",
        f"- Уникальных символов в paired TXT: `{len(char_counts)}`",
        f"- Символов из CHAR_SET, не встретившихся в датасете: `{len(unused_chars)}`",
        f"- Символов вне CHAR_SET: `{sum(unsupported.values())}`",
        "",
    ]

    if npz_summary is not None:
        lines.extend(
            [
                "## Сводка all_trajectories.npz",
                "",
                f"- Samples: `{npz_summary['samples']}`",
                f"- Длины траекторий: `{format_stats(basic_stats(npz_summary['point_lengths']))}`",
                f"- Длины закодированного текста: `{format_stats(basic_stats(npz_summary['text_lengths']))}`",
                f"- EOS-токенов в закодированном тексте: `{npz_summary['eos_tokens']}`",
                f"- Samples с EOS: `{npz_summary['samples_with_eos']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Длины",
            "",
            f"- Текст, сырой TXT: `{format_stats(basic_stats(raw_text_lengths))}`",
            f"- Текст без одного финального `\\n`: `{format_stats(basic_stats(content_lengths))}`",
            f"- Точки траектории: `{format_stats(basic_stats(point_lengths))}`",
            f"- Точек на символ: `{format_stats(basic_stats(points_per_char))}`",
            "",
            "### Бакеты по длине текста",
            "",
            "| Длина без финального `\\n` | Примеров |",
            "| --- | ---: |",
        ]
    )

    for label, count in bucket_counts(
        content_lengths,
        [
            ("0-5", 0, 5),
            ("6-10", 6, 10),
            ("11-20", 11, 20),
            ("21-40", 21, 40),
            ("41-80", 41, 80),
            ("81+", 81, None),
        ],
    ):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "### Бакеты по длине траектории",
            "",
            "| Точек пера | Примеров |",
            "| --- | ---: |",
        ]
    )
    for label, count in bucket_counts(
        point_lengths,
        [
            ("0-500", 0, 500),
            ("501-1000", 501, 1000),
            ("1001-1500", 1001, 1500),
            ("1501-2000", 1501, 2000),
            ("2001-3000", 2001, 3000),
            ("3001+", 3001, None),
        ],
    ):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Типы подписей",
            "",
            f"- С пробелами: `{categories['with_space']}`",
            f"- С пунктуацией: `{categories['with_punctuation']}`",
            f"- С цифрами: `{categories['with_digits']}`",
            f"- С заглавными буквами: `{categories['with_uppercase']}`",
            f"- Заканчиваются `\\n`: `{categories['ending_newline']}`",
            f"- Не заканчиваются `\\n`: `{categories['without_ending_newline']}`",
            f"- Есть внутренний `\\n`: `{categories['internal_newline']}`",
            f"- Начинаются с whitespace: `{categories['leading_whitespace']}`",
            f"- Заканчиваются whitespace, но не `\\n`: `{categories['trailing_whitespace_not_newline']}`",
            "",
            "## Покрытие символов",
            "",
            "Здесь считается сырой текст из `dataset/texts/`, без автоматически добавляемого `<EOS>`.",
            "",
        ]
    )

    if unsupported:
        lines.extend(
            [
                "### Символы вне CHAR_SET",
                "",
                "| Символ | Количество |",
                "| --- | ---: |",
            ]
        )
        for ch, count in unsupported.most_common():
            lines.append(f"| `{md_escape(char_label(ch))}` | {count} |")
        lines.append("")
    else:
        lines.extend(["- Символов вне CHAR_SET нет.", ""])

    lines.extend(
        [
            f"### Низкочастотные символы `< {args.low_count}`",
            "",
        ]
    )
    if low_frequency:
        lines.extend(["| Символ | Количество |", "| --- | ---: |"])
        for ch, count in low_frequency:
            lines.append(f"| `{md_escape(char_label(ch))}` | {count} |")
    else:
        lines.append("- Низкочастотных использованных символов нет.")
    lines.append("")

    lines.extend(
        [
            "### Неиспользованные символы из CHAR_SET",
            "",
            "`" + " ".join(md_escape(char_label(ch)) for ch in unused_chars) + "`",
            "",
            "### Полная таблица символов",
            "",
            "| Символ | Количество |",
            "| --- | ---: |",
        ]
    )
    for ch in config.CHAR_SET:
        lines.append(f"| `{md_escape(char_label(ch))}` | {char_counts[ch]} |")

    if repeated_labels:
        lines.extend(
            [
                "",
                "## Повторяющиеся подписи",
                "",
                "| Подпись | Количество |",
                "| --- | ---: |",
            ]
        )
        for label, count in repeated_labels[:30]:
            lines.append(f"| `{md_escape(text_preview(label))}` | {count} |")

    longest_ids = sorted(paired_ids, key=lambda item_id: point_lengths_by_id[item_id], reverse=True)[:15]
    lines.extend(
        [
            "",
            "## Самые длинные траектории",
            "",
            "| ID | Точек | Символов | Текст |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for item_id in longest_ids:
        text = strip_one_final_newline(texts_by_id[item_id])
        lines.append(
            f"| {item_id} | {point_lengths_by_id[item_id]} | {len(text)} | "
            f"`{md_escape(text_preview(text))}` |"
        )

    longest_text_ids = sorted(
        paired_ids,
        key=lambda item_id: len(strip_one_final_newline(texts_by_id[item_id])),
        reverse=True,
    )[:15]
    lines.extend(
        [
            "",
            "## Самые длинные подписи",
            "",
            "| ID | Символов | Точек | Текст |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for item_id in longest_text_ids:
        text = strip_one_final_newline(texts_by_id[item_id])
        lines.append(
            f"| {item_id} | {len(text)} | {point_lengths_by_id[item_id]} | "
            f"`{md_escape(text_preview(text))}` |"
        )

    if missing_text or missing_json or json_issues:
        lines.extend(["", "## Проблемы с файлами", ""])
        if missing_text:
            lines.append("- Нет TXT для JSON: `" + ", ".join(map(str, missing_text[:100])) + "`")
        if missing_json:
            lines.append("- Нет JSON для TXT: `" + ", ".join(map(str, missing_json[:100])) + "`")
        for issue in json_issues[:100]:
            lines.append(f"- {issue}")

    lines.extend(["", "## Выводы", ""])
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- Критичных проблем по покрытию символов и длинам не найдено.")

    return "\n".join(lines) + "\n", {
        "paired": len(paired_ids),
        "text_chars": sum(raw_text_lengths),
        "unique_chars": len(char_counts),
        "unsupported_chars": sum(unsupported.values()),
        "unused_charset": len(unused_chars),
        "low_frequency": len(low_frequency),
        "max_text_length": max(content_lengths) if content_lengths else 0,
        "max_points": max(point_lengths) if point_lengths else 0,
        "with_space": categories["with_space"],
        "with_punctuation": categories["with_punctuation"],
        "with_digits": categories["with_digits"],
        "with_uppercase": categories["with_uppercase"],
        "truncated": sum(length > args.max_seq_len for length in point_lengths),
        "recommendations": len(recommendations),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Audit handwriting dataset character coverage and sequence lengths.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "dataset" / "all_trajectories.npz"))
    parser.add_argument("--json-dir", default=str(PROJECT_ROOT / "dataset" / "jsons"))
    parser.add_argument("--text-dir", default=str(PROJECT_ROOT / "dataset" / "texts"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "dataset_audit.md"))
    parser.add_argument("--low-count", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=getattr(config, "max_seq_len", 3000))
    return parser.parse_args()


def main():
    args = parse_args()
    texts_by_id = read_texts(args.text_dir)
    point_lengths_by_id, json_issues = read_json_lengths(args.json_dir)
    npz_summary = load_npz_summary(args.dataset)
    report, summary = render_report(args, texts_by_id, point_lengths_by_id, json_issues, npz_summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print("Dataset audit")
    print("report:", output)
    print("paired_samples:", summary["paired"])
    print("text_chars:", summary["text_chars"])
    print("unique_chars:", summary["unique_chars"])
    print("unsupported_chars:", summary["unsupported_chars"])
    print("unused_charset_chars:", summary["unused_charset"])
    print("low_frequency_chars:", summary["low_frequency"])
    print("max_text_length:", summary["max_text_length"])
    print("max_points:", summary["max_points"])
    print("with_space:", summary["with_space"])
    print("with_punctuation:", summary["with_punctuation"])
    print("with_digits:", summary["with_digits"])
    print("with_uppercase:", summary["with_uppercase"])
    print("longer_than_max_seq_len:", summary["truncated"])
    print("recommendations:", summary["recommendations"])


if __name__ == "__main__":
    main()
