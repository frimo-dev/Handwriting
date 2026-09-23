# -*- coding: utf-8 -*-
"""One command-line entry point for the handwriting project."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_python(args, env=None):
    full_env = os.environ.copy()
    cache_dir = ROOT / "runs" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_env.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    full_env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    if env:
        full_env.update({key: str(value) for key, value in env.items() if value is not None})
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=full_env, check=True)


def latest_checkpoint(checkpoint_dir):
    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return max(paths, key=lambda item: item[0])[1] if paths else None


def cmd_status(args):
    sys.path.insert(0, str(SRC))
    import config

    data_path = Path(config.data_path)
    checkpoint_dir = Path(args.checkpoints or config.checkpoints)
    runs_dir = Path(config.runs_dir)
    latest = latest_checkpoint(checkpoint_dir)
    best = checkpoint_dir / "best.pth"

    print("Project:", ROOT)
    print("Python:", sys.executable)
    print("Dataset:", data_path, "OK" if data_path.exists() else "missing")
    print("Checkpoints:", checkpoint_dir, "OK" if checkpoint_dir.exists() else "missing")
    print("Latest checkpoint:", latest if latest else "missing")
    print("Best checkpoint:", best if best.exists() else "missing")
    print("Runs:", runs_dir)
    print("Target words:", ROOT / "dataset" / "target_texts.txt")

    if data_path.exists():
        try:
            import numpy as np

            data = np.load(data_path, allow_pickle=True)
            points = data["points"]
            lengths = [len(item) for item in points]
            print("Samples:", len(lengths), "points:", sum(lengths))
        except Exception as exc:
            print("Dataset summary failed:", exc)


def cmd_dataset(args):
    run_python(["dataset/scripts/converter.py", "--dataset-dir", args.dataset_dir])


def cmd_label_dataset(args):
    command = ["dataset/scripts/label_trajectories.py", "--dataset-dir", args.dataset_dir]
    command.append("--edit" if args.edit else "--review")
    if args.start is not None:
        command.extend(["--start", str(args.start)])
    run_python(command)


def cmd_record_dataset(args):
    run_python(["dataset/scripts/record_trajectories.py"])


def cmd_audit_dataset(args):
    command = [
        "dataset/scripts/audit_dataset.py",
        "--dataset",
        args.dataset,
        "--json-dir",
        args.json_dir,
        "--text-dir",
        args.text_dir,
        "--output",
        args.output,
        "--low-count",
        str(args.low_count),
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    run_python(command)


def cmd_compare_spacing(args):
    command = [
        "dataset/scripts/compare_line_spacing.py",
        "--sample",
        args.sample,
        "--json-dir",
        args.json_dir,
        "--text-dir",
        args.text_dir,
        "--expected-lines",
        str(args.expected_lines),
        "--try-mm",
        args.try_mm,
    ]
    if args.meta:
        command.extend(["--meta", args.meta])
    run_python(command)


def cmd_train(args):
    env = {
        "CHECKPOINTS": args.checkpoints,
        "EPOCHS": args.epochs,
        "MORE_EPOCHS": args.more_epochs,
        "BATCH_SIZE": args.batch_size,
        "GRAD_ACCUM_STEPS": args.grad_accum_steps,
        "LR": args.lr,
        "NUM_WORKERS": args.num_workers,
        "BUCKET_BY_LENGTH": str(args.bucket_by_length).lower(),
        "BUCKET_SIZE_MULTIPLIER": args.bucket_size_multiplier,
        "SAVE_EVERY": args.save_every,
        "SAVE_TEMPORARY_EACH_EPOCH": str(args.temporary_checkpoints).lower(),
        "MAX_SEQ_LEN": args.max_seq_len,
        "VALIDATION_SPLIT": args.validation_split,
        "EVAL_EVERY": args.eval_every,
        "PROGRESS_MODE": args.progress_mode,
        "AUTO_RESUME": str(args.auto_resume).lower(),
        "RESUME": args.resume,
        "TEACHER_FORCING_RATIO": args.teacher_forcing_ratio,
        "SCHEDULED_SAMPLING_MODE": args.scheduled_sampling_mode,
    }
    if not args.rebuild_dataset:
        dataset_path = ROOT / "dataset" / "all_trajectories.npz"
        if not dataset_path.exists():
            print("Dataset is missing; rebuilding first.")
            cmd_dataset(args)
        run_python(["scripts/server_train.py"], env=env)
    else:
        cmd_dataset(args)
        run_python(["scripts/server_train.py"], env=env)


def cmd_evaluate(args):
    command = ["src/evaluate_checkpoint.py", "--checkpoints", args.checkpoints]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.output:
        command.extend(["--output", args.output])
    run_python(command)


def cmd_diagnose_checkpoint(args):
    command = [
        "src/diagnose_checkpoint.py",
        "--checkpoints",
        args.checkpoints,
        "--sample-ids",
        args.sample_ids,
        "--texts",
        args.texts,
        "--output-dir",
        args.output_dir,
        "--sampling-mode",
        args.sampling_mode,
        "--stop-strategy",
        args.stop_strategy,
        "--min-len-per-char",
        str(args.min_len_per_char),
        "--max-len-per-char",
        str(args.max_len_per_char),
        "--bias",
        str(args.bias),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    run_python(command)


def cmd_overfit_words(args):
    command = [
        "src/overfit_words.py",
        "--checkpoints",
        args.checkpoints,
        "--checkpoint",
        args.checkpoint,
        "--dataset",
        args.dataset,
        "--json-dir",
        args.json_dir,
        "--text-dir",
        args.text_dir,
        "--words-file",
        args.words_file,
        "--sample-ids",
        args.sample_ids,
        "--match",
        args.match,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--teacher-forcing-ratio",
        str(args.teacher_forcing_ratio),
        "--scheduled-sampling-mode",
        args.scheduled_sampling_mode,
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--output-dir",
        args.output_dir,
        "--checkpoint-output-dir",
        args.checkpoint_output_dir,
        "--biases",
        args.biases,
        "--sampling-modes",
        args.sampling_modes,
        "--min-len-per-char",
        str(args.min_len_per_char),
        "--max-len-per-char",
        str(args.max_len_per_char),
        "--stop-strategy",
        args.stop_strategy,
    ]
    if args.words:
        command.append("--words")
        command.extend(args.words)
    if args.require_all_words:
        command.append("--require-all-words")
    run_python(command)


def cmd_profile_train(args):
    command = [
        "src/profile_training.py",
        "--checkpoints",
        args.checkpoints,
        "--batches",
        str(args.batches),
        "--warmup",
        str(args.warmup),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-len",
        str(args.max_seq_len),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.bucket_by_length:
        command.append("--bucket-by-length")
    else:
        command.append("--no-bucket-by-length")
    command.extend(["--bucket-size-multiplier", str(args.bucket_size_multiplier)])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.no_load_checkpoint:
        command.append("--no-load-checkpoint")
    run_python(command)


def cmd_generate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_text = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", args.text, flags=re.UNICODE).strip("_") or "text"
    stem = args.name or safe_text.lower()
    env = {
        "CHECKPOINTS": args.checkpoints,
        "TEXT": args.text,
        "BIAS": args.bias,
        "MIN_GEN_LEN": args.min_len,
        "MAX_GEN_LEN": args.max_len,
        "STOP_STRATEGY": args.stop_strategy,
        "SAMPLING_MODE": args.sampling_mode,
        "APPEND_EOS": str(args.append_eos).lower(),
        "OUTPUT_JSON": output_dir / f"{stem}.json",
        "OUTPUT_PNG": output_dir / f"{stem}.png",
        "OUTPUT_META": output_dir / f"{stem}.meta.json",
    }
    run_python(["scripts/server_generate.py"], env=env)


def cmd_candidates(args):
    command = [
        "src/generate_candidates.py",
        "--checkpoints",
        args.checkpoints,
        "--texts",
        args.texts,
        "--output-dir",
        args.output_dir,
        "--biases",
        args.biases,
        "--variants",
        str(args.variants),
        "--base-seed",
        str(args.base_seed),
        "--min-len-per-char",
        str(args.min_len_per_char),
        "--max-len-per-char",
        str(args.max_len_per_char),
        "--sampling-mode",
        args.sampling_mode,
        "--stop-strategy",
        args.stop_strategy,
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    command.append("--append-eos" if args.append_eos else "--no-append-eos")
    run_python(command)


def cmd_clean(args):
    paths = [
        ROOT / "generated_trajectory.json",
        ROOT / "generated_trajectory.png",
        ROOT / "generated_trajectory.meta.json",
        ROOT / "src" / "generated_trajectory.json",
        ROOT / "src" / "generated_variants",
        ROOT / "src" / "generation_eval",
        ROOT / "candidate_runs",
    ]
    if args.include_runs:
        paths.append(ROOT / "runs")

    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print("Removed:", path.relative_to(ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description="Handwriting synthesis project helper.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoints", default="checkpoints_attention")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", parents=[common], help="Show project, dataset and checkpoint status.")
    status.set_defaults(func=cmd_status)

    dataset = subparsers.add_parser("dataset", help="Rebuild dataset/all_trajectories.npz.")
    dataset.add_argument("--dataset-dir", default="dataset")
    dataset.set_defaults(func=cmd_dataset)

    label = subparsers.add_parser("label-dataset", help="View trajectories or edit their text labels.")
    label.add_argument("--dataset-dir", default="dataset")
    label.add_argument("--start", type=int)
    label.add_argument("--edit", action="store_true", help="Enable label editing; default is read-only.")
    label.set_defaults(func=cmd_label_dataset)

    record = subparsers.add_parser("record-dataset", help="Open the mouse/pen trajectory recorder.")
    record.set_defaults(func=cmd_record_dataset)

    audit = subparsers.add_parser("audit-dataset", help="Audit dataset character coverage and sequence lengths.")
    audit.add_argument("--dataset", default="dataset/all_trajectories.npz")
    audit.add_argument("--json-dir", default="dataset/jsons")
    audit.add_argument("--text-dir", default="dataset/texts")
    audit.add_argument("--output", default="runs/dataset_audit.md")
    audit.add_argument("--low-count", type=int, default=10)
    audit.add_argument("--max-seq-len", type=int, default=3000)
    audit.set_defaults(func=cmd_audit_dataset)

    spacing = subparsers.add_parser("compare-spacing", help="Compare a new multi-line sample with dataset vertical stats.")
    spacing.add_argument("--sample", required=True)
    spacing.add_argument("--meta", default="")
    spacing.add_argument("--json-dir", default="dataset/jsons")
    spacing.add_argument("--text-dir", default="dataset/texts")
    spacing.add_argument("--expected-lines", type=int, default=2)
    spacing.add_argument("--try-mm", default="8,9,10,11,12")
    spacing.set_defaults(func=cmd_compare_spacing)

    train = subparsers.add_parser("train", parents=[common], help="Continue model training.")
    train.add_argument("--epochs", default="")
    train.add_argument("--more-epochs", default="20")
    train.add_argument("--batch-size", default="")
    train.add_argument("--grad-accum-steps", default="")
    train.add_argument("--lr", default="")
    train.add_argument("--num-workers", default="0")
    train.add_argument("--bucket-by-length", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--bucket-size-multiplier", default="")
    train.add_argument("--save-every", default="")
    train.add_argument("--temporary-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--max-seq-len", default="")
    train.add_argument("--validation-split", default="")
    train.add_argument("--eval-every", default="")
    train.add_argument("--progress-mode", default="live")
    train.add_argument("--resume", default="")
    train.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--teacher-forcing-ratio", default="")
    train.add_argument("--scheduled-sampling-mode", choices=["argmax", "mean", "sample"], default="argmax")
    train.add_argument("--rebuild-dataset", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", parents=[common], help="Evaluate a checkpoint on validation split.")
    evaluate.add_argument("--checkpoint", default="")
    evaluate.add_argument("--output", default="")
    evaluate.set_defaults(func=cmd_evaluate)

    diagnose = subparsers.add_parser(
        "diagnose-checkpoint",
        parents=[common],
        help="Render teacher-forced reconstructions and deterministic generations.",
    )
    diagnose.add_argument("--checkpoint", default="")
    diagnose.add_argument("--sample-ids", default="1,2,3")
    diagnose.add_argument("--texts", default="dataset/target_texts.txt")
    diagnose.add_argument("--output-dir", default="runs/diagnostics/latest")
    diagnose.add_argument("--sampling-mode", choices=["sample", "argmax", "mean"], default="argmax")
    diagnose.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    diagnose.add_argument("--min-len-per-char", type=int, default=45)
    diagnose.add_argument("--max-len-per-char", type=int, default=350)
    diagnose.add_argument("--bias", type=float, default=1.0)
    diagnose.set_defaults(func=cmd_diagnose_checkpoint)

    overfit = subparsers.add_parser(
        "overfit-words",
        parents=[common],
        help="Overfit on a few existing word samples and render free-run generations.",
    )
    overfit.add_argument("--checkpoint", default="")
    overfit.add_argument("--dataset", default="dataset/all_trajectories.npz")
    overfit.add_argument("--json-dir", default="dataset/jsons")
    overfit.add_argument("--text-dir", default="dataset/texts")
    overfit.add_argument("--words", nargs="*", default=None)
    overfit.add_argument("--words-file", default="dataset/target_texts.txt")
    overfit.add_argument("--sample-ids", default="")
    overfit.add_argument("--match", choices=["exact", "contains"], default="exact")
    overfit.add_argument("--require-all-words", action="store_true")
    overfit.add_argument("--epochs", type=int, default=200)
    overfit.add_argument("--batch-size", type=int, default=4)
    overfit.add_argument("--lr", type=float, default=0.00003)
    overfit.add_argument("--teacher-forcing-ratio", type=float, default=0.98)
    overfit.add_argument("--scheduled-sampling-mode", choices=["argmax", "mean", "sample"], default="mean")
    overfit.add_argument("--eval-every", type=int, default=25)
    overfit.add_argument("--save-every", type=int, default=50)
    overfit.add_argument("--output-dir", default="runs/overfit_words/latest")
    overfit.add_argument("--checkpoint-output-dir", default="checkpoints_overfit")
    overfit.add_argument("--biases", default="1.0")
    overfit.add_argument("--sampling-modes", default="argmax")
    overfit.add_argument("--min-len-per-char", type=int, default=45)
    overfit.add_argument("--max-len-per-char", type=int, default=350)
    overfit.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    overfit.set_defaults(func=cmd_overfit_words)

    profile = subparsers.add_parser("profile-train", parents=[common], help="Profile training bottlenecks on a few batches.")
    profile.add_argument("--checkpoint", default="")
    profile.add_argument("--batches", type=int, default=8)
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--batch-size", type=int, default=2)
    profile.add_argument("--max-seq-len", type=int, default=3000)
    profile.add_argument("--num-workers", type=int, default=0)
    profile.add_argument("--bucket-by-length", action=argparse.BooleanOptionalAction, default=True)
    profile.add_argument("--bucket-size-multiplier", type=int, default=50)
    profile.add_argument("--no-load-checkpoint", action="store_true")
    profile.set_defaults(func=cmd_profile_train)

    generate = subparsers.add_parser("generate", parents=[common], help="Generate one handwriting sample.")
    generate.add_argument("--text", required=True)
    generate.add_argument("--bias", default="1.25")
    generate.add_argument("--min-len", default="200")
    generate.add_argument("--max-len", default="3000")
    generate.add_argument("--output-dir", default="runs/single")
    generate.add_argument("--name", default="")
    generate.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    generate.add_argument("--sampling-mode", choices=["sample", "argmax", "mean"], default="sample")
    generate.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(func=cmd_generate)

    candidates = subparsers.add_parser("candidates", parents=[common], help="Generate and rank many candidates.")
    candidates.add_argument("--texts", default="dataset/target_texts.txt")
    candidates.add_argument("--checkpoint", default="")
    candidates.add_argument("--output-dir", default="runs/candidates/latest")
    candidates.add_argument("--biases", default="0.75,1.0,1.25,1.5,1.75")
    candidates.add_argument("--variants", type=int, default=8)
    candidates.add_argument("--base-seed", type=int, default=12345)
    candidates.add_argument("--min-len-per-char", type=int, default=45)
    candidates.add_argument("--max-len-per-char", type=int, default=350)
    candidates.add_argument("--sampling-mode", choices=["sample", "argmax", "mean"], default="sample")
    candidates.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    candidates.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True)
    candidates.set_defaults(func=cmd_candidates)

    clean = subparsers.add_parser("clean", help="Remove generated outputs outside source/data.")
    clean.add_argument("--include-runs", action="store_true")
    clean.set_defaults(func=cmd_clean)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
