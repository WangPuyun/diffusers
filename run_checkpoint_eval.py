"""Driver: iterate over the 15 LoRA checkpoints in outputs/v1 and collect metrics.

For each checkpoint, runs test_multi_gpu.py with --lora-path pointing at the
checkpoint's pytorch_lora_weights.safetensors, parses LPIPS / FID / ArcFace from
stdout, and writes results to CSV + Markdown next to this script.
"""

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/root/autodl-tmp/diffusers")
CHECKPOINT_ROOT = REPO_ROOT / "outputs" / "v1"
TEST_SCRIPT = REPO_ROOT / "test_multi_gpu.py"
LOG_DIR = REPO_ROOT / "outputs" / "v1" / "eval_logs"
RESULTS_CSV = CHECKPOINT_ROOT / "eval_results.csv"
RESULTS_MD = CHECKPOINT_ROOT / "eval_results.md"

CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900,
               1000, 1100, 1200, 1300, 1400, 1500]

METRIC_RE = {
    "LPIPS":   re.compile(r"^LPIPS\s+([0-9.]+)\s*$", re.MULTILINE),
    "FID":     re.compile(r"^FID\s+([0-9.]+)\s*$", re.MULTILINE),
    "ArcFace": re.compile(r"^ArcFace\s+([0-9.]+)\s*$", re.MULTILINE),
}


def parse_metrics(stdout: str):
    metrics = {}
    for name, pattern in METRIC_RE.items():
        m = pattern.search(stdout)
        metrics[name] = float(m.group(1)) if m else None
    return metrics


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for step in CHECKPOINTS:
        ckpt_dir = CHECKPOINT_ROOT / f"checkpoint-{step}"
        lora_path = ckpt_dir / "pytorch_lora_weights.safetensors"
        log_path = LOG_DIR / f"checkpoint-{step}.log"

        if not lora_path.is_file():
            print(f"[checkpoint-{step}] MISSING {lora_path}", flush=True)
            rows.append({"checkpoint": step, "LPIPS": None, "FID": None,
                         "ArcFace": None, "status": "missing",
                         "elapsed_s": 0})
            continue

        print(f"\n=== checkpoint-{step} ({lora_path}) ===", flush=True)
        t0 = time.time()
        status = "ok"
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    [sys.executable, str(TEST_SCRIPT), "--lora-path", str(lora_path)],
                    cwd=str(REPO_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                status = f"exit={proc.returncode}"

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = parse_metrics(log_text)
        except Exception as exc:
            metrics = {"LPIPS": None, "FID": None, "ArcFace": None}
            status = f"exc={type(exc).__name__}"
            log_text = ""

        elapsed = time.time() - t0
        row = {
            "checkpoint": step,
            "LPIPS": metrics["LPIPS"],
            "FID": metrics["FID"],
            "ArcFace": metrics["ArcFace"],
            "status": status,
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"[checkpoint-{step}] status={status} "
            f"LPIPS={row['LPIPS']} FID={row['FID']} ArcFace={row['ArcFace']} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        write_results(rows)

    write_results(rows)
    print(f"\nWrote {RESULTS_CSV} and {RESULTS_MD}", flush=True)


def write_results(rows):
    fieldnames = ["checkpoint", "LPIPS", "FID", "ArcFace", "status", "elapsed_s"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else (v if v is not None else "—")

    lines = [
        "# Checkpoint Evaluation Results",
        "",
        "| Checkpoint | LPIPS ↓ | FID ↓ | ArcFace ↑ | Status | Elapsed (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["checkpoint"]):
        lines.append(
            f"| {r['checkpoint']} | {fmt(r['LPIPS'])} | {fmt(r['FID'])} | "
            f"{fmt(r['ArcFace'])} | {r['status']} | {r['elapsed_s']} |"
        )
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
