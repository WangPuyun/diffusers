#!/usr/bin/env python

import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
STUDY_PATH = REPO_ROOT / "experiments/flux2_klein_loss_search/study.json"
CANDIDATE_PATH = REPO_ROOT / "experiments/flux2_klein_loss_search/candidate.json"
RESULT_PATH = REPO_ROOT / "outputs/autoresearch_loss_search/latest_result.json"
RESULTS_TSV = RUN_DIR / "classic-results.tsv"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_sequence():
    yield 0, "baseline_mse", 0.0, 0.0
    for index, cosine in enumerate((0.0625, 0.125, 0.25, 0.5), start=1):
        yield index, f"cosine_only_{cosine:g}", cosine, 0.0
    for index, hf in enumerate((0.0625, 0.125, 0.25, 0.5), start=5):
        yield index, f"hf_only_{hf:g}", 0.0, hf
    index = 9
    for cosine in (0.0625, 0.125, 0.25, 0.5):
        for hf in (0.0625, 0.125, 0.25, 0.5):
            yield index, f"cosine_{cosine:g}_hf_{hf:g}", cosine, hf
            index += 1


def append_result(iteration, label, cosine, hf, result):
    metrics = result["metrics"]
    face = result["face_detection"]
    row = [
        iteration,
        label,
        cosine,
        hf,
        result["score"],
        metrics["lpips"],
        metrics["arcface"],
        metrics["ssim"],
        metrics["psnr_db"],
        metrics["fid"],
        metrics["kid_raw"],
        face["pair_count"],
        face["pair_rate"],
        result["fingerprint"],
        result["status"],
    ]
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(value) for value in row) + "\n")


def run_command(command, log_path):
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def summarize(iteration):
    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]
    successes = [row for row in rows if row["status"] == "success"]
    best = min(successes, key=lambda row: float(row["score_J"])) if successes else None
    summary = {
        "iteration": iteration,
        "completed": len(rows),
        "successes": len(successes),
        "best": best,
    }
    write_json(RUN_DIR / f"iter{iteration}-evals.json", summary)


def main():
    if not RESULTS_TSV.is_file():
        RESULTS_TSV.write_text(
            "iteration\tlabel\tcosine_weight\thf_weight\tscore_J\tlpips\tarcface\tssim\tpsnr_db\tfid\tkid_raw\tface_pair_count\tface_pair_rate\tfingerprint\tstatus\n",
            encoding="utf-8",
        )

    completed_iterations = {
        int(line.split("\t", 1)[0])
        for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    }

    single_item_results = {}
    for iteration, label, cosine, hf in candidate_sequence():
        if iteration in completed_iterations:
            continue
        write_json(
            CANDIDATE_PATH,
            {"cosine_weight": cosine, "hf_weight": hf, "mse_weight": 1.0},
        )
        verify_log = RUN_DIR / f"iter{iteration:02d}-{label}-verify.log"
        verify_command = [
            sys.executable,
            "experiments/flux2_klein_loss_search/run_trial.py",
            "--study",
            "experiments/flux2_klein_loss_search/study.json",
            "--candidate",
            "experiments/flux2_klein_loss_search/candidate.json",
        ]
        while run_command(verify_command, verify_log) != 0:
            log_text = verify_log.read_text(encoding="utf-8", errors="replace")
            if "Trial lock already exists" not in log_text:
                raise RuntimeError(f"iteration {iteration} verify failed; see {verify_log}")
            time.sleep(300)

        guard_log = RUN_DIR / f"iter{iteration:02d}-{label}-guard.log"
        guard_command = [
            sys.executable,
            "experiments/flux2_klein_loss_search/guard_trial.py",
            "--study",
            "experiments/flux2_klein_loss_search/study.json",
            "--candidate",
            "experiments/flux2_klein_loss_search/candidate.json",
            "--result",
            "outputs/autoresearch_loss_search/latest_result.json",
        ]
        if run_command(guard_command, guard_log) != 0:
            raise RuntimeError(f"iteration {iteration} guard failed; see {guard_log}")

        result = read_json(RESULT_PATH)
        append_result(iteration, label, cosine, hf, result)
        if cosine > 0.0 and hf == 0.0:
            single_item_results[("cosine", cosine)] = result["score"]
        if hf > 0.0 and cosine == 0.0:
            single_item_results[("hf", hf)] = result["score"]
        if iteration in {8, 16, 24}:
            summarize(iteration)

    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]
    cosine_only = [row for row in rows if float(row["cosine_weight"]) > 0.0 and float(row["hf_weight"]) == 0.0]
    hf_only = [row for row in rows if float(row["hf_weight"]) > 0.0 and float(row["cosine_weight"]) == 0.0]
    best_cosine = min(cosine_only, key=lambda row: float(row["score_J"]))
    best_hf = min(hf_only, key=lambda row: float(row["score_J"]))
    if float(best_cosine["score_J"]) <= float(best_hf["score_J"]):
        final_label = "final_cosine_only_1"
        final_cosine, final_hf = 1.0, 0.0
    else:
        final_label = "final_hf_only_1"
        final_cosine, final_hf = 0.0, 1.0

    if 25 not in {int(row["iteration"]) for row in rows}:
        write_json(
            CANDIDATE_PATH,
            {"cosine_weight": final_cosine, "hf_weight": final_hf, "mse_weight": 1.0},
        )
        verify_log = RUN_DIR / f"iter25-{final_label}-verify.log"
        if run_command(
            [
                sys.executable,
                "experiments/flux2_klein_loss_search/run_trial.py",
                "--study",
                "experiments/flux2_klein_loss_search/study.json",
                "--candidate",
                "experiments/flux2_klein_loss_search/candidate.json",
            ],
            verify_log,
        ) != 0:
            raise RuntimeError(f"iteration 25 verify failed; see {verify_log}")
        guard_log = RUN_DIR / f"iter25-{final_label}-guard.log"
        if run_command(
            [
                sys.executable,
                "experiments/flux2_klein_loss_search/guard_trial.py",
                "--study",
                "experiments/flux2_klein_loss_search/study.json",
                "--candidate",
                "experiments/flux2_klein_loss_search/candidate.json",
                "--result",
                "outputs/autoresearch_loss_search/latest_result.json",
            ],
            guard_log,
        ) != 0:
            raise RuntimeError(f"iteration 25 guard failed; see {guard_log}")
        append_result(25, final_label, final_cosine, final_hf, read_json(RESULT_PATH))

    summarize(25)
    rows = [dict(zip(header, line.split("\t"))) for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]]
    best = min(rows, key=lambda row: float(row["score_J"]))
    write_json(RUN_DIR / "handoff.json", {"status": "complete", "best": best, "results": str(RESULTS_TSV)})


if __name__ == "__main__":
    main()
