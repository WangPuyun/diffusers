#!/usr/bin/env python

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from run_trial import read_json, sha256_file, snapshot_input_trees, write_json_atomic


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate cosine and HF loss weights from LoRA gradient norms")
    parser.add_argument("--study", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    study_path = Path(args.study).resolve()
    study = read_json(study_path)
    repo_root = (study_path.parent / study["repo_root"]).resolve()
    training = study["training"]
    calibration = study["calibration"]

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    calibration_directory = repo_root / study["output_root"] / "calibration" / stamp
    calibration_directory.mkdir(parents=True)
    raw_result_path = repo_root / calibration["result_file"]
    train_log_path = calibration_directory / "calibration.log"

    environment = os.environ.copy()
    environment.update(
        {
            "DEBUGPY": "0",
            "FLUX_RUN_VERSION": f"calibration-{stamp}",
            "FLUX_MODEL_SIZE": training["model_name"].removeprefix("flux-klein-base-"),
            "FLUX_GPU_IDS": training["gpu_ids"].split(",")[0],
            "FLUX_GLOBAL_BATCH_SIZE": "1",
            "FLUX_TRAIN_BATCH_SIZE": "1",
            "FLUX_MAX_TRAIN_STEPS": str(training["max_train_steps"]),
            "FLUX_TRAIN_SEED": str(training["train_seed"]),
            "FLUX_WEIGHTING_SCHEME": training["weighting_scheme"],
            "FLUX_CHECKPOINTING_STEPS": str(training["max_train_steps"] + 1),
            "FLUX_CACHE_LATENTS": "0",
            "FLUX_MSE_LOSS_WEIGHT": "1.0",
            "FLUX_COSINE_LOSS_WEIGHT": "0.0",
            "FLUX_HF_LOSS_WEIGHT": "0.0",
            "FLUX_TRAIN_OUTPUT_DIR": str(calibration_directory / "train"),
            "FLUX_LOG_DIR": str(calibration_directory / "launcher_logs"),
            "FLUX_LOG_FILE": str(calibration_directory / "launcher.log"),
            "FLUX_LOSS_GRADIENT_CALIBRATION_STEPS": str(calibration["microbatches"]),
            "FLUX_LOSS_GRADIENT_CALIBRATION_OUTPUT": str(raw_result_path),
        }
    )

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["bash", str(repo_root / training["shell_script"])],
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        print(f"Calibration failed; see {train_log_path}", file=sys.stderr)
        return 1
    if not raw_result_path.is_file():
        print(f"Calibration did not produce {raw_result_path}", file=sys.stderr)
        return 1

    raw_result = read_json(raw_result_path)
    expected_epsilon = float(calibration["gradient_norm_epsilon"])
    if raw_result.get("gradient_norm_epsilon") != expected_epsilon:
        raise ValueError("Calibration used a different gradient-norm epsilon")
    if raw_result.get("weighting_scheme") != training["weighting_scheme"]:
        raise ValueError("Calibration used a different timestep weighting scheme")
    cosine_unit = float(raw_result["cosine_unit_weight"])
    hf_unit = float(raw_result["hf_unit_weight"])
    if not math.isfinite(cosine_unit) or cosine_unit <= 0.0:
        raise ValueError(f"Invalid calibrated cosine unit weight: {cosine_unit}")
    if not math.isfinite(hf_unit) or hf_unit <= 0.0:
        raise ValueError(f"Invalid calibrated HF unit weight: {hf_unit}")

    study["calibration"]["cosine_unit_weight"] = cosine_unit
    study["calibration"]["hf_unit_weight"] = hf_unit
    study["calibration"]["raw_result"] = calibration["result_file"]
    write_json_atomic(study_path, study)

    protected_files = {
        relative_path: sha256_file(repo_root / relative_path) for relative_path in study["protected_files"]
    }
    input_tree_snapshots = snapshot_input_trees(study, repo_root)
    study_lock = {
        "schema_version": 1,
        "study_sha256": sha256_file(study_path),
        "protected_files": protected_files,
        "input_tree_snapshots": input_tree_snapshots,
    }
    write_json_atomic(repo_root / study["study_lock"], study_lock)

    print(f"cosine_unit_weight={cosine_unit:.12g}")
    print(f"hf_unit_weight={hf_unit:.12g}")
    print(f"study_lock={repo_root / study['study_lock']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
