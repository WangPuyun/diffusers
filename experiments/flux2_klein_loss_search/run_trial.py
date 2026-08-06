#!/usr/bin/env python

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


STUDY_KEYS = {
    "schema_version",
    "study_id",
    "repo_root",
    "training",
    "evaluation",
    "calibration",
    "search",
    "objective",
    "output_root",
    "study_lock",
    "input_trees",
    "protected_files",
}
CANDIDATE_KEYS = {"mse_weight", "cosine_weight", "hf_weight"}
METRIC_KEYS = {"lpips", "arcface", "ssim", "psnr_db", "fid", "kid_raw"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def installed_version(*distribution_names):
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def runtime_signature():
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": installed_version("torch"),
        "torchao": installed_version("torchao"),
        "torchvision": installed_version("torchvision"),
        "diffusers": installed_version("diffusers"),
        "accelerate": installed_version("accelerate"),
        "transformers": installed_version("transformers"),
        "peft": installed_version("peft"),
        "bitsandbytes": installed_version("bitsandbytes"),
        "torchmetrics": installed_version("torchmetrics"),
        "lpips": installed_version("lpips"),
        "insightface": installed_version("insightface"),
        "onnxruntime": installed_version("onnxruntime-gpu", "onnxruntime"),
        "opencv": installed_version("opencv-python-headless", "opencv-python"),
        "gpu": gpu_query.stdout.strip() if gpu_query.returncode == 0 else "unavailable",
    }


def resolve_input_tree(repo_root, tree):
    if set(tree) != {"path", "extensions"}:
        raise ValueError("Each input_trees entry must contain exactly path and extensions")
    configured_path = tree["path"]
    extensions = tree["extensions"]
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("Input-tree paths must be non-empty strings")
    if not isinstance(extensions, list) or any(
        not isinstance(extension, str) or not extension.startswith(".") for extension in extensions
    ):
        raise ValueError(f"Invalid extensions for input tree {configured_path}")
    tree_path = Path(configured_path)
    if not tree_path.is_absolute():
        tree_path = repo_root / tree_path
    tree_path = tree_path.resolve()
    if not tree_path.is_dir():
        raise ValueError(f"Input tree is missing: {tree_path}")
    return configured_path, tree_path, sorted(set(extensions))


def iter_input_files(tree_path, extensions):
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(tree_path)
        if "__pycache__" in relative_path.parts or file_path.suffix in {".lock", ".pyc", ".tmp"}:
            continue
        if extensions and file_path.suffix not in extensions:
            continue
        yield relative_path, file_path


def snapshot_input_trees(study, repo_root):
    snapshots = {}
    for tree in study["input_trees"]:
        configured_path, tree_path, extensions = resolve_input_tree(repo_root, tree)
        if configured_path in snapshots:
            raise ValueError(f"Duplicate input tree: {configured_path}")
        files = {}
        for relative_path, file_path in iter_input_files(tree_path, extensions):
            stat = file_path.stat()
            files[str(relative_path)] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(file_path),
            }
        if not files:
            raise ValueError(f"Input tree has no selected files: {tree_path}")
        snapshots[configured_path] = {
            "resolved_path": str(tree_path),
            "extensions": extensions,
            "files": files,
        }
    return snapshots


def validate_input_tree_snapshots(study, repo_root, snapshots):
    configured_paths = [tree.get("path") for tree in study["input_trees"]]
    if len(configured_paths) != len(set(configured_paths)) or set(configured_paths) != set(snapshots):
        raise ValueError("study_lock.json has a different input-tree set")

    digests = {}
    for tree in study["input_trees"]:
        configured_path, tree_path, extensions = resolve_input_tree(repo_root, tree)
        snapshot = snapshots[configured_path]
        if snapshot.get("resolved_path") != str(tree_path) or snapshot.get("extensions") != extensions:
            raise ValueError(f"Input-tree definition changed: {configured_path}")
        expected_files = snapshot.get("files", {})
        current_files = {}
        for relative_path, file_path in iter_input_files(tree_path, extensions):
            stat = file_path.stat()
            current_files[str(relative_path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        expected_metadata = {
            relative_path: {"size": metadata.get("size"), "mtime_ns": metadata.get("mtime_ns")}
            for relative_path, metadata in expected_files.items()
        }
        if current_files != expected_metadata:
            raise ValueError(f"Input tree changed after calibration: {configured_path}")
        if any(not isinstance(metadata.get("sha256"), str) for metadata in expected_files.values()):
            raise ValueError(f"Input-tree snapshot is missing hashes: {configured_path}")
        digests[configured_path] = sha256_bytes(canonical_json(snapshot))
    return digests


def load_context(study_path, candidate_path):
    study_path = Path(study_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    study = read_json(study_path)
    candidate = read_json(candidate_path)

    if set(study) != STUDY_KEYS or study["schema_version"] != 1:
        raise ValueError("study.json does not match schema version 1")
    if set(candidate) != CANDIDATE_KEYS:
        raise ValueError(f"candidate.json must contain exactly {sorted(CANDIDATE_KEYS)}")

    repo_root = (study_path.parent / study["repo_root"]).resolve()
    if not (repo_root / ".git").exists():
        raise ValueError(f"Resolved repository root is invalid: {repo_root}")

    mse_weight = finite_number(candidate["mse_weight"], "mse_weight")
    cosine_weight = finite_number(candidate["cosine_weight"], "cosine_weight")
    hf_weight = finite_number(candidate["hf_weight"], "hf_weight")
    expected_mse = finite_number(study["search"]["mse_weight"], "search.mse_weight")
    if mse_weight != expected_mse:
        raise ValueError(f"mse_weight is frozen at {expected_mse}")
    if cosine_weight < 0.0 or hf_weight < 0.0:
        raise ValueError("Auxiliary loss weights must be non-negative")

    cosine_unit = study["calibration"]["cosine_unit_weight"]
    hf_unit = study["calibration"]["hf_unit_weight"]
    if cosine_unit is None and hf_unit is None:
        cosine_unit = 1.0
        hf_unit = 1.0
    elif cosine_unit is None or hf_unit is None:
        raise ValueError("Cosine and HF calibration weights must either both be set or both be null")
    else:
        cosine_unit = finite_number(cosine_unit, "calibration.cosine_unit_weight")
        hf_unit = finite_number(hf_unit, "calibration.hf_unit_weight")
    if cosine_unit <= 0.0 or hf_unit <= 0.0:
        raise ValueError("Calibrated unit weights must be positive")

    relative_cosine = cosine_weight / cosine_unit
    relative_hf = hf_weight / hf_unit
    allowed_strengths = [
        finite_number(value, "search.relative_strengths") for value in study["search"]["relative_strengths"]
    ]
    cosine_matches = [
        value for value in allowed_strengths if math.isclose(relative_cosine, value, rel_tol=1e-9, abs_tol=1e-12)
    ]
    hf_matches = [
        value for value in allowed_strengths if math.isclose(relative_hf, value, rel_tol=1e-9, abs_tol=1e-12)
    ]
    if len(cosine_matches) != 1:
        raise ValueError(f"cosine relative strength {relative_cosine} is outside the registered grid")
    if len(hf_matches) != 1:
        raise ValueError(f"HF relative strength {relative_hf} is outside the registered grid")
    relative_cosine = cosine_matches[0]
    relative_hf = hf_matches[0]
    max_budget = finite_number(study["search"]["max_total_relative_strength"], "search.max_total_relative_strength")
    if relative_cosine + relative_hf > max_budget + 1e-9:
        raise ValueError("Combined relative auxiliary strength exceeds the registered budget")

    study_digest = sha256_file(study_path)
    lock_path = repo_root / study["study_lock"]
    study_lock_present = lock_path.is_file()
    protected_files = {}
    input_tree_digests = {}
    if study_lock_present:
        study_lock = read_json(lock_path)
        if study_lock.get("study_sha256") != study_digest:
            raise ValueError("study.json does not match its frozen digest")
        expected_protected = study_lock.get("protected_files", {})
        if set(expected_protected) != set(study["protected_files"]):
            raise ValueError("study_lock.json has a different protected-file set")
        for relative_path in study["protected_files"]:
            file_path = repo_root / relative_path
            digest = sha256_file(file_path)
            if expected_protected[relative_path] != digest:
                raise ValueError(f"Protected file changed: {relative_path}")
            protected_files[relative_path] = digest
        input_tree_digests = validate_input_tree_snapshots(
            study,
            repo_root,
            study_lock.get("input_tree_snapshots", {}),
        )
    else:
        for relative_path in study["protected_files"]:
            protected_files[relative_path] = sha256_file(repo_root / relative_path)

    stems_path = repo_root / study["evaluation"]["stems_file"]
    stems = sorted(
        line.strip()
        for line in stems_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(stems) != len(set(stems)) or len(stems) < 2:
        raise ValueError("Evaluation stems must be unique and contain at least two samples")

    normalized_candidate = {
        "mse_weight": mse_weight,
        "cosine_weight": cosine_unit * relative_cosine,
        "hf_weight": hf_unit * relative_hf,
    }
    signature = runtime_signature()
    fingerprint_payload = {
        "study_sha256": study_digest,
        "candidate": normalized_candidate,
        "relative_strengths": {"cosine": relative_cosine, "hf": relative_hf},
        "protected_files": protected_files,
        "input_trees": input_tree_digests,
        "runtime": signature,
    }
    fingerprint = sha256_bytes(canonical_json(fingerprint_payload))
    output_root = repo_root / study["output_root"]

    return {
        "study": study,
        "candidate": normalized_candidate,
        "relative_strengths": {"cosine": relative_cosine, "hf": relative_hf},
        "study_path": study_path,
        "candidate_path": candidate_path,
        "repo_root": repo_root,
        "study_sha256": study_digest,
        "study_lock_present": study_lock_present,
        "protected_files": protected_files,
        "input_tree_digests": input_tree_digests,
        "runtime": signature,
        "fingerprint": fingerprint,
        "output_root": output_root,
        "stems": stems,
    }


def validate_metrics(metrics, context):
    study = context["study"]
    if metrics.get("schema_version") != 1 or metrics.get("status") != "ok":
        raise ValueError("metrics.json does not match schema version 1")
    if metrics.get("expected_count") != len(context["stems"]):
        raise ValueError("metrics.json has the wrong expected sample count")
    if metrics.get("processed_count") != len(context["stems"]):
        raise ValueError("metrics.json is incomplete")
    if metrics.get("stems") != context["stems"]:
        raise ValueError("metrics.json stems do not match the frozen manifest")
    if metrics.get("seed") != study["evaluation"]["seed"]:
        raise ValueError("metrics.json uses the wrong inference seed")
    if metrics.get("seed_strategy") != "base_plus_sorted_sample_index":
        raise ValueError("metrics.json uses the wrong seed strategy")
    if metrics.get("arcface_missing_policy") != "minus_one":
        raise ValueError("metrics.json uses the wrong missing-face policy")
    if metrics.get("num_inference_steps") != study["evaluation"]["num_inference_steps"]:
        raise ValueError("metrics.json uses the wrong inference-step count")
    if metrics.get("guidance_scale") != study["evaluation"]["guidance_scale"]:
        raise ValueError("metrics.json uses the wrong guidance scale")
    if set(metrics.get("metrics", {})) != METRIC_KEYS:
        raise ValueError("metrics.json has an unexpected metric set")
    for name, value in metrics["metrics"].items():
        finite_number(value, f"metrics.{name}")

    samples = metrics.get("samples", [])
    if len(samples) != len(context["stems"]):
        raise ValueError("metrics.json has the wrong per-sample count")
    for index, (sample, stem) in enumerate(zip(samples, context["stems"])):
        if sample.get("index") != index or sample.get("stem") != stem:
            raise ValueError("metrics.json per-sample ordering is invalid")
        if sample.get("seed") != study["evaluation"]["seed"] + index:
            raise ValueError("metrics.json per-sample seed mapping is invalid")
        for name in ("lpips", "arcface", "ssim", "psnr_db"):
            finite_number(sample.get(name), f"samples[{index}].{name}")
        generated_face = sample.get("generated_face_detected")
        target_face = sample.get("target_face_detected")
        if type(generated_face) is not bool or type(target_face) is not bool:
            raise ValueError("metrics.json face-detection flags must be booleans")
        if not (generated_face and target_face) and sample["arcface"] != -1.0:
            raise ValueError("Missing-face samples must use the registered ArcFace sentinel")

    aggregate_sample_metrics = {
        "lpips": "lpips",
        "arcface": "arcface",
        "ssim": "ssim",
        "psnr_db": "psnr_db",
    }
    for aggregate_name, sample_name in aggregate_sample_metrics.items():
        expected_value = math.fsum(float(sample[sample_name]) for sample in samples) / len(samples)
        if not math.isclose(
            float(metrics["metrics"][aggregate_name]),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"metrics.{aggregate_name} does not match the per-sample mean")

    face_detection = metrics.get("face_detection", {})
    expected_face_keys = {
        "generated_count",
        "generated_rate",
        "target_count",
        "target_rate",
        "pair_count",
        "pair_rate",
    }
    if set(face_detection) != expected_face_keys:
        raise ValueError("metrics.json has an unexpected face-detection summary")
    generated_count = sum(sample["generated_face_detected"] for sample in samples)
    target_count = sum(sample["target_face_detected"] for sample in samples)
    pair_count = sum(sample["generated_face_detected"] and sample["target_face_detected"] for sample in samples)
    expected_counts = {
        "generated_count": generated_count,
        "target_count": target_count,
        "pair_count": pair_count,
    }
    for name, value in expected_counts.items():
        if face_detection[name] != value:
            raise ValueError(f"face_detection.{name} does not match the per-sample flags")
        rate_name = name.replace("count", "rate")
        rate = finite_number(face_detection[rate_name], f"face_detection.{rate_name}")
        if not math.isclose(rate, value / len(samples), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"face_detection.{rate_name} does not match its count")


def compute_score(metrics, tie_break_weight):
    values = metrics["metrics"]
    arcface = min(1.0, max(-1.0, float(values["arcface"])))
    ssim = min(1.0, max(-1.0, float(values["ssim"])))
    tie_break = 0.5 * ((1.0 - arcface) / 2.0 + (1.0 - ssim) / 2.0)
    score = float(values["lpips"]) + tie_break_weight * tie_break
    if not math.isfinite(score):
        raise ValueError("Objective score is not finite")
    return score


def expected_artifact_names(context):
    training = context["study"]["training"]
    return {
        f"train/{training['lora_filename']}",
        "metrics.json",
        "train.log",
        "eval.log",
        *{f"images/{stem}.png" for stem in context["stems"]},
    }


def validate_result(result_path, context):
    result = read_json(result_path)
    if result.get("schema_version") != 1 or result.get("status") != "success":
        raise ValueError("Cached result is not successful")
    if result.get("fingerprint") != context["fingerprint"]:
        raise ValueError("Cached result has the wrong fingerprint")
    if result.get("study_sha256") != context["study_sha256"]:
        raise ValueError("Cached result has the wrong study digest")
    if result.get("candidate") != context["candidate"]:
        raise ValueError("Cached result has different weights")
    if result.get("relative_strengths") != context["relative_strengths"]:
        raise ValueError("Cached result has different relative strengths")
    if result.get("runtime") != context["runtime"]:
        raise ValueError("Cached result has a different runtime signature")
    score = finite_number(result.get("score"), "result.score")

    expected_trial_directory = (context["output_root"] / "trials" / context["fingerprint"]).resolve()
    trial_directory = (context["repo_root"] / result.get("trial_directory", "")).resolve()
    if trial_directory != expected_trial_directory:
        raise ValueError("Cached result points to the wrong trial directory")
    artifacts = result.get("artifacts", {})
    if set(artifacts) != expected_artifact_names(context):
        raise ValueError("Cached result has an incomplete or unexpected artifact set")
    for relative_path, expected_digest in artifacts.items():
        artifact_path = trial_directory / relative_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_digest:
            raise ValueError(f"Cached artifact is missing or changed: {relative_path}")

    metrics_path = trial_directory / "metrics.json"
    metrics = read_json(metrics_path)
    validate_metrics(metrics, context)
    expected_lora_path = trial_directory / "train" / context["study"]["training"]["lora_filename"]
    if Path(metrics.get("lora_path", "")).resolve() != expected_lora_path:
        raise ValueError("Cached metrics point to the wrong LoRA")
    if Path(metrics.get("output_dir", "")).resolve() != trial_directory / "images":
        raise ValueError("Cached metrics point to the wrong image directory")
    if sha256_file(metrics_path) != result.get("metrics_sha256"):
        raise ValueError("Cached metrics digest is invalid")
    if result.get("metrics") != metrics["metrics"] or result.get("face_detection") != metrics["face_detection"]:
        raise ValueError("Cached result summary does not match metrics.json")
    expected_score = compute_score(metrics, context["study"]["objective"]["tie_break_weight"])
    if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Cached score does not match metrics.json")
    return result


def run_trial(context):
    study = context["study"]
    fingerprint = context["fingerprint"]
    output_root = context["output_root"]
    trials_root = output_root / "trials"
    failures_root = output_root / "failures"
    working_root = output_root / "working"
    locks_root = output_root / "locks"
    for directory in (trials_root, failures_root, working_root, locks_root):
        directory.mkdir(parents=True, exist_ok=True)

    final_directory = trials_root / fingerprint
    final_result_path = final_directory / "result.json"
    latest_result_path = output_root / "latest_result.json"
    if final_result_path.is_file():
        result = validate_result(final_result_path, context)
        write_json_atomic(latest_result_path, result)
        print(f"cache hit: {fingerprint}", file=sys.stderr)
        return float(result["score"])
    if final_directory.exists():
        raise ValueError(f"Incomplete immutable trial directory exists: {final_directory}")

    lock_path = locks_root / f"{fingerprint}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Trial lock already exists: {lock_path}") from error
    try:
        lock_record = {
            "pid": os.getpid(),
            "host": platform.node(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fingerprint": fingerprint,
        }
        os.write(lock_descriptor, canonical_json(lock_record) + b"\n")
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(lock_descriptor)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    working_directory = working_root / f"{fingerprint}-{stamp}-{os.getpid()}"

    try:
        working_directory.mkdir()
        pending_result = {
            "schema_version": 1,
            "status": "running",
            "fingerprint": fingerprint,
            "study_sha256": context["study_sha256"],
            "candidate": context["candidate"],
        }
        write_json_atomic(latest_result_path, pending_result)

        training = study["training"]
        train_directory = working_directory / "train"
        train_log = working_directory / "train.log"
        environment = os.environ.copy()
        environment.update(
            {
                "DEBUGPY": "0",
                "FLUX_RUN_VERSION": fingerprint[:12],
                "FLUX_MODEL_SIZE": training["model_name"].removeprefix("flux-klein-base-"),
                "FLUX_GPU_IDS": training["gpu_ids"],
                "FLUX_GLOBAL_BATCH_SIZE": str(training["global_batch_size"]),
                "FLUX_TRAIN_BATCH_SIZE": str(training["train_batch_size"]),
                "FLUX_MAX_TRAIN_STEPS": str(training["max_train_steps"]),
                "FLUX_TRAIN_SEED": str(training["train_seed"]),
                "FLUX_WEIGHTING_SCHEME": training["weighting_scheme"],
                "FLUX_CHECKPOINTING_STEPS": str(training["checkpointing_steps"]),
                "FLUX_CACHE_LATENTS": "1",
                "FLUX_MSE_LOSS_WEIGHT": repr(context["candidate"]["mse_weight"]),
                "FLUX_COSINE_LOSS_WEIGHT": repr(context["candidate"]["cosine_weight"]),
                "FLUX_HF_LOSS_WEIGHT": repr(context["candidate"]["hf_weight"]),
                "FLUX_TRAIN_OUTPUT_DIR": str(train_directory),
                "FLUX_LOG_DIR": str(working_directory / "launcher_logs"),
            }
        )
        environment.pop("FLUX_LOSS_GRADIENT_CALIBRATION_OUTPUT", None)
        environment.pop("FLUX_LOSS_GRADIENT_CALIBRATION_STEPS", None)

        with train_log.open("w", encoding="utf-8") as log_handle:
            train_process = subprocess.run(
                ["bash", str(context["repo_root"] / training["shell_script"])],
                cwd=context["repo_root"],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if train_process.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {train_process.returncode}")

        lora_path = train_directory / training["lora_filename"]
        if not lora_path.is_file():
            raise FileNotFoundError(f"Training did not produce {lora_path}")

        evaluation = study["evaluation"]
        metrics_path = working_directory / "metrics.json"
        images_directory = working_directory / "images"
        eval_log = working_directory / "eval.log"
        eval_environment = os.environ.copy()
        eval_environment["CUDA_VISIBLE_DEVICES"] = training["gpu_ids"]
        eval_command = [
            sys.executable,
            str(context["repo_root"] / evaluation["script"]),
            "--model-name",
            evaluation["model_name"],
            "--lora-path",
            str(lora_path),
            "--seed",
            str(evaluation["seed"]),
            "--stems-file",
            str(context["repo_root"] / evaluation["stems_file"]),
            "--output-dir",
            str(images_directory),
            "--metrics-json",
            str(metrics_path),
        ]
        with eval_log.open("w", encoding="utf-8") as log_handle:
            eval_process = subprocess.run(
                eval_command,
                cwd=context["repo_root"],
                env=eval_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if eval_process.returncode != 0:
            raise RuntimeError(f"Evaluation failed with exit code {eval_process.returncode}")

        if sha256_file(context["study_path"]) != context["study_sha256"]:
            raise ValueError("study.json changed while the trial was running")
        current_lock_path = context["repo_root"] / study["study_lock"]
        if current_lock_path.is_file() != context["study_lock_present"]:
            raise ValueError("Study-lock state changed while the trial was running")
        for relative_path, expected_digest in context["protected_files"].items():
            if sha256_file(context["repo_root"] / relative_path) != expected_digest:
                raise ValueError(f"Protected file changed while the trial was running: {relative_path}")
        if context["study_lock_present"]:
            current_study_lock = read_json(current_lock_path)
            if current_study_lock.get("study_sha256") != context["study_sha256"]:
                raise ValueError("Study lock changed while the trial was running")
            current_input_digests = validate_input_tree_snapshots(
                study,
                context["repo_root"],
                current_study_lock.get("input_tree_snapshots", {}),
            )
            if current_input_digests != context["input_tree_digests"]:
                raise ValueError("Frozen inputs changed while the trial was running")

        metrics = read_json(metrics_path)
        # The evaluator runs in a temporary directory that is atomically promoted
        # after validation. Persist the paths that will be valid in the archive.
        metrics["lora_path"] = str(final_directory / "train" / training["lora_filename"])
        metrics["output_dir"] = str(final_directory / "images")
        write_json_atomic(metrics_path, metrics)
        validate_metrics(metrics, context)
        tie_break_weight = finite_number(study["objective"]["tie_break_weight"], "tie_break_weight")
        score = compute_score(metrics, tie_break_weight)

        artifact_paths = [
            lora_path,
            metrics_path,
            train_log,
            eval_log,
            *[images_directory / f"{stem}.png" for stem in context["stems"]],
        ]
        artifacts = {str(path.relative_to(working_directory)): sha256_file(path) for path in artifact_paths}
        trial_directory = final_directory.relative_to(context["repo_root"])
        result = {
            "schema_version": 1,
            "status": "success",
            "fingerprint": fingerprint,
            "study_sha256": context["study_sha256"],
            "candidate": context["candidate"],
            "relative_strengths": context["relative_strengths"],
            "runtime": context["runtime"],
            "score": score,
            "metrics": metrics["metrics"],
            "face_detection": metrics["face_detection"],
            "metrics_sha256": sha256_file(metrics_path),
            "trial_directory": str(trial_directory),
            "artifacts": artifacts,
        }
        write_json_atomic(working_directory / "result.json", result)
        working_directory.replace(final_directory)
        write_json_atomic(latest_result_path, result)
        return score
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failure",
            "fingerprint": fingerprint,
            "study_sha256": context["study_sha256"],
            "candidate": context["candidate"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        if working_directory.exists():
            write_json_atomic(working_directory / "failure.json", failure)
            failure_directory = failures_root / f"{fingerprint}-{stamp}-{os.getpid()}"
            working_directory.replace(failure_directory)
        write_json_atomic(latest_result_path, failure)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run one immutable Flux2 Klein loss-weight trial")
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        context = load_context(args.study, args.candidate)
        score = run_trial(context)
    except Exception as error:
        print(f"run_trial failed: {error}", file=sys.stderr)
        return 1
    print(format(score, ".17g"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
