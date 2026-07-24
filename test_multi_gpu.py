import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.multiprocessing as mp
from insightface.app import FaceAnalysis
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from diffusers import Flux2KleinPipeline


MODELS_ROOT = "/root/autodl-tmp/models"
LORA_ROOT = "/root/autodl-tmp/diffusers/output"

DEFAULT_MODEL_NAME = "flux-klein-base-4b"
DEFAULT_LORA_NAME = "v1"

CONTROL1_DIR = "/root/autodl-tmp/diffusers/datasets/test_control1"
TARGET_DIR = "/root/autodl-tmp/diffusers/datasets/test_target"
OUTPUT_DIR = "/root/autodl-tmp/diffusers/datasets/test_output"

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
ARCFACE_MISSING_POLICY = "minus_one"

# Metric roles:
# - Primary: LPIPS (lower is better)
# - Tie-break diagnostics: ArcFace and SSIM (higher is better)
# - Pixel-level diagnostic: PSNR (higher is better)
# - Distributional auxiliary: FID and KID (lower is better; exploratory for the current 10-image test set)


def image_to_tensor(image):
    """
    Convert a PIL image to the tensor format required by LPIPS.
    LPIPS expects values in [-1, 1].
    """
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor


def get_device(rank):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def get_face_embedding(face_app, image):
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_app.get(image_cv2)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    return face.embedding


def run_worker(rank, world_size, indexed_txt_files, model_path, lora_path, output_dir, seed):
    device = get_device(rank)
    local_items = indexed_txt_files[rank::world_size]

    print(
        f"[rank {rank}] processing {len(local_items)} prompt(s) on {device}",
        flush=True,
    )

    if len(local_items) == 0:
        return {"rank": rank, "samples": []}

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(lora_path)
    pipe.to(device)

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    face_providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    face_ctx_id = rank if device.type == "cuda" else -1
    face_app = FaceAnalysis(name="antelopev2", providers=face_providers)
    face_app.prepare(ctx_id=face_ctx_id, det_size=(640, 640))

    samples = []

    for sample_index, txt_path in local_items:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        input_image_path = os.path.join(CONTROL1_DIR, f"{stem}.png")
        target_image_path = os.path.join(TARGET_DIR, f"{stem}.png")
        output_path = os.path.join(output_dir, f"{stem}.png")

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"[rank {rank}] processing {stem} ...", flush=True)

        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        width, height = input_image.size
        sample_seed = seed + sample_index
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                image=[input_image],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]

        image.save(output_path)
        print(f"[rank {rank}] saved {output_path}", flush=True)

        target_image = target_image.resize(image.size, Image.LANCZOS)

        image_tensor = image_to_tensor(image).to(device)
        target_tensor = image_to_tensor(target_image).to(device)

        with torch.no_grad():
            lpips_score = loss_fn(image_tensor, target_tensor).item()
            image_01 = (image_tensor + 1.0) / 2.0
            target_01 = (target_tensor + 1.0) / 2.0
            ssim_score = (
                structural_similarity_index_measure(image_01, target_01, data_range=1.0, reduction="none")
                .mean()
                .item()
            )
            psnr_score = (
                peak_signal_noise_ratio(
                    image_01,
                    target_01,
                    data_range=1.0,
                    dim=(1, 2, 3),
                    reduction="none",
                )
                .mean()
                .item()
            )

        gen_emb = get_face_embedding(face_app, image)
        target_emb = get_face_embedding(face_app, target_image)
        generated_face_detected = gen_emb is not None
        target_face_detected = target_emb is not None
        if generated_face_detected and target_face_detected:
            arcface_score = float(np.dot(gen_emb, target_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(target_emb)))
        else:
            arcface_score = -1.0

        samples.append(
            {
                "index": sample_index,
                "stem": stem,
                "seed": sample_seed,
                "lpips": float(lpips_score),
                "arcface": arcface_score,
                "ssim": float(ssim_score),
                "psnr_db": float(psnr_score),
                "generated_face_detected": generated_face_detected,
                "target_face_detected": target_face_detected,
            }
        )

        print(f"[rank {rank}] ArcFace cosine similarity: {arcface_score:.4f}", flush=True)

    return {"rank": rank, "samples": samples}


def worker_entry(rank, world_size, indexed_txt_files, result_queue, model_path, lora_path, output_dir, seed):
    result_queue.put(run_worker(rank, world_size, indexed_txt_files, model_path, lora_path, output_dir, seed))


def compute_distribution_metrics(txt_files, device, output_dir):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    # Full-set KID avoids random subset selection; its one-subset std is not a confidence interval.
    # The raw unbiased estimate can be negative on this small test set.
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=1,
        subset_size=len(txt_files),
        normalize=False,
    ).to(device)

    for txt_path in txt_files:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        target_image_path = os.path.join(TARGET_DIR, f"{stem}.png")
        output_path = os.path.join(output_dir, f"{stem}.png")

        image = Image.open(output_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB").resize(image.size, Image.LANCZOS)

        image_uint8 = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
        target_uint8 = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(target_uint8, real=True)
        fid.update(image_uint8, real=False)
        kid.update(target_uint8, real=True)
        kid.update(image_uint8, real=False)

    kid_score, _ = kid.compute()
    return fid.compute().item(), kid_score.item()


def run_all_workers(world_size, indexed_txt_files, model_path, lora_path, output_dir, seed):
    if world_size == 1:
        return [run_worker(0, world_size, indexed_txt_files, model_path, lora_path, output_dir, seed)]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker_entry,
            args=(rank, world_size, indexed_txt_files, result_queue, model_path, lora_path, output_dir, seed),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker process(es) failed: exit codes {failed}")

    return [result_queue.get() for _ in processes]


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Multi-GPU Flux2Klein inference and evaluation")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Subdirectory under {MODELS_ROOT} (e.g. flux-klein-base-4b, flux-klein-base-9b)",
    )
    parser.add_argument(
        "--lora-name",
        default=DEFAULT_LORA_NAME,
        help=(
            f"Subdirectory under {LORA_ROOT} (expects <name>/<name>.safetensors inside). "
            "Ignored when --lora-path is set."
        ),
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Absolute path to a .safetensors LoRA file. Overrides --lora-name when set.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for deterministic per-sample inference.")
    parser.add_argument("--stems-file", default=None, help="UTF-8 file containing one sample stem per line.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for generated images.")
    parser.add_argument("--metrics-json", default=None, help="Optional machine-readable metrics output path.")

    return parser.parse_args(input_args)


def load_txt_files(stems_file):
    if stems_file is None:
        return sorted(glob.glob(os.path.join(CONTROL1_DIR, "*.txt")))

    stems = [
        line.strip()
        for line in Path(stems_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate stems in {stems_file}")
    stems = sorted(stems)
    txt_files = [os.path.join(CONTROL1_DIR, f"{stem}.txt") for stem in stems]
    missing = [path for path in txt_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing prompt files: {missing}")
    return txt_files


def main():
    args = parse_args()

    model_path = f"{MODELS_ROOT}/{args.model_name}"
    if args.lora_path:
        lora_path = args.lora_path
    else:
        lora_path = f"{LORA_ROOT}/{args.lora_name}/{args.lora_name}.safetensors"

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    txt_files = load_txt_files(args.stems_file)

    if len(txt_files) == 0:
        print(f"No txt files found in {CONTROL1_DIR}")
        return
    if len(txt_files) < 2:
        raise ValueError("FID/KID evaluation requires at least two samples")

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = max(gpu_count, 1)

    print(f"Found {len(txt_files)} prompts.")
    print(f"Detected {gpu_count} CUDA GPU(s). Starting {world_size} worker process(es).")

    indexed_txt_files = list(enumerate(txt_files))
    results = run_all_workers(world_size, indexed_txt_files, model_path, lora_path, output_dir, args.seed)
    samples = sorted(
        (sample for result in results for sample in result["samples"]),
        key=lambda sample: sample["index"],
    )
    total_count = len(samples)

    if total_count == 0:
        print("No prompts were processed.")
        return
    if [sample["index"] for sample in samples] != list(range(len(txt_files))):
        raise RuntimeError("Worker results do not match the requested sample indices")

    distribution_device = torch.device("cuda:0" if gpu_count > 0 else "cpu")
    fid_score, kid_score = compute_distribution_metrics(txt_files, distribution_device, output_dir)

    avg_lpips = math.fsum(sample["lpips"] for sample in samples) / total_count
    avg_arcface = math.fsum(sample["arcface"] for sample in samples) / total_count
    avg_ssim = math.fsum(sample["ssim"] for sample in samples) / total_count
    avg_psnr = math.fsum(sample["psnr_db"] for sample in samples) / total_count
    generated_face_count = sum(sample["generated_face_detected"] for sample in samples)
    target_face_count = sum(sample["target_face_detected"] for sample in samples)
    face_pair_count = sum(sample["generated_face_detected"] and sample["target_face_detected"] for sample in samples)

    metrics = {
        "lpips": float(avg_lpips),
        "arcface": float(avg_arcface),
        "ssim": float(avg_ssim),
        "psnr_db": float(avg_psnr),
        "fid": float(fid_score),
        "kid_raw": float(kid_score),
    }
    payload = {
        "schema_version": 1,
        "status": "ok",
        "model_path": os.path.abspath(model_path),
        "lora_path": os.path.abspath(lora_path),
        "output_dir": output_dir,
        "seed": args.seed,
        "seed_strategy": "base_plus_sorted_sample_index",
        "arcface_missing_policy": ARCFACE_MISSING_POLICY,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "gpu_count": gpu_count,
        "world_size": world_size,
        "expected_count": len(txt_files),
        "processed_count": total_count,
        "stems": [sample["stem"] for sample in samples],
        "face_detection": {
            "generated_count": generated_face_count,
            "generated_rate": generated_face_count / total_count,
            "target_count": target_face_count,
            "target_rate": target_face_count / total_count,
            "pair_count": face_pair_count,
            "pair_rate": face_pair_count / total_count,
        },
        "metrics": metrics,
        "samples": samples,
    }

    if args.metrics_json is not None:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(metrics_path)

    print("\n" + "=" * 40)
    print(f"{'Metric':<16}{'Value':>20}")
    print("-" * 40)
    print(f"{'LPIPS ↓':<16}{avg_lpips:>20.4f}")
    print(f"{'ArcFace ↑':<16}{avg_arcface:>20.4f}")
    print(f"{'SSIM ↑':<16}{avg_ssim:>20.4f}")
    print(f"{'PSNR ↑ (dB)':<16}{avg_psnr:>20.4f}")
    print(f"{'FID ↓':<16}{fid_score:>20.4f}")
    print(f"{'KID ↓ (raw)':<16}{kid_score:>20.6f}")
    print(f"{'Face pairs':<16}{f'{face_pair_count}/{total_count}':>20}")
    print("=" * 40)


if __name__ == "__main__":
    main()
