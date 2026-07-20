import argparse
import glob
import os

import cv2
import lpips
import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import Flux2KleinPipeline
from insightface.app import FaceAnalysis
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance


MODELS_ROOT = "/root/autodl-tmp/models"
LORA_ROOT = "/root/autodl-tmp/diffusers/output"

DEFAULT_MODEL_NAME = "flux-klein-base-4b"
DEFAULT_LORA_NAME = "v1"

CONTROL1_DIR = "/root/autodl-tmp/diffusers/datasets/test_control1"
MASK_DIR = "/root/autodl-tmp/diffusers/datasets/test_control3"
TARGET_DIR = "/root/autodl-tmp/diffusers/datasets/test_target"
OUTPUT_DIR = "/root/autodl-tmp/diffusers/datasets/test_output"

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0


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
    return face_app.get(image_cv2)[0].embedding


def run_worker(rank, world_size, txt_files, model_path, lora_path):
    device = get_device(rank)
    local_txt_files = txt_files[rank::world_size]

    print(
        f"[rank {rank}] processing {len(local_txt_files)} prompt(s) on {device}",
        flush=True,
    )

    if len(local_txt_files) == 0:
        return {
            "rank": rank,
            "lpips": 0.0,
            "arcface": 0.0,
            "count": 0,
        }

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

    total_lpips = 0.0
    total_arcface = 0.0
    local_count = 0

    for txt_path in local_txt_files:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        input_image_path = os.path.join(CONTROL1_DIR, f"{stem}.png")
        mask_image_path = os.path.join(MASK_DIR, f"{stem}.png")
        target_image_path = os.path.join(TARGET_DIR, f"{stem}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{stem}.png")

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"[rank {rank}] processing {stem} ...", flush=True)

        input_image = Image.open(input_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        width, height = input_image.size
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                image=[input_image, mask_image],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

        image.save(output_path)
        print(f"[rank {rank}] saved {output_path}", flush=True)

        target_image = target_image.resize(image.size, Image.LANCZOS)

        image_tensor = image_to_tensor(image).to(device)
        target_tensor = image_to_tensor(target_image).to(device)

        with torch.no_grad():
            score = loss_fn(image_tensor, target_tensor).item()

        total_lpips += score

        gen_emb = get_face_embedding(face_app, image)
        target_emb = get_face_embedding(face_app, target_image)
        cos_sim = np.dot(gen_emb, target_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(target_emb))
        total_arcface += cos_sim
        local_count += 1

        print(f"[rank {rank}] ArcFace cosine similarity: {cos_sim:.4f}", flush=True)

    return {
        "rank": rank,
        "lpips": total_lpips,
        "arcface": total_arcface,
        "count": local_count,
    }


def worker_entry(rank, world_size, txt_files, result_queue, model_path, lora_path):
    result_queue.put(run_worker(rank, world_size, txt_files, model_path, lora_path))


def compute_fid(txt_files, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for txt_path in txt_files:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        target_image_path = os.path.join(TARGET_DIR, f"{stem}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{stem}.png")

        image = Image.open(output_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB").resize(image.size, Image.LANCZOS)

        image_uint8 = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
        target_uint8 = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(target_uint8, real=True)
        fid.update(image_uint8, real=False)

    return fid.compute().item()


def run_all_workers(world_size, txt_files, model_path, lora_path):
    if world_size == 1:
        return [run_worker(0, world_size, txt_files, model_path, lora_path)]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker_entry, args=(rank, world_size, txt_files, result_queue, model_path, lora_path))
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

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Flux2Klein inference and evaluation")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help=f"Subdirectory under {MODELS_ROOT} (e.g. flux-klein-base-4b, flux-klein-base-9b)",)
    parser.add_argument("--lora-name", default=DEFAULT_LORA_NAME,
                    help=f"Subdirectory under {LORA_ROOT} (expects <name>/<name>.safetensors inside)",)

    return parser.parse_args()

def main():
    args = parse_args()

    model_path = f"{MODELS_ROOT}/{args.model_name}"
    lora_path = f"{LORA_ROOT}/{args.lora_name}/{args.lora_name}.safetensors"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(CONTROL1_DIR, "*.txt")))

    if len(txt_files) == 0:
        print(f"No txt files found in {CONTROL1_DIR}")
        return

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = max(gpu_count, 1)

    print(f"Found {len(txt_files)} prompts.")
    print(f"Detected {gpu_count} CUDA GPU(s). Starting {world_size} worker process(es).")

    results = run_all_workers(world_size, txt_files, model_path, lora_path)
    total_count = sum(result["count"] for result in results)

    if total_count == 0:
        print("No prompts were processed.")
        return

    total_lpips = sum(result["lpips"] for result in results)
    total_arcface = sum(result["arcface"] for result in results)

    fid_device = torch.device("cuda:0" if gpu_count > 0 else "cpu")
    fid_score = compute_fid(txt_files, fid_device)

    avg_lpips = total_lpips / total_count
    avg_arcface = total_arcface / total_count

    print("\n" + "=" * 40)
    print(f"{'Metric':<16}{'Value':>20}")
    print("-" * 40)
    print(f"{'LPIPS':<16}{avg_lpips:>20.4f}")
    print(f"{'FID':<16}{fid_score:>20.4f}")
    print(f"{'ArcFace':<16}{avg_arcface:>20.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
