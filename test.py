import os
import glob
import argparse
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
import cv2
import numpy as np
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
import insightface
from insightface.app import FaceAnalysis

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/flux-klein-base-4b")
parser.add_argument("--lora_path", type=str, default="/root/autodl-tmp/diffusers/output/Flux2_klein_base_4b_v1/Flux2_klein_base_4b_v1_000001500.safetensors")
args = parser.parse_args()

def image_to_tensor(image):
    """
    把 PIL 图像转成 LPIPS 需要的 tensor。
    LPIPS 要求数值范围在 [-1, 1]。
    """
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor

pipe = Flux2KleinPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(args.lora_path)
pipe.to("cuda")

control1_dir = "/root/autodl-tmp/diffusers/datasets/test_control1"
control2_dir = "/root/autodl-tmp/diffusers/datasets/test_control3"
target_dir = "/root/autodl-tmp/diffusers/datasets/test_target"
output_dir = "/root/autodl-tmp/diffusers/datasets/test_output"
os.makedirs(output_dir, exist_ok=True)

txt_files = sorted(glob.glob(os.path.join(control1_dir, "*.txt")))

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = lpips.LPIPS(net="vgg").to(device)
loss_fn.eval()

total_lpips = 0.0
total_arcface = 0.0
n_arcface = 0
fid = FrechetInceptionDistance(feature=2048).to(device)

face_app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

for txt_path in txt_files:
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    input_image_path = os.path.join(control1_dir, f"{stem}.png")
    mask_image_path = os.path.join(control2_dir, f"{stem}.png")
    target_image_path = os.path.join(target_dir, f"{stem}.png")
    output_path = os.path.join(output_dir, f"{stem}.png")

    with open(txt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    print(f"处理 {stem} ...")

    input_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("RGB")
    target_image = Image.open(target_image_path).convert("RGB")

    width, height = input_image.size
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        image=[input_image, mask_image],
        num_inference_steps=50,
        guidance_scale=4.0,
    ).images[0]

    image.save(output_path)
    print(f"已保存 {output_path}")

    target_image = target_image.resize(image.size, Image.LANCZOS)

    image_tensor = image_to_tensor(image).to(device)
    target_tensor = image_to_tensor(target_image).to(device)

    with torch.no_grad():
        score = loss_fn(image_tensor, target_tensor).item()

    total_lpips += score

    image_uint8 = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    target_uint8 = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).to(device)
    fid.update(target_uint8, real=True)
    fid.update(image_uint8, real=False)

    gen_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    target_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    gen_faces = face_app.get(gen_cv2)
    target_faces = face_app.get(target_cv2)
    if gen_faces and target_faces:
        gen_emb = gen_faces[0].embedding
        target_emb = target_faces[0].embedding
        cos_sim = np.dot(gen_emb, target_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(target_emb))
        total_arcface += cos_sim
        n_arcface += 1
        print(f"  ArcFace 余弦相似度: {cos_sim:.4f}")
    else:
        print(f"  ⚠ {stem}: face detection failed (gen={len(gen_faces)}, tgt={len(target_faces)}), skipping ArcFace")

avg_lpips = total_lpips / len(txt_files)
fid_score = fid.compute().item()
avg_arcface = total_arcface / n_arcface if n_arcface > 0 else 0.0

print("\n" + "=" * 40)
print(f"{'指标':<16}{'值':>20}")
print("-" * 40)
print(f"{'LPIPS ↓':<16}{avg_lpips:>20.4f}")
print(f"{'FID ↓':<16}{fid_score:>20.4f}")
print(f"{'ArcFace ↑':<16}{avg_arcface:>20.4f}")
print("=" * 40)

