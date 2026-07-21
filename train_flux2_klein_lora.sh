#!/usr/bin/env bash
# FLUX.2 Klein LoRA 训练统一入口；修改下面配置即可切换版本、模型和 GPU。
set -euo pipefail
RUN_VERSION="v1"
MODEL_SIZE="4b"
GPU_IDS="0"
GLOBAL_BATCH_SIZE=2
TRAIN_BATCH_SIZE=1
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/root/autodl-tmp/models/flux-klein-base-${MODEL_SIZE}"
DATASET_DIR="${REPO_DIR}/datasets/train_target"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_VERSION}"
LOG_DIR="${REPO_DIR}/outputs/logs"
LOG_FILE="${LOG_DIR}/flux2-klein-${MODEL_SIZE}-${RUN_VERSION}-$(date -u +%Y%m%dT%H%M%SZ).log"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GLOBAL_BATCH_SIZE % (TRAIN_BATCH_SIZE * GPU_COUNT) != 0 )); then echo "GLOBAL_BATCH_SIZE 必须能被 TRAIN_BATCH_SIZE × GPU 数量整除" >&2; exit 1; fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (TRAIN_BATCH_SIZE * GPU_COUNT)))
ASPECT_RATIO_BUCKETS="$(DATASET_DIR="${REPO_DIR}/datasets/train_control1" python - <<'PY2'
from pathlib import Path
from PIL import Image
import os
files=sorted(Path(os.environ["DATASET_DIR"]).glob("*.png"))
if not files: raise SystemExit("条件图目录为空")
sizes={Image.open(p).size for p in files}
if any(w==h for w,h in sizes): raise SystemExit("不支持方图")
if any(w%8 or h%8 for w,h in sizes): raise SystemExit("所有条件图宽高必须能被 8 整除")
v={(h,w) for w,h in sizes if h>w}; q={(h,w) for w,h in sizes if w>h}
if len(v)!=1 or len(q)!=1: raise SystemExit("要求恰好一种竖图尺寸和一种横图尺寸")
print(";".join(f"{h},{w}" for h,w in sorted(v|q)))
PY2
)"
mkdir -p "${LOG_DIR}" "${TRAIN_OUTPUT_DIR}"; cd "${REPO_DIR}"
TRAIN_COMMAND=(
  accelerate launch
)
if [[ "${DEBUGPY:-0}" == "1" ]]; then
  TRAIN_COMMAND+=(
    -m debugpy
    --listen 127.0.0.1:5678
    --wait-for-client
  )
fi
TRAIN_COMMAND+=(
  examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py
  --pretrained_model_name_or_path="${MODEL_DIR}"
  --dataset_name="${DATASET_DIR}"
  --image_column=image
  --cond_image_column=cond_image
  --caption_column=caption
  --output_dir="${TRAIN_OUTPUT_DIR}"
  --aspect_ratio_buckets="${ASPECT_RATIO_BUCKETS}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  --max_train_steps=1500
  --rank=16
  --lora_alpha=16
  --learning_rate=1e-4
  --lr_scheduler=constant_with_warmup
  --lr_warmup_steps=50
  --optimizer=adamw
  --use_8bit_adam
  --mixed_precision=bf16
  --do_fp8_training
  --gradient_checkpointing
  --cache_latents
  --offload
  --allow_tf32
  --checkpointing_steps=100
  --checkpoints_total_limit=30
  --report_to=tensorboard
  --seed=42
)
if (( GPU_COUNT > 1 )); then TRAIN_COMMAND+=(--multi_gpu "--num_processes=${GPU_COUNT}"); fi
echo "GPU=${GPU_IDS}, buckets=${ASPECT_RATIO_BUCKETS}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "DDP 每张 GPU 保存完整模型，不会合并显存。"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
