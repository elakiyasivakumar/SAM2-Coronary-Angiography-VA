#!/usr/bin/env bash
# ==============================================================================
# DGX Headless Runner: Prompt-Aligned 1024x1024 SAM2 -> MobileSAM Distillation
# Budget: $0 (Runs on local/institutional DGX GPU server)
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Starting CA-SAM2 1024x1024 Distillation Pipeline on DGX "
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "=========================================================="

WORK_DIR="${WORK_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${WORK_DIR}/data}"
CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"

mkdir -p "${DATA_ROOT}" "${CKPT_DIR}" "${LOG_DIR}"

# 1. Python Environment Setup
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please load appropriate python/conda module on DGX."
    exit 1
fi

echo "--- Installing / Verifying Dependencies ---"
pip install -q --upgrade pip
pip install -q scikit-image scipy opencv-python-headless huggingface_hub timm pillow numpy torchvision

if [ ! -d "/opt/sam2_meta" ] && [ ! -d "${WORK_DIR}/sam2_meta" ]; then
    echo "Cloning Meta SAM2 repo..."
    git clone https://github.com/facebookresearch/sam2.git "${WORK_DIR}/sam2_meta"
    pip install -q -e "${WORK_DIR}/sam2_meta" --no-build-isolation
fi
export SAM2_REPO="${WORK_DIR}/sam2_meta"

if ! python -c "import mobile_sam" &>/dev/null; then
    echo "Installing MobileSAM..."
    pip install -q git+https://github.com/ChaoningZhang/MobileSAM.git
fi

# 2. Checkpoints Acquisition ($0 Free Sources via HuggingFace & GitHub)
echo "--- Downloading Checkpoints ---"
export MOBILE_CKPT="${CKPT_DIR}/mobile_sam.pt"
if [ ! -f "${MOBILE_CKPT}" ]; then
    echo "Fetching MobileSAM base weights..."
    wget -q -O "${MOBILE_CKPT}" https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
fi

export TEACHER_BASE_CKPT="${CKPT_DIR}/sam2.1_hiera_tiny.pt"
export TEACHER_CKPT="${CKPT_DIR}/sam2_fluoroscopy_1024.pt"

python3 -c "
from huggingface_hub import hf_hub_download
import os
ckpt_dir = '${CKPT_DIR}'
if not os.path.exists('${TEACHER_BASE_CKPT}'):
    print('Downloading base SAM2.1 Hiera-Tiny...')
    hf_hub_download(repo_id='facebook/sam2.1-hiera-tiny', filename='sam2.1_hiera_tiny.pt', local_dir=ckpt_dir)
if not os.path.exists('${TEACHER_CKPT}'):
    print('Downloading 1024 fluoroscopy teacher checkpoint...')
    hf_hub_download(repo_id='Elakiya17/fluoroscopy-sam2', filename='sam2_fluoroscopy_1024.pt', local_dir=ckpt_dir)
print('✓ Checkpoints verified.')
"

# 3. ARCADE Dataset Setup ($0 via Zenodo)
ARCADE_DIR="${DATA_ROOT}/arcade"
if [ ! -d "${DATA_ROOT}/arcade_train/images" ] || [ ! -d "${DATA_ROOT}/arcade_test/images" ]; then
    echo "--- Preparing ARCADE Dataset from Zenodo (Free) ---"
    mkdir -p "${ARCADE_DIR}"
    if [ ! -f "${DATA_ROOT}/arcade.zip" ]; then
        echo "Downloading ARCADE zip (~451 MB)..."
        wget -q --show-progress "https://zenodo.org/records/10390295/files/arcade.zip?download=1" -O "${DATA_ROOT}/arcade.zip"
    fi
    echo "Extracting ARCADE..."
    unzip -q -o "${DATA_ROOT}/arcade.zip" -d "${ARCADE_DIR}"

    echo "Converting COCO annotations to binary masks..."
    python3 -c "
import os, json, cv2, numpy as np
from PIL import Image

def convert_split(img_dir, ann_file, out_img_dir, out_mask_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    with open(ann_file) as f:
        coco = json.load(f)
    img_map = {im['id']: im for im in coco['images']}
    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)
    for img_id, im_info in img_map.items():
        src_im = os.path.join(img_dir, im_info['file_name'])
        if not os.path.exists(src_im): continue
        dest_im = os.path.join(out_img_dir, im_info['file_name'])
        Image.open(src_im).save(dest_im)
        mask = np.zeros((im_info['height'], im_info['width']), dtype=np.uint8)
        for ann in anns_by_img.get(img_id, []):
            for poly in ann.get('segmentation', []):
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
        dest_msk = os.path.join(out_mask_dir, os.path.splitext(im_info['file_name'])[0] + '.png')
        Image.fromarray(mask).save(dest_msk)

base = '${ARCADE_DIR}/syntax'
convert_split(f'{base}/train/images', f'{base}/train/annotations/train.json',
              '${DATA_ROOT}/arcade_train/images', '${DATA_ROOT}/arcade_train/masks')
convert_split(f'{base}/val/images', f'{base}/val/annotations/val.json',
              '${DATA_ROOT}/arcade_test/images', '${DATA_ROOT}/arcade_test/masks')
print('✓ ARCADE conversion complete.')
"
fi

export DATA_DIR="${DATA_ROOT}/arcade_train"
export TEST_DIR="${DATA_ROOT}/arcade_test"

# 4. Run Distillation Matrix
echo "=========================================================="
echo " Executing Distillation Experiments "
echo "=========================================================="

# Run Ablation 1 (GT Only baseline)
echo "--> Running Ablation 1: GT Only"
python3 "${WORK_DIR}/distill/distill_student_1024.py" \
    --ablation 1 --epochs 30 --batch_size 4 | tee "${LOG_DIR}/run_ablation_1.log"

# Run Ablation 2 (GT + Prompt-Aligned Response KD)
echo "--> Running Ablation 2: GT + Prompt-Aligned Response KD"
python3 "${WORK_DIR}/distill/distill_student_1024.py" \
    --ablation 2 --epochs 30 --batch_size 4 | tee "${LOG_DIR}/run_ablation_2.log"

# Run Ablation 4 (GT + Prompt-Aligned Response KD + clDice)
echo "--> Running Ablation 4: Full Prompt-Aligned KD + clDice"
python3 "${WORK_DIR}/distill/distill_student_1024.py" \
    --ablation 4 --epochs 30 --batch_size 4 | tee "${LOG_DIR}/run_ablation_4.log"

echo "=========================================================="
echo " All experiments finished successfully! "
echo " Logs stored in: ${LOG_DIR} "
echo "=========================================================="
