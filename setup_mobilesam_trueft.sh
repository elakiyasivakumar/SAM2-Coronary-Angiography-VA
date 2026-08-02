#!/bin/bash
# MobileSAM true fine-tune: SA-1B encoder + SA-1B decoder (no teacher transplant), ARCADE GT only
# Isolates decoder transplant contribution: compare against v4 (0.806) which has partial CA-SAM2 decoder
# TRANSPLANT_DECODER=0 keeps vanilla SA-1B weights in mask_decoder + prompt_encoder
set -euo pipefail
LOG=/tmp/mobilesam_trueft.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/mobilesam_trueft_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== MobileSAM true fine-tune (SA-1B only, no transplant, GT only) $(date) ==="

apt-get install -y -q git python3-pip
pip3 install -q -U pip setuptools wheel
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm
git clone https://github.com/ChaoningZhang/MobileSAM.git /opt/MobileSAM
pip3 install -q -e /opt/MobileSAM --no-build-isolation

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

# Teacher checkpoint NOT needed (no transplant, no KD), but code path still references it
# during build_mobilesam when teacher_ckpt is passed; we skip loading it by setting TRANSPLANT_DECODER=0
# Still need it to exist on disk so the code path doesn't crash on os.path.exists check.
# Workaround: download it (small overhead) so the train() preamble doesn't fail.
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt', local_dir='/home/jupyter')
print('Teacher checkpoint downloaded (unused — TRANSPLANT_DECODER=0).')
"

mkdir -p /home/jupyter/arcade_train/images /home/jupyter/arcade_train/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/images/* /home/jupyter/arcade_train/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/masks/*  /home/jupyter/arcade_train/masks/

mkdir -p /home/jupyter/arcade_test/images /home/jupyter/arcade_test/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_test/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/*  /home/jupyter/arcade_test/masks/

export PYTHONPATH=/opt/MedSAM2
export MAX_EPOCHS=30
export ES_PATIENCE=10
export BATCH_SIZE=4
export MODEL_VERSION=trueft
export TRANSPLANT_DECODER=0   # key flag: keep vanilla SA-1B decoder, no teacher weights

cd /opt/SAM2/distill
python3 distill_student.py --student mobilesam --ablation 1

gsutil cp /home/jupyter/mobilesam_abl1_trueft.pt \
  gs://coronary-angio-v2/checkpoints/mobilesam_abl1_trueft.pt
echo "Done $(date)"
