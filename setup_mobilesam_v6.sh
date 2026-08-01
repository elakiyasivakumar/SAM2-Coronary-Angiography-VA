#!/bin/bash
# MobileSAM v6: GT loss + foreground-weighted encoder feature KD
# Same split/augmentation as v4 (ablation=1) + FeatureAdapter + teacher encoder runs each batch
# Tests whether matching student encoder features to teacher improves over GT-only (v4 = 0.806)
set -euo pipefail
LOG=/tmp/mobilesam_v6.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/mobilesam_v6_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== MobileSAM v6 (GT + feature KD, clean split) $(date) ==="

apt-get install -y -q git
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm
pip3 install -q git+https://github.com/ChaoningZhang/MobileSAM.git

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

mkdir -p /home/jupyter
ln -sfn /opt/MedSAM2 /home/jupyter/MedSAM2

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/CA-SAM2', filename='medsam2_arcade_v2.pt', local_dir='/home/jupyter')
print('Teacher checkpoint ready.')
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
export MODEL_VERSION=v6
export USE_FEAT_KD=1     # enable foreground-weighted encoder feature matching
export FEAT_KD_W=0.1     # feature loss weight (conservative start)

cd /opt/SAM2/distill
python3 distill_student.py --student mobilesam --ablation 1

gsutil cp /home/jupyter/mobilesam_abl1_v6.pt \
  gs://coronary-angio-v2/checkpoints/mobilesam_abl1_v6.pt
echo "Done $(date)"
