#!/bin/bash
set -euo pipefail
LOG=/tmp/repvit.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/repvit_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== RepViT-SAM distillation $(date) ==="

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
print('Checkpoint ready.')
"

mkdir -p /home/jupyter/arcade_train/images /home/jupyter/arcade_train/masks
mkdir -p /home/jupyter/arcade_val/images /home/jupyter/arcade_val/masks
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/images/* /home/jupyter/arcade_train/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/train/masks/* /home/jupyter/arcade_train/masks/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/arcade_val/images/
gsutil -m cp -r gs://coronary-angio-v2/datasets/arcade/val/masks/* /home/jupyter/arcade_val/masks/

export PYTHONPATH=/opt/MedSAM2
cd /opt/SAM2/distill
python3 distill_student.py --student repvitsam --ablation 4

gsutil cp /home/jupyter/repvitsam_abl4.pt gs://coronary-angio-v2/checkpoints/repvitsam_abl4.pt
echo "Done $(date)"
