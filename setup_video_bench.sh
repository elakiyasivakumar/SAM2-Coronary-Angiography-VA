#!/bin/bash
# MobileSAM v4 video inference benchmark on CPU-only 8GB-RAM VM
# Tests edge deployability: can v4 run real-time on cath-lab hardware?
# Uses ARCADE val images as proxy frames (real fluoroscopy would be a .mp4)
set -euo pipefail
LOG=/tmp/video_bench.log
exec > >(tee -a $LOG) 2>&1
trap 'gsutil cp $LOG gs://coronary-angio-v2/results/distillation/video_bench_run.log 2>/dev/null || true; shutdown -h now' EXIT

echo "=== MobileSAM v4 video inference benchmark $(date) ==="
echo "CPU: $(nproc) cores, RAM: $(free -h | awk '/^Mem:/{print $2}')"

apt-get install -y -q git
pip3 install -q scikit-image scipy opencv-python-headless huggingface_hub timm
pip3 install -q git+https://github.com/ChaoningZhang/MobileSAM.git

git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
pip3 install -q -e /opt/MedSAM2
git clone https://github.com/elakiyasivakumar/SAM2-Coronary-Angiography-VA.git /opt/SAM2

# Download v4 checkpoint from HuggingFace
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Elakiya17/medsam_distill_models', filename='mobilesam_v4.pt',
                local_dir='/home/jupyter')
print('v4 checkpoint ready:', __import__('os').path.getsize('/home/jupyter/mobilesam_v4.pt') // 1e6, 'MB')
"

# Use ARCADE val images as proxy fluoroscopy frames
mkdir -p /home/jupyter/frames
gsutil -m cp gs://coronary-angio-v2/datasets/arcade/val/images/* /home/jupyter/frames/

export PYTHONPATH=/opt/MedSAM2

echo ""
echo "--- CPU inference benchmark ---"
python3 /opt/SAM2/distill/infer_video.py \
  --ckpt /home/jupyter/mobilesam_v4.pt \
  --frames /home/jupyter/frames \
  --device cpu \
  --max_frames 200

echo ""
echo "Done $(date)"
