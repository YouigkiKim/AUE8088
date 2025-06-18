# !/bin/bash
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

conda activate aue8088

cd /home/yk/git/AUE8088

python train_simple.py \
    --img 640 \
    --batch-size 8 \
    --epochs 50 \
    --data data/kaist-rgbt.yaml \
    --cfg models/yolov5n_kaist-rgbt.yaml \
    --weights yolov5n.pt \
    --workers 4 \
    --name yolov5n-rgbt_loss_modified \
    --entity $WANDB_ENTITY \
    --rgbt \
    --single-cls
