# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=4

# set the training args
torchrun --nproc_per_node 1 -m src.open_clip_train.main \
    --batch-size 128 \
    --train-data "/home/xshadow/Datasets" \
    --precision amp \
    --workers 8 \
    --report-to tensorboard \
    --save-frequency 1 \
    --dataset-type geolb \
    --warmup 1000 \
    --lr=1e-5 \
    --wd=0.01 \
    --lock-text \
    --epochs=16 \
    --siglip \
    --DOFA \
    --model GeoLB-ViT-B-16-SigLIP \
    --pretrained webli \
    --distill-model ViT-B-16-SigLIP \
    --distill-pretrained webli
