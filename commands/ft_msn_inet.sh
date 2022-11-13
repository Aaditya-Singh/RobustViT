# TODO: Hardcoded model name and assumes saved LP checkpoint
SS=subsets1
NIMGS=1
MODEL=MSN_VITB

python imagenet_finetune_tokencut.py \
  --port 1234 --data /srv/datasets/ImageNet/imagenet/train/ \
  --seg_data subsets/tokencut_${SS}/${NIMGS}imgs_class/ \
  --val_data /srv/datasets/ImageNet/imagenet/val/ \
  --experiment_folder logs/${MODEL}_FT/IN_${SS}_${NIMGS}imgs_class/ \
  --model_name deit_base --temperature 0.65 \
  --pretrained logs/${MODEL}_LP/IN_${SS}_${NIMGS}imgs_class/model_best.pth.tar \
  --epochs 50 --batch-size 8 --lr 8e-7 \
  --lambda_acc 0.2 --lambda_seg 0.8 --lambda_foreground 0.3 --lambda_background 2