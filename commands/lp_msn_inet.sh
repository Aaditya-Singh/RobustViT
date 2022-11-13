# TODO: Hardcoded model name
SS=subsets1
NIMGS=1
MODEL=MSN_VITB

python imagenet_finetune_tokencut.py \
  --port 1234 --data /srv/datasets/ImageNet/imagenet/train/ \
  --seg_data subsets/tokencut_${SS}/${NIMGS}imgs_class/ \
  --val_data /srv/datasets/ImageNet/imagenet/val/ \
  --experiment_folder logs/${MODEL}_LP/IN_${SS}_${NIMGS}imgs_class/ \
  --model_name deit_base --linear_eval --temperature 1.0 \
  --pretrained pretrained/msn/vitb16_600ep.pth.tar \
  --epochs 50 --batch-size 128 --lr 6.4 --wd 0.0 \
  --lambda_acc 1.0 --lambda_seg 0.0 --lambda_foreground 0.0 --lambda_background 0.0