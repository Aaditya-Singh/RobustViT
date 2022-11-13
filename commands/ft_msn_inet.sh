SS=subsets1
NIMGS=1
MODEL=MSN_VITB_LP

python imagenet_finetune_tokencut.py \
  --port 1234 --data /srv/datasets/ImageNet/imagenet/train/ \
  --seg_data subsets/tokencut_${SS}/${NIMGS}imgs_class/ \
  --val_data /srv/datasets/ImageNet/imagenet/val/ \
  --experiment_folder logs/${MODEL}/IN_${SS}_${NIMGS}imgs_class/ \
  --model-name deit_base --pretrained pretrained/msn/vitb16_600ep.pth.tar \
  --epochs 50 --batch-size 8 --lr 3e-6 --wd 0.0 --temperature 1.0 \
  --lambda_acc 1.0 --lambda_seg 0.0 --lambda_foreground 0.0 --lambda_background 0.0

